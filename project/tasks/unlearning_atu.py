from omegaconf import DictConfig
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from project.eval import eval_llm
from hydra.utils import instantiate
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from typing import Literal, Optional
import copy
from project.utils.mean_pool import mean_pooling_reference_encoder
from project.utils import get_logger, log_hyperparameters
from lightning.pytorch.loggers import Logger
from project.utils.callbacks import get_default_callbacks
from project.utils.get_data_root import get_data_root
import os
import random
from hydra.core.hydra_config import HydraConfig

log = get_logger()


class UnlearningATU:
    def __init__(
        self,
        global_config: DictConfig,
        target_id: str,
        target_name: str,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        logger: Logger,
        baseline_mmlu: float = None,
        **kwargs,
    ):
        self.global_config = global_config
        self.task_config = global_config.task
        self.target_id = target_id
        self.target_name = target_name
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.logger = logger
        self.baseline_mmlu = baseline_mmlu

        # Build a unique save directory using Hydra's output dir to avoid collisions
        run_name = self.global_config.wandb.get("name", "default")
        hydra_run_dir = HydraConfig.get().run.dir
        self.save_dir = os.path.join(hydra_run_dir, "checkpoints", run_name, self.target_id)
        os.makedirs(self.save_dir, exist_ok=True)
        log.info(f"Checkpoint directory: {self.save_dir}")

    def unlearn(self) -> float:
        """Run the full unlearning pipeline. Returns baseline_mmlu for GUR tracking."""
        log.info("Task: unlearning_atu")

        log.info("Validating config")
        for stage in self.task_config.stages:
            assert stage["type"] in ["training", "unlearning"], f"Invalid stage: {stage['type']}"
            assert stage["steps"] > 0, f"Steps must be greater than 0: {stage['steps']}"
            if stage["type"] == "unlearning":
                assert stage["threshold"] is not None, "Threshold must be set for unlearning stage"

        log.info("Instantiating text encoder")
        text_encoder, text_encoder_tokenizer = instantiate(self.task_config.text_encoder)

        log.info("Instantiating embedding prediction model")
        log.info(
            f"Pre-trained model hidden size: {self.pre_trained_llm.config.hidden_size}"
        )
        log.info(f"Text encoder hidden size: {text_encoder.config.hidden_size}")
        embedding_prediction_model = instantiate(
            self.task_config.embedding_prediction_model,
            input_dim=self.pre_trained_llm.config.hidden_size,
            output_dim=text_encoder.config.hidden_size,
        )

        log_hyperparameters(
            self.logger,
            self.global_config,
            [
                ("pre_trained_llm", self.pre_trained_llm),
                ("text_encoder", text_encoder),
                ("embedding_prediction_model", embedding_prediction_model),
            ],
        )
        
        log.info("Loading training data")
        other_target_ids = []
        subfolders = [f for f in os.listdir(get_data_root()) if f != self.target_id]
        other_target_ids = subfolders[:self.task_config.num_other_targets]
        random.shuffle(other_target_ids)

        # tokenizer_variant is stashed on model.config by load_pre_trained_llm.
        # Falls back to "phi" for backwards compatibility with the Phi setup.
        tokenizer_variant = getattr(
            self.pre_trained_llm.config, "tokenizer_variant", "phi"
        )

        # Map target_id -> display name so the subject-mask regex can localise
        # the target in each sample's raw text. Training uses multiple targets
        # (self + distractors); unlearning uses only self.
        def _name_for(tid: str) -> str:
            return tid.split("_", 1)[1].replace("_", " ") if "_" in tid else tid

        training_target_names = {
            tid: _name_for(tid) for tid in [self.target_id] + other_target_ids
        }
        training_target_names[self.target_id] = self.target_name

        subject_mask_window = int(self.task_config.get("subject_mask_window", 0))
        training_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id] + other_target_ids,
            tokenizer_variant=tokenizer_variant,
            target_names=training_target_names,
            subject_mask_window=subject_mask_window,
        )
        training_datamodule.prepare_data()
        training_datamodule.setup("train")

        log.info("Loading unlearning data")
        unlearning_datamodule = instantiate(
            self.task_config.unlearning_data,
            primary_tokenizer=self.pre_trained_llm_tokenizer,
            secondary_tokenizer=text_encoder_tokenizer,
            target_ids=[self.target_id],
            tokenizer_variant=tokenizer_variant,
            target_names={self.target_id: self.target_name},
            subject_mask_window=subject_mask_window,
        )
        unlearning_datamodule.prepare_data()
        unlearning_datamodule.setup("train")

        log.info("Instantiating UnlearningATU")
        checkpoint_interval = self.task_config.get("checkpoint_interval", None)
        task = UnlearningATUTrainingModule(
            embedding_prediction_model=embedding_prediction_model,
            pre_trained_llm=self.pre_trained_llm,
            pre_trained_llm_tokenizer=self.pre_trained_llm_tokenizer,
            text_encoder=text_encoder,
            text_encoder_tokenizer=text_encoder_tokenizer,
            unlearning_target=self.target_name,
            save_dir=self.save_dir,
            checkpoint_interval=checkpoint_interval,
            **self.task_config.training_module,
        )

        log.info("Instantiating trainer")
        trainer = Trainer(
            **self.global_config.trainer,
            callbacks=get_default_callbacks(),
            logger=self.logger,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_checkpointing=False,
        )

        metric_prefix = f"target/{self.target_id}/"

        log.info("Starting initial evaluation!")
        baseline_mmlu = self.baseline_mmlu
        if self.global_config.skip_initial_eval:
            log.info("Skipping initial evaluation!")
        else:
            results = eval_llm(
                self.pre_trained_llm,
                self.pre_trained_llm_tokenizer,
                self.target_id,
                trainer.strategy.root_device,
                0,
                metric_prefix=metric_prefix,
            )
            trainer.logger.log_metrics(results)
            if baseline_mmlu is None:
                baseline_mmlu = results.get(f"{metric_prefix}eval/utility/gen")
            log.info(f"Baseline MMLU: {baseline_mmlu}")

        log.info("Starting training!")

        stage1_ckpt = self.task_config.get("stage1_checkpoint", None)
        if stage1_ckpt:
            log.info(f"Resuming from Stage 1 checkpoint: {stage1_ckpt}")
            task.pre_trained_llm.load_state_dict(
                torch.load(f"{stage1_ckpt}/pre_trained_llm.pt", map_location="cpu")
            )
            task.embedding_prediction_model.load_state_dict(
                torch.load(f"{stage1_ckpt}/embedding_prediction_model.pt", map_location="cpu")
            )
            self.task_config.stages = [s for s in self.task_config.stages if s["type"] != "training"]
            log.info("Stage 1 skipped, resuming from Stage 2")

        for idx, stage in enumerate(self.task_config.stages):
            log.info(
                f"Starting stage {idx + 1} ({stage['type']}) of {len(self.task_config.stages)}"
            )
            new_max_steps = (
                stage["steps"] if idx == 0 else stage["steps"] + trainer.max_steps
            )
            log.info(f"Setting max steps to {new_max_steps}")
            trainer.fit_loop.epoch_loop.max_steps = new_max_steps
            task.update_stage(stage["type"])
            if stage["type"] == "training":
                trainer.fit(task, datamodule=training_datamodule)
            elif stage["type"] == "unlearning":
                task.update_unlearning_threshold(stage["threshold"])
                trainer.fit(task, datamodule=unlearning_datamodule)
            else:
                raise ValueError(f"Invalid stage: {stage['type']}")
            log.info(f"Stage {idx + 1} ({stage['type']}) completed!")

            is_last_stage = idx == len(self.task_config.stages) - 1
            if is_last_stage:
                torch.save(task.pre_trained_llm.state_dict(), f"{self.save_dir}/unlearned_pre_trained_llm.pt")
                torch.save(task.embedding_prediction_model.state_dict(), f"{self.save_dir}/unlearned_embedding_prediction_model.pt")
            else:
                torch.save(task.pre_trained_llm.state_dict(), f"{self.save_dir}/pre_trained_llm.pt")
                torch.save(task.embedding_prediction_model.state_dict(), f"{self.save_dir}/embedding_prediction_model.pt")
            # Write a small metadata file so it's clear what stage was last saved
            with open(f"{self.save_dir}/last_stage.txt", "w") as f:
                f.write(f"stage {idx + 1}/{len(self.task_config.stages)} ({stage['type']})\n")
            log.info(f"Stage {idx + 1} weights saved to {self.save_dir}")

            if stage["type"] == "unlearning":
                log.info("Starting testing!")
                results = eval_llm(
                    self.pre_trained_llm,
                    self.pre_trained_llm_tokenizer,
                    self.target_id,
                    device=trainer.strategy.root_device,
                    stage_number=idx + 1,
                    baseline_mmlu=baseline_mmlu,
                    metric_prefix=metric_prefix,
                )
                trainer.logger.log_metrics(results)
        log.info("Unlearning complete!")
        return baseline_mmlu


class UnlearningATUTrainingModule(pl.LightningModule):
    def __init__(
        self,
        embedding_prediction_model: nn.Module,
        pre_trained_llm: AutoModelForCausalLM,
        pre_trained_llm_tokenizer: AutoTokenizer,
        text_encoder: AutoModel,
        text_encoder_tokenizer: AutoTokenizer,
        unlearning_target: str,
        training_warmup_steps: int,
        training_lr: float,
        training_weight_decay: float,
        unlearning_lr: float,
        unlearning_weight_decay: float,
        pretrained_model_hook_layer: int,
        clip_grad_norm: float,
        save_dir: str = None,
        checkpoint_interval: int = None,
        stage: Literal["training", "unlearning"] = "training",
        require_subject_mask: bool = True,
        kl_retain_weight: float = 0.0,
        disable_grad_checkpointing_on_unlearn: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=("embedding_prediction_model", "pre_trained_llm", "text_encoder")
        )
        self.unlearning_similarity_threshold = None
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        self.embedding_prediction_model = embedding_prediction_model
        self.pre_trained_llm = pre_trained_llm
        self.pre_trained_llm_tokenizer = pre_trained_llm_tokenizer
        self.text_encoder = text_encoder
        self.text_encoder_tokenizer = text_encoder_tokenizer
        self.unlearning_target = unlearning_target
        self.stage = stage
        self.require_subject_mask = require_subject_mask
        self.kl_retain_weight = float(kl_retain_weight)
        self.disable_grad_checkpointing_on_unlearn = disable_grad_checkpointing_on_unlearn
        # Snapshot of the student weights at the moment we entered unlearning,
        # used as the KL-retain teacher on non-subject positions. Built lazily
        # on the first unlearning step (lives on the same device as the student,
        # in bf16 to halve VRAM — on H200 this is cheap).
        self._kl_ref_model: Optional[nn.Module] = None
        self._empty_mask_skip_count = 0
        self.update_stage(stage)
        self.automatic_optimization = False

        log.info(f"Unlearning target: {unlearning_target}, computing embedding...")
        unlearning_target_tokens = text_encoder_tokenizer.encode(
            unlearning_target, add_special_tokens=True
        )
        unlearning_target_tokens = torch.tensor(
            unlearning_target_tokens, device=self.device
        ).unsqueeze(0)  # (1, seq_len)
        self.unlearning_target_embedding = mean_pooling_reference_encoder(
            text_encoder(
                input_ids=unlearning_target_tokens,
                output_hidden_states=True,
            ),
            attention_mask=torch.ones_like(unlearning_target_tokens, device=self.device),
        )[0]

    def on_fit_start(self):
        self.unlearning_target_embedding = self.unlearning_target_embedding.to(
            self.device
        )

    def _disable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def _enable_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def _unfreeze_hooked_layer_only(self):
        """Freeze the entire pre_trained_llm, then unfreeze only the decoder
        block whose *output* is `hidden_states[hook_layer]`.

        Rationale: hidden_states[k] is the output of transformer block k-1
        (index 0 is the embedding). So the block to edit is
        `model.model.layers[hook_layer - 1]`. On MoE models this confines
        updates to a single block's experts + router, instead of silently
        updating every expert in the stack.
        """
        hook_layer = self.hparams.pretrained_model_hook_layer
        assert hook_layer >= 1, (
            f"pretrained_model_hook_layer must be >= 1 to map to a decoder "
            f"block; got {hook_layer}"
        )

        # Freeze everything first
        self._disable_grad(self.pre_trained_llm)

        # Resolve the layer list (standard HF causal-LM path)
        layers = getattr(
            getattr(self.pre_trained_llm, "model", None), "layers", None
        )
        if layers is None:
            raise RuntimeError(
                "Could not locate `pre_trained_llm.model.layers` for layer-"
                "scoped unfreezing. Check model architecture."
            )
        block_idx = hook_layer - 1
        if not 0 <= block_idx < len(layers):
            raise IndexError(
                f"hook_layer={hook_layer} out of range for {len(layers)} "
                f"decoder blocks (valid: 1..{len(layers)})"
            )

        target_block = layers[block_idx]
        self._enable_grad(target_block)

        trainable = sum(
            p.numel() for p in self.pre_trained_llm.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.pre_trained_llm.parameters())
        log.info(
            f"Layer-scoped unfreeze: block {block_idx} "
            f"(hidden_states[{hook_layer}]) -> {trainable:,}/{total:,} "
            f"params trainable ({100 * trainable / total:.2f}%)"
        )

    def update_stage(self, stage: Literal["training", "unlearning"]):
        log.info(f"Updating stage to {stage}")
        self.stage = stage
        if self.stage == "training":
            self._disable_grad(self.text_encoder)
            self._disable_grad(self.pre_trained_llm)
            self._enable_grad(self.embedding_prediction_model)
            self.pre_trained_llm.eval()
            self.text_encoder.eval()
            self.embedding_prediction_model.train()
            # Re-enable grad checkpointing in case we turned it off for unlearn.
            if hasattr(self.pre_trained_llm, "gradient_checkpointing_enable"):
                try:
                    self.pre_trained_llm.gradient_checkpointing_enable()
                except Exception:
                    pass
        elif self.stage == "unlearning":
            self._disable_grad(self.text_encoder)
            self._enable_grad(self.embedding_prediction_model)
            # MoE-safe: only unfreeze the single decoder block the causal
            # trace picked. Avoids silently updating untouched experts.
            self._unfreeze_hooked_layer_only()
            self.pre_trained_llm.train()
            self.embedding_prediction_model.eval()
            self.text_encoder.eval()
            if self.disable_grad_checkpointing_on_unlearn and hasattr(
                self.pre_trained_llm, "gradient_checkpointing_disable"
            ):
                log.info("Disabling gradient checkpointing for unlearn stage (H200 VRAM)")
                try:
                    self.pre_trained_llm.gradient_checkpointing_disable()
                except Exception as e:
                    log.warning(f"Could not disable gradient checkpointing: {e}")
            # Snapshot KL-retain teacher now so non-subject positions are anchored
            # to the pre-unlearn student, not the original pretrained weights —
            # preserves the alignment progress from stage 1.
            if self.kl_retain_weight > 0 and self._kl_ref_model is None:
                log.info(
                    f"Snapshotting KL-retain reference (weight={self.kl_retain_weight})"
                )
                # deepcopy on a device_map="auto" / accelerate-hooked model can
                # be fragile; go through a CPU state_dict round-trip to avoid
                # copying accelerate hooks and to cap peak VRAM at snapshot.
                try:
                    ref_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.pre_trained_llm.state_dict().items()
                    }
                    ref = copy.deepcopy(self.pre_trained_llm).to("cpu")
                    ref.load_state_dict(ref_state)
                    ref = ref.to(self.pre_trained_llm.device)
                except Exception as e:
                    log.warning(
                        f"CPU-roundtrip ref snapshot failed ({e}); "
                        f"falling back to in-place deepcopy."
                    )
                    ref = copy.deepcopy(self.pre_trained_llm)
                for p in ref.parameters():
                    p.requires_grad = False
                ref.eval()
                if hasattr(ref, "gradient_checkpointing_disable"):
                    try:
                        ref.gradient_checkpointing_disable()
                    except Exception:
                        pass
                self._kl_ref_model = ref
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def train(self, mode=True):
        if mode:
            super().train()
            if self.stage == "training":
                self.text_encoder.eval()
                self.pre_trained_llm.eval()
            elif self.stage == "unlearning":
                self.text_encoder.eval()
                self.embedding_prediction_model.eval()
        else:
            super().train(False)
        return self

    def update_unlearning_threshold(self, threshold: float):
        log.info(f"Updating unlearning threshold to {threshold}")
        self.unlearning_similarity_threshold = threshold

    def training_step(self, batch, batch_idx):
        input_ids = batch["primary_input_ids"]  # shape (batch_size, max_length)
        context_windows = batch[
            "secondary_context_windows"
        ]  # shape (batch_size, max_length, context_window_length)
        has_full_window = batch["has_full_window"]  # shape (batch_size, max_length)
        attention_mask = batch["attention_mask"]  # shape (batch_size, max_length)

        batch_size, seq_len, window_len = context_windows.shape

        opt_list = self.optimizers()

        if self.stage == "training":
            # Forward pass through the reference encoder to get the target embeddings
            with torch.no_grad():
                reference_outputs = self.text_encoder(
                    input_ids=context_windows.view(-1, context_windows.size(-1)),
                    output_hidden_states=True,
                )
                attention_mask_enc = torch.ones_like(
                    context_windows.view(-1, context_windows.size(-1))
                )
                target_embeddings = mean_pooling_reference_encoder(
                    reference_outputs, attention_mask_enc
                ).view(batch_size, seq_len, -1)

            # Forward pass through the pretrained model
            with torch.no_grad():
                pretrained_outputs = self.pre_trained_llm(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                )
                hidden_states = pretrained_outputs.hidden_states[
                    self.hparams.pretrained_model_hook_layer
                ]

            # Cast to the embedding prediction model's dtype. The LLM runs in
            # bfloat16 (Qwen3.5), but the embedding prediction model is
            # float32, and nn.Linear rejects mixed dtypes.
            hidden_states = hidden_states.to(
                dtype=next(self.embedding_prediction_model.parameters()).dtype
            )

            # Forward pass through embedding prediction model
            outputs = self.embedding_prediction_model(hidden_states)
            loss = -torch.nn.functional.cosine_similarity(
                outputs, target_embeddings, dim=-1
            ) # shape (batch_size, max_length)
            loss = loss * has_full_window
            loss = loss.sum() / (has_full_window.sum() + 1e-8)
            assert not torch.isnan(loss), "Loss is NaN"

            opt_list[0].zero_grad()
            self.manual_backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.embedding_prediction_model.parameters(),
                max_norm=self.hparams.clip_grad_norm,
            )
            self.log("train/grad_norm_embedding_clip", grad_norm, batch_size=batch_size)
            opt_list[0].step()

            if batch_idx == 0:
                assert not torch.all(outputs == 0), "Outputs are all zero"
                assert not torch.all(target_embeddings == 0), (
                    "Target embeddings are all zero"
                )

        elif self.stage == "unlearning":
            # Single student forward: we need hidden_states at the hook layer
            # for the forget loss, and (when KL-retain is on) logits for the
            # retain KL. Doing both in one forward halves student activation
            # memory vs. two separate forwards.
            pretrained_outputs = self.pre_trained_llm(
                input_ids=input_ids,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
            hidden_states = pretrained_outputs.hidden_states[
                self.hparams.pretrained_model_hook_layer
            ]
            student_logits = (
                pretrained_outputs.logits if self.kl_retain_weight > 0 else None
            )

            # Cast to the embedding prediction model's dtype (see training
            # stage above). Keep grad flow back into the LLM — `.to()`
            # preserves autograd when it's just a dtype change.
            hidden_states = hidden_states.to(
                dtype=next(self.embedding_prediction_model.parameters()).dtype
            )

            # Forward pass through embedding prediction model
            outputs = self.embedding_prediction_model(
                hidden_states
            )  # shape (batch_size, seq_len, emb_dim)

            assert self.unlearning_similarity_threshold is not None, (
                "Unlearning similarity threshold must be set before unlearning stage"
            )

            # fp32 cosine + hinge: bf16 cos-sim has ~1e-2 error, which swallows
            # the signal around the typical 0.2-0.3 threshold.
            cos = torch.nn.functional.cosine_similarity(
                outputs.float(),
                self.unlearning_target_embedding.unsqueeze(0).unsqueeze(0).float(),
                dim=-1,
            )  # (B, T)
            per_pos_loss = torch.nn.functional.relu(
                cos - self.unlearning_similarity_threshold
            )

            # Masked mean over (subject_mask AND attention_mask). Falls back to
            # the full sequence only when require_subject_mask is False.
            subject_mask = batch.get("subject_mask")
            if subject_mask is None and self.require_subject_mask:
                raise RuntimeError(
                    "require_subject_mask=True but batch has no 'subject_mask'. "
                    "Check the datamodule wired target_name through."
                )
            if subject_mask is None:
                mask = attention_mask.to(per_pos_loss.dtype)
            else:
                mask = (subject_mask * attention_mask).to(per_pos_loss.dtype)

            local_num = (per_pos_loss * mask).sum()
            local_den = mask.sum()

            # Log coverage stats (per-batch sanity — if mostly zero we have a
            # no-op loss, per the expert's mandatory sanity check).
            has_any = (subject_mask.sum(dim=1) > 0).float() if subject_mask is not None else torch.ones(batch_size, device=per_pos_loss.device)
            self.log("train/subject_mask_coverage_frac", has_any.mean(), batch_size=batch_size)
            self.log("train/subject_mask_tokens_per_seq", (mask.sum(dim=1).mean() if mask.dim() == 2 else local_den / max(1, batch_size)), batch_size=batch_size)

            # DDP-correct reduction: with world_size W ranks, DDP averages each
            # parameter's gradient by 1/W. For the true global mean we want
            #   grad[W avg of local_num / global_den]  ==  grad[(sum local_num) / global_den].
            # So each rank divides its local_num by (global_den / W).
            # Rank-sync global_den via all_reduce so the skip decision is
            # identical on every rank (otherwise DDP deadlocks at backward).
            world_size = self.trainer.world_size if self.trainer is not None else 1
            if world_size > 1:
                from torch import distributed as dist
                global_den_t = local_den.detach().clone()
                dist.all_reduce(global_den_t, op=dist.ReduceOp.SUM)
                global_den = global_den_t.item()
            else:
                global_den = local_den.item()

            skip_unlearn = global_den <= 0
            if skip_unlearn:
                self._empty_mask_skip_count += 1
                self.log("train/unlearn_skipped_empty_mask", 1.0, batch_size=batch_size)
                forget_loss = None
            else:
                # Per-rank scaling that yields the true global mean after
                # DDP's 1/W gradient averaging.
                forget_loss = local_num * world_size / global_den

            # Allow a KL-retain-only step if the forget signal is empty:
            # even with no subject tokens, anchoring non-subject positions
            # to the pre-unlearn ref is a valid update.
            total_loss = forget_loss if forget_loss is not None else None

            # KL retain on non-subject, non-pad positions keeps the rest of the
            # distribution anchored to the pre-unlearn reference, preserving
            # MMLU and other capabilities. Opt-in via kl_retain_weight > 0.
            if self.kl_retain_weight > 0 and self._kl_ref_model is not None:
                retain_mask = attention_mask
                if subject_mask is not None:
                    retain_mask = retain_mask * (1 - subject_mask)
                retain_mask = retain_mask.to(per_pos_loss.dtype)
                retain_den = retain_mask.sum().clamp(min=1.0)

                with torch.no_grad():
                    ref_logits = self._kl_ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                # student_logits was produced by the single forward above.
                # Forward KL(ref || student) on retained positions, fp32 for stability.
                logp_student = torch.nn.functional.log_softmax(student_logits.float(), dim=-1)
                p_ref = torch.nn.functional.softmax(ref_logits.float(), dim=-1)
                per_pos_kl = (p_ref * (p_ref.clamp_min(1e-12).log() - logp_student)).sum(-1)
                kl_loss = (per_pos_kl * retain_mask).sum() / retain_den
                weighted_kl = self.kl_retain_weight * kl_loss
                total_loss = weighted_kl if total_loss is None else total_loss + weighted_kl
                self.log("train/kl_retain", kl_loss, batch_size=batch_size)

            if total_loss is None:
                # Every rank arrived at skip_unlearn=True via the same
                # all-reduced global_den, so this is DDP-safe.
                loss = torch.zeros((), device=input_ids.device)
            else:
                assert not torch.isnan(total_loss), "Loss is NaN"
                opt_list[1].zero_grad()
                self.manual_backward(total_loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.pre_trained_llm.parameters(),
                    max_norm=self.hparams.clip_grad_norm,
                )
                self.log("train/grad_norm_pre_clip", grad_norm, batch_size=batch_size)
                opt_list[1].step()
                loss = total_loss
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        self.log("train/loss", loss.mean(), batch_size=batch_size)
        if self.stage == "training":
            self.log("train/training_loss", loss.mean(), batch_size=batch_size)
        else:
            self.log("train/unlearning_loss", loss.mean(), batch_size=batch_size)
            self.log("train/unlearning_threshold", self.unlearning_similarity_threshold)

        # Interval checkpointing
        if (
            self.save_dir is not None
            and self.checkpoint_interval is not None
            and self.global_step % self.checkpoint_interval == 0
            and self.global_step > 0
        ):
            ckpt_dir = os.path.join(self.save_dir, f"step_{self.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.pre_trained_llm.state_dict(), f"{ckpt_dir}/pre_trained_llm.pt")
            torch.save(self.embedding_prediction_model.state_dict(), f"{ckpt_dir}/embedding_prediction_model.pt")
            log.info(f"Interval checkpoint saved to {ckpt_dir}")

        return {"loss": loss}

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.embedding_prediction_model.parameters(),
                lr=self.hparams.training_lr,
                weight_decay=self.hparams.training_weight_decay,
            ),
            torch.optim.SGD(
                self.pre_trained_llm.parameters(),
                lr=self.hparams.unlearning_lr,
                weight_decay=self.hparams.unlearning_weight_decay,
            ),
        ]
