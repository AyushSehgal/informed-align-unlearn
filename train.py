#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings
import torch
import wandb
from hydra.utils import instantiate, get_class
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from project.eval import eval_llm

from project.utils import (
    filter_device_available,
    get_logger,
    print_config,
    set_seed,
)

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The '\w+_dataloader' does not have many workers",
#     module="lightning",
# )
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    filter_device_available
)


log = get_logger()


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id



def _run_task(config, target_id, target_name, pre_trained_llm, pre_trained_llm_tokenizer, logger, baseline_mmlu=None, skip_all_evals=False):
    log.info("Running task")
    task_class = get_class(config.task._target_)
    task = task_class(
        config,
        target_id=target_id,
        target_name=target_name,
        pre_trained_llm=pre_trained_llm,
        pre_trained_llm_tokenizer=pre_trained_llm_tokenizer,
        logger=logger,
        baseline_mmlu=baseline_mmlu,
        skip_all_evals=skip_all_evals,
    )
    return task.unlearn()


def _eval_targets(pre_trained_llm, pre_trained_llm_tokenizer, target_ids, device, logger, baseline_mmlu_per_target, metric_prefix_fmt, stage_number):
    """Evaluate USR/APR/GUR for each target_id, logging under metric_prefix_fmt.format(target_id)."""
    for target_id in target_ids:
        log.info(f"Evaluating {target_id} (prefix={metric_prefix_fmt.format(target_id)})")
        results = eval_llm(
            pre_trained_llm,
            pre_trained_llm_tokenizer,
            target_id,
            device=device,
            stage_number=stage_number,
            baseline_mmlu=baseline_mmlu_per_target.get(target_id),
            metric_prefix=metric_prefix_fmt.format(target_id),
            fast=True,
        )
        logger.log_metrics(results)


# @hydra.main(config_path="config", config_name="train", version_base=None)
# @print_exceptions
def train(config: DictConfig):
    print(f"Running training on {socket.gethostname()}")
    rng = set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Instantiating pre-trained model")
    pre_trained_llm, pre_trained_llm_tokenizer = instantiate(config.pre_trained_llm)

    chain = config.get("unlearning_chain", None)

    if chain:
        chain_slug = "-".join(f"{step.target}L{step.layer}" for step in chain)
        with open_dict(config):
            config.wandb.name = f"atu-chain-{chain_slug}-phi3.5-mini"

    log.info("Instantiating logger")
    logger = instantiate(
        config.wandb,
        _target_="lightning.pytorch.loggers.WandbLogger",
        resume=(config.wandb.mode == "online") and "allow",
        log_model=True,
    )
    if chain:
        log.info(f"Running chained unlearning with {len(chain)} steps")
        all_target_ids = [step.target for step in chain]

        # Need a device reference for standalone evals — spin up a minimal trainer
        from project.utils.callbacks import get_default_callbacks
        _trainer = Trainer(
            **config.trainer,
            callbacks=get_default_callbacks(),
            logger=logger,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_checkpointing=False,
        )
        device = _trainer.strategy.root_device

        # Pre-chain eval: USR/APR/GUR for every target on the original model
        log.info("Pre-chain evaluation across all targets")
        baseline_mmlu_per_target: dict[str, float] = {}
        for target_id in all_target_ids:
            results = eval_llm(
                pre_trained_llm, pre_trained_llm_tokenizer, target_id,
                device=device, stage_number=0,
                metric_prefix=f"pre_chain/{target_id}/",
                fast=True,
            )
            logger.log_metrics(results)
            mmlu = results.get(f"pre_chain/{target_id}/eval/utility/gen")
            if mmlu is not None:
                baseline_mmlu_per_target[target_id] = mmlu

        # Run each unlearning step, skipping all intermediate evals
        for step_idx, step in enumerate(chain):
            log.info(f"Chain step {step_idx + 1}/{len(chain)}: target={step.target}, layer={step.layer}")
            with open_dict(config):
                config.unlearning_target = step.target
                config.task.training_module.pretrained_model_hook_layer = int(step.layer)
            target_id = step.target
            target_name = target_id.split("_", 1)[1].replace("_", " ")
            _run_task(
                config, target_id, target_name,
                pre_trained_llm, pre_trained_llm_tokenizer, logger,
                skip_all_evals=True,
            )

        # Post-chain eval: USR/APR/GUR for every target on the final model
        log.info("Post-chain evaluation across all targets")
        _eval_targets(
            pre_trained_llm, pre_trained_llm_tokenizer,
            all_target_ids, device, logger,
            baseline_mmlu_per_target=baseline_mmlu_per_target,
            metric_prefix_fmt="post_chain/{}/",
            stage_number=len(chain),
        )
    else:
        target_id: str = config.unlearning_target
        target_name = target_id.split("_", 1)[1].replace("_", " ")
        _run_task(config, target_id, target_name, pre_trained_llm, pre_trained_llm_tokenizer, logger)

    wandb.finish()


if __name__ == "__main__":
    print("Use launch_training.py to run this script")
#     main()
