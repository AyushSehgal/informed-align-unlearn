## Fixes 
-    np.random.seed(int(npg_ss.generate_state(1, np.uint32)))
+    np.random.seed(int(npg_ss.generate_state(1, np.uint32)[0]))

pytorch-lightning==2.6.1
torch==2.8.0
torchmetrics==1.9.0
torchvision==0.23.0
