import deeplake
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ["ACTIVELOOP_TOKEN"] = ""

ds = deeplake.load("hub://activeloop/kth-actions")
# dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
dataloader = ds.tensorflow()

for i in range(3):
    print(ds[i])
