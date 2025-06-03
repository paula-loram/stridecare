import deeplake
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0ODk1MTUxNiwiZXhwIjoxNzgwNDg3NDk1fQ.eyJpZCI6ImphbW15bmluamE5NSIsIm9yZ19pZCI6ImphbW15bmluamE5NSJ9.thC8cfrTgZJSN_1D4-rnd_dLjBTMPanrHshiAUavzXX9OyB94AofbbFIAnNOalsYVRhsGw87C9I0j9C_vZ4X_A"


ds = deeplake.load("hub://activeloop/kth-actions")
# dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
dataloader = ds.tensorflow()

for i in range(3):
    print(ds[i])
