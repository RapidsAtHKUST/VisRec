from utils.data_utils.transform_util import *
import numpy as np
import matplotlib.pyplot as plt
import torch

uv_dense1 = np.load("data/uv_dense.npy") 
uv_dense1 = torch.tensor(uv_dense1)

for i in range(0, 1024):
    x_full = np.load(""+str(i)+".npy") 
    x_full = x_full[None, ...]
    image_full = to_img_th(torch.tensor(x_full[:,0,...]), torch.tensor(x_full[:,1,...]), uv_dense1)
    plt.imsave(fname=""+str(i)+"_image.png",arr=image_full[0,0,...].cpu().detach().numpy(), cmap="hot")

   