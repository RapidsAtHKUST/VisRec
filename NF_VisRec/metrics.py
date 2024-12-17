import os
import glob
import csv
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Function to compute the metrics for two images
def compute_metrics(img1, img2):
    mse_val = mse(img1, img2)
    psnr_val = psnr(img1, img2)
    ssim_val = ssim(img1, img2, multichannel=True)
    
    return mse_val, psnr_val, ssim_val



mse_list = []
psnr_list = []
ssim_list = []

for i in range(0, 1024):

    asamples_mean_img_path = ''+str(i)+'/recon_image.png'
    image_img_path = ''+str(i)+'/image.png'

    asamples_mean_img = cv2.imread(asamples_mean_img_path, cv2.COLOR_RGB2GRAY)
 
    image_img = cv2.imread(image_img_path, cv2.COLOR_RGB2GRAY)
    mse_val, psnr_val, ssim_val = compute_metrics(asamples_mean_img, image_img)

    mse_list.append(mse_val)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

mse_mean = np.mean(mse_list)
mse_std = np.std(mse_list)
psnr_mean = np.mean(psnr_list)
psnr_std = np.std(psnr_list)
ssim_mean = np.mean(ssim_list)
ssim_std = np.std(ssim_list)

import fileinput  
  
file = open("res.txt", encoding="utf-8",mode="a")  
file.write("PSNR ") 
file.write(str(psnr_mean))
file.write(" ") 
file.write(str(psnr_std)) 
file.write("\n")  
file.write("SSIM ") 
file.write(str(ssim_mean))
file.write(" ") 
file.write(str(ssim_std)) 
file.write("\n")  
file.close()  
print("PSNR", psnr_mean, psnr_std)
print("SSIM", ssim_mean, ssim_std)

import numpy as np
import matplotlib.pyplot as plt
import os

def LFD(recon_freq, real_freq):

    tmp = (recon_freq - real_freq) ** 2
    freq_distance = tmp[:,0,:,:] + tmp[:,1,:,:]

    LFD = np.log(freq_distance + 1)
    return LFD




data_list_1 = []
data_list_2 = []

for i in range(1024):
    folder_name = ""+str(i)
    file_path = os.path.join(folder_name, "GT_vis.npy")
    
    data = np.load(file_path)
    
    data_list_1.append(data)

    folder_name = ""+str(i)
    file_path = os.path.join(folder_name, "recon_vis.npy")

    data = np.load(file_path)
    
    data_list_2.append(data)

result_1 = np.stack(data_list_1, axis=0)
result_2 = np.stack(data_list_2, axis=0)


res = LFD(result_1, result_2)
res_vector = np.mean(res, axis=(1, 2))
mean = np.mean(res_vector)
std_dev = np.std(res_vector)

print("LFD:", mean, std_dev)

file = open("res.txt", encoding="utf-8",mode="a")  
file.write("LFD ") 
file.write(str(mean))
file.write(" ") 
file.write(str(std_dev)) 
file.write("\n")   
file.write("\n") 
file.close()  
