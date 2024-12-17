import numpy as np
import torch as th
from utils.dataset_utils.data_ehtim_cont import *


# -------- FFT transform --------

def to_img_th(s_vis_real, s_vis_imag, uv_dense):
    nF = 128
    # s_vis_imag[:, 0, 0] = 0
    # s_vis_imag[:, 0, nF//2] = 0
    # s_vis_imag[:, nF//2, 0] = 0
    # s_vis_imag[:, nF//2, nF//2] = 0

    s_fft = s_vis_real + 1j * s_vis_imag

    # # NEW: set border to zero to counteract weird border issues
    # s_fft[:, 0, :] = 0.0
    # s_fft[:, :, 0] = 0.0
    # s_fft[:, :, -1] = 0.0
    # s_fft[:, -1, :] = 0.0

    eht_fov = 1.4108078120287498e-09
    max_base = 8368481300.0
    img_res = 256
    scale_ux = max_base * eht_fov / img_res
    b = s_vis_imag.shape[0]
    uv_dense_per = uv_dense #.unsqueeze(0) #.repeat(s_vis_real.size(0), 1, 1)
    u_dense, v_dense = uv_dense_per[:, 0].unique(), uv_dense_per[:, 1].unique()
    u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ).to(u_dense)
    v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ).to(u_dense)
    uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
    uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input

    img_recon = make_im_torch(uv_arr, s_fft, img_res, eht_fov, norm_fact=1., return_im=True)

    img_real = img_recon.real.unsqueeze(1)
    img_imag = img_recon.imag.unsqueeze(1)
    img_recon = torch.cat([img_real, img_imag], dim=1)

    img_recon = img_recon.reshape(b, 2, 256, 256)

    return img_recon.float()




