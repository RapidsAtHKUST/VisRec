import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from copy import Error
import os
import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from thop import profile


from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image



#import pylab as plt
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from pytorch_lightning.plugins import DDPPlugin
#from context_encoder.vae import VAE
import context_encoder.encoders as m_encoder

from mlp import PosEncodedMLP_FiLM
#from data import AstroMNIST_Sparse
#from data_galaxy10 import Galaxy10_Dataset_Sparse
from data_continuous_EHT import EHTIM_Dataset, CombinedDataLoader
from data_ehtim_cont import make_dirtyim, make_im_torch

import logging
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

#import astro_utils

from scipy import interpolate
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import socket
hostname= socket.gethostname()

if hostname!= 'NV':
    matplotlib.use('Agg')
    
    
    

class NF_Model(pl.LightningModule):

    def __init__(
            self, args, 
            learning_rate=1e-4, L_embed=5, 
            input_encoding='nerf', sigma=2.5, 
            hidden_dims=[256,256], 
            latent_dim=64, kl_coeff=100.0, 
            num_fourier_coeff=32, batch_size=32, 
            input_size=28, model_checkpoint='', ngpu=None):

        super().__init__()
        
        self.save_hyperparameters()
     
        self.loss_func = nn.MSELoss(reduction='mean')
        self.loss_type = args.loss_type
        self.ngpu = ngpu

        
        self.uv_dense_sparse_index=None
        self.num_fourier_coeff = num_fourier_coeff
        self.scale_loss_image= False #args.scale_loss_image

        
        self.cond_mlp = PosEncodedMLP_FiLM(
            context_dim=latent_dim, 
            input_size=2, output_size=2, 
            hidden_dims=hidden_dims, 
            L_embed=L_embed, embed_type=input_encoding, 
            activation=nn.ReLU, 
            sigma=sigma, 
            context_type='Transformer')
        print(latent_dim, hidden_dims, L_embed, input_encoding, sigma)

        encoder = m_encoder.Visibility_Transformer(
            input_dim=2, #value dim 
            # PE dim for MLP, we are going to use the same PE as the MLP
            pe_dim=self.cond_mlp.input_size, 
            dim=512, depth=4, heads=16, 
            dim_head=512//16, 
            output_dim=latent_dim,
            dropout=.1, emb_dropout=0., 
            mlp_dim=512, 
            output_tokens=args.mlp_layers,
            has_global_token=False)

        self.pe_encoder = self.cond_mlp.embed_fun
        self.context_encoder = encoder
             

        self.norm_fact=None

        self.numEpoch = 0

        self.uv_arr= None
        self.U, self.V= None, None
        self.uv_coords_grid_query= None

        #validation plots
        self.folder_val = f'{args.val_fldr}/imgs/'
        self.folder_anim = f'{args.val_fldr}/anims/'
        os.makedirs(self.folder_val, exist_ok=True)
        os.makedirs(self.folder_anim, exist_ok=True)
        self.numPlot = 0
        self.plotFreq = 10        

        # testing
        self.test_iter = 0
        self.test_log_step = 50
        self.test_zs = []
        self.test_imgs = []
        self.test_fldr= f'../test_res/{args.exp_name}'


    
    def load_pe_encoder(self, file_path):
        print("loading checkpoint...")
        self.pe_encoder.load_state_dict(torch.load(file_path))
        print("finish loading!")

        
    def forward(self, x, z):
        pred_visibilities = self.cond_mlp(x, context=z)
        return pred_visibilities

        
    def _f(self, x):
        return ((x+0.5)%1)-0.5    
        
    def inference_w_conjugate(self, uv_coords, z, nF=0, return_np=True):
        halfspace = self._get_halfspace(uv_coords)
        
        # does this modify visibilities in place?
        uv_coords_flipped = self._flip_uv(uv_coords, halfspace)
     
        pred_visibilities = self(uv_coords_flipped, z) 
        pred_vis_real = pred_visibilities[:,:,0]
        pred_vis_imag = pred_visibilities[:,:,1]
        pred_vis_imag[halfspace] = -pred_vis_imag[halfspace] 

        if nF == 0: nF = self.hparams.num_fourier_coeff 

        pred_vis_imag = pred_vis_imag.reshape((-1, nF, nF))
        pred_vis_real = pred_vis_real.reshape((-1, nF, nF))

        pred_vis_imag[:,0,0] = 0
        pred_vis_imag[:,0,nF//2] = 0
        pred_vis_imag[:,nF//2,0] = 0
        pred_vis_imag[:,nF//2,nF//2] = 0

        if return_np:
            pred_fft =  pred_vis_real.detach().cpu().numpy() + 1j*pred_vis_imag.detach().cpu().numpy() 
        else:
            pred_fft =  pred_vis_real + 1j*pred_vis_imag

        # NEW: set border to zero to counteract weird border issues
        pred_fft[:,0,:] = 0.0
        pred_fft[:,:,0] = 0.0
        pred_fft[:,:,-1] = 0.0
        pred_fft[:,-1,:] = 0.0  
        
        return pred_fft                                   

    def _get_halfspace(self, uv_coords):
        #left_halfspace = torch.logical_and(uv_coords[:,0] > 0, uv_coords[:,1] > 0)
        left_halfspace = torch.logical_and(torch.logical_or(
            uv_coords[:,:,0] < 0, 
            torch.logical_and(uv_coords[:,:,0] == 0, uv_coords[:,:,1] > 0)), 
            ~torch.logical_and(uv_coords[:,:,0] == -.5, uv_coords[:,:,1] > 0))   
        
        return left_halfspace
    
    def _conjugate_vis(self, vis, halfspace):
        # take complex conjugate if flipped uv coords
        # so network doesn't receive confusing gradient information        
        vis[halfspace] = torch.conj(vis[halfspace])
        return vis
        
    def _flip_uv(self, uv_coords, halfspace):    

        halfspace_2d = torch.stack((halfspace, halfspace), axis=-1)
        uv_coords_flipped = torch.where(halfspace_2d, self._f(-uv_coords), uv_coords)  

        '''plt.figure()
        for i in range(3):
            plt.plot(uv_coords[i,:,0].cpu(), uv_coords[i,:,1].cpu(), 'x', alpha=0.5)
            plt.plot(uv_coords_flipped[i,:,0].cpu(), uv_coords_flipped[i,:,1].cpu(), 'o', alpha=0.5)
        plt.show()'''        

        return uv_coords_flipped

    def _recon_image_rfft(self, uv_dense, z, imsize, max_base, eht_fov, ):
        #get the query uv's
        B= uv_dense.shape[0]
        img_res=imsize[0]
        uv_dense_per=uv_dense[0]
        u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
        u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 + 1).to(u_dense)
        v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 + 1).to(u_dense)
        uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
        scale_ux= max_base * eht_fov/ img_res
        uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
        U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
        uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(B,1,1)
        #get predicted visibilities
        pred_visibilities = self(uv_coords_grid_query, z)  #Bx (HW) x 2
        pred_visibilities_map= torch.view_as_complex(pred_visibilities).reshape(B, U.shape[0], U.shape[1])
        img_recon = make_im_torch(uv_arr, pred_visibilities_map, img_res, eht_fov,
                                  norm_fact=self.norm_fact if self.norm_fact is not None else 1., 
                                  return_im=True)

        return img_recon


    def _step_image_loss(self, batch, batch_idx, num_zero_samples=0, loss_type='image',):
       pass


    def _step(self, batch, batch_idx, num_zero_samples=0):
        # batch is a set of uv coords and complex visibilities
        # print(batch)
        uv_coords, uv_dense, vis_sparse, visibilities, img, label, N_s, N_u = batch 

        pos = uv_coords
        pe_uv = self.pe_encoder(uv_coords)

        inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
        z = self.context_encoder(inputs_encoder, pos)

        
        halfspace = self._get_halfspace(uv_dense)
        uv_coords_flipped = self._flip_uv(uv_dense, halfspace)
        vis_conj = self._conjugate_vis(visibilities, halfspace[:visibilities.shape[0]])

        # now condition MLP on z #
        pred_visibilities = self(uv_coords_flipped, z)  #Bx HW x2
        vis_real = vis_conj.real.float()
        vis_imag = vis_conj.imag.float()
        
        freq_norms = torch.sqrt(torch.sum(uv_dense**2, -1))  
        abs_pred = torch.sqrt(pred_visibilities[:,:,0]**2 + pred_visibilities[:,:,1]**2)
        energy = torch.mean(freq_norms*abs_pred)
        
        real_loss = self.loss_func(torch.cat((vis_real, pred_visibilities[-N_u:,:,0]), dim = 0), pred_visibilities[:N_s+N_u,:,0])
        imaginary_loss = self.loss_func(torch.cat((vis_imag, pred_visibilities[-N_u:,:,1]), dim = 0), pred_visibilities[:N_s+N_u,:,1])

        real_loss_s = self.loss_func(vis_real, pred_visibilities[:N_s, :, 0])
        real_loss_u = self.loss_func(pred_visibilities[-N_u:, :, 0], pred_visibilities[N_s:N_s+N_u, :, 0] )
        imag_loss_s = self.loss_func(vis_imag, pred_visibilities[:N_s, :, 1])
        imag_loss_u = self.loss_func(pred_visibilities[-N_u:, :, 1], pred_visibilities[N_s:N_s+N_u, :, 1] )

        loss = 0.5 * (real_loss_s + imag_loss_s + 0.1 * (real_loss_u + imag_loss_u))
        
        return real_loss, imaginary_loss, loss, energy

    def training_step(self, batch, batch_idx, if_profile=False):

        if if_profile:
            print('start: training step')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        real_loss, imaginary_loss, loss, energy= self._step(batch, batch_idx)

            
        log_vars = [real_loss,
                    loss,
                    imaginary_loss,
                    energy]
        log_names = ['train/real_loss',
                     'train/total_loss',
                     'train/imaginary_loss',
                     'train_metadata/energy']
        
        for name, var in zip(log_names, log_vars):
            self.log(name, var,
                     sync_dist=True if self.ngpu > 1 else False,
                     rank_zero_only=True if self.ngpu > 1 else False)

        if if_profile:
            end.record()
            torch.cuda.synchronize()
            print(f'one training step took {start.elapsed_time(end)}')
            print('end: training step\n')



        return loss

    def test_step(self, batch, batch_idx):
        os.makedirs(self.test_fldr, exist_ok=True)

        uv_coords, uv_dense, vis_sparse, visibilities, img, label = batch 
        B= uv_dense.shape[0]
        vis_maps = visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
        eht_fov  = 1.4108078120287498e-09 
        max_base = 8368481300.0
        # img_res = self.hparams.input_size 
        img_res = img.shape[-1]
        nF = self.hparams.num_fourier_coeff 
        scale_ux= max_base * eht_fov/ img_res

        pos = uv_coords
        pe_uv = self.pe_encoder(uv_coords, pos)
        inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
        z = self.context_encoder(inputs_encoder, pos)

        #get the query uv's
        if self.uv_coords_grid_query is None:
            uv_dense_per=uv_dense[0]
            u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
            u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ).to(u_dense)
            v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ).to(u_dense)
            uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
            uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
            U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
            uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(B,1,1)
            self.uv_arr= uv_arr
            self.U, self.V= U,V
            self.uv_coords_grid_query= uv_coords_grid_query
            print('initilized self.uv_coords_grid_query')
        if self.norm_fact is None: # get the normalization factor, which is fixed given image/spectral domain dimensions
            visibilities_map= visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
            uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
            _, _, norm_fact = make_dirtyim(uv_dense_physical,
                                   visibilities_map.detach().cpu().numpy()[ 0, :, :].reshape(-1),
                                   img_res, eht_fov, return_im=True)
            self.norm_fact= norm_fact
            print('initiliazed the norm fact')

        
            
        elif self.loss_type == 'spectral':
            pred_fft = self.inference_w_conjugate(uv_dense, z, return_np=False)
            uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
            img_recon   = make_im_torch(self.uv_arr, pred_fft, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon_gt= make_im_torch(self.uv_arr, vis_maps, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon = (img_recon.real).float()
            img_recon_gt = (img_recon_gt.real).float()

        elif self.loss_type in ('image', 'image_spectral'):
            pred_visibilities = self(self.uv_coords_grid_query, z)  #Bx (HW) x 2
            pred_visibilities_map= torch.view_as_complex(pred_visibilities).reshape(B, self.U.shape[0], self.U.shape[1])
            img_recon = make_im_torch(self.uv_arr, pred_visibilities_map, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon_gt= make_im_torch(self.uv_arr, vis_maps, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon = (img_recon.real).float() / img_recon.abs().max()
            img_recon_gt = (img_recon_gt.real).float() / img_recon_gt.abs().max()

        img_log= torch.cat([img_recon_gt.reshape(-1, img.shape[-1]).cpu(), img_recon.reshape(-1, img_recon.shape[-1]).cpu()], dim=-1)
        plt.imsave(f'{self.test_fldr}/img_recon_{batch_idx:05d}.png', img_log,cmap='hot')

    def validation_step(self, batch, batch_idx):
        pass
     

    def validation_epoch_end(self, outputs):
        pass
    

    def from_pretrained(self, checkpoint_name):
        return self.load_from_checkpoint(checkpoint_name, strict=False)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--L_embed', type=int, default=128)
        parser.add_argument('--input_encoding', type=str, choices=['fourier','nerf','none'], default='nerf')
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--sigma', type=float, default=5.0) #sigma=1 seems to underfit and 4 overfits/memorizes
        parser.add_argument('--model_checkpoint', type=str, default='') #, default='./vae_flow_e2e_kl0.1_epoch139.ckpt')
        parser.add_argument('--val_fldr', type=str, default=f'./val_fldr-test')

        return parser


def parse_yaml(args,):
    '''
    Parse the yaml file, the settings in the yaml file are given higher priority
    args:
    argparse.Namespace 
    '''
    import yaml

    opt=vars(args)
    opt_raw= vars(args).copy()
    args_yaml= yaml.unsafe_load(open(args.yaml_file))
    opt.update(args_yaml,)

    opt['eval'] =opt_raw['eval']
    opt['exp_name'] =opt_raw['exp_name']
    opt['ngpus'] =opt_raw['ngpus']
    opt['dataset']= opt_raw['dataset']
    opt['model_checkpoint'] =opt_raw['model_checkpoint']
    opt['dataset_path']= opt_raw['dataset_path']
    opt['data_path_imgs']= opt_raw['data_path_imgs']
    opt['data_path_cont']= opt_raw['data_path_cont']
    opt['loss_type']= opt_raw['loss_type']
    opt['num_fourier']= opt_raw['num_fourier']
    opt['input_size']= opt_raw['input_size']

    args= argparse.Namespace(**opt)
    return args


def cli_main():
    pl.seed_everything(42)
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test') #default='Galaxy10_DECals_cont_mlp8')
    parser.add_argument('--ngpus', nargs='+', type=int, default=[0])
    parser.add_argument('--eval', action='store_true',
                        default=False, help='if evaluation mode [False]')

    parser.add_argument('--batch_size',  default=32, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dataset', type=str,
                        # default='Galaxy10',
                        default='Galaxy10_DECals',
                        help='MNIST | Galaxy10 | Galaxy10_DECals')

    parser.add_argument('--dataset_path', type=str, 
            #default=f'/astroim//data/eht_grid_256FC_200im_MNIST_full.h5', 
            #default=f'/astroim//data/eht_grid_256FC_200im_Galaxy10_full.h5', 
            #default=f'/astroim/data/eht_grid_256FC_200im_Galaxy10_DECals_full.h5', 
            # default=f'/astroim/data/eht_grid_256FC_200im_Galaxy10_DECals_test100.h5', 
            default=f'../data/eht_grid_256FC_200im_Galaxy10_DECals_full.h5', 
            # default=f'../data/eht_grid_128FC_200im_Galaxy10_full.h5', 
            help='dataset path to precomputed spectral data (dense grid and sparse grid)')

    parser.add_argument('--data_path_cont', type=str, 
            #default=f'/astroim/data/eht_cont_200im_MNIST_full.h5', 
            # default=f'../data/eht_cont_200im_Galaxy10_full.h5', 
            # default=f'../data/eht_cont_200im_Galaxy10_DECals_full.h5', 
            # default=f'/astroim/data/eht_cont_200im_Galaxy10_DECals_full.h5', 
            default=None,
            help='dataset path to precomputed spectral data (continuous)')

    parser.add_argument('--data_path_imgs', type=str, 
            # default=None, 
            # default='../data/Galaxy10.h5', 
            default='../data/Galaxy10_DECals.h5', 
            help='dataset path to Galaxy10 images; for MNIST, it is by default at ./MNIST; if None, sets to 0s (faster, imgs usually not needed)')
    
    parser.add_argument('--input_size',  default=64, type=int)
    parser.add_argument('--num_fourier',  default=256, type=int)
    parser.add_argument('--loss_type', type=str, default='spectral', help='spectral | image | spectral_image [spectral]')
    parser.add_argument('--scale_loss_image', type=float, default=1., help='only valid if use spectral_image as the loss_type' )
    parser.add_argument('--mlp_layers',  default=8, type=int, help=' # of layers in mlp, this will also decide the # of tokens [8]')
    parser.add_argument('--mlp_hidden_dim',  default=256, type=int, help=' hidden dims in mlp [256]')
    parser.add_argument('--m_epochs', default=300, type=int, help= '# of max training epochs [1000]')

    parser.add_argument('--yaml_file', default='', type=str, help ='path to yaml file')
    
    parser = pl.Trainer.add_argparse_args(parser) # get lightning-specific commandline options
    #parser = VAE.add_model_specific_args(parser) # get model-defined commandline options
    parser = NF_Model.add_model_specific_args(parser) # get model-defined commandline options 
    args = parser.parse_args()


    yaml_file= args.yaml_file
    if len(yaml_file)>0:
        parse_yaml(args)
    
    # print(args)
    
    numVal = 5000
    #numVal = 32 # for test100
    latent_dim = 1024

    # ------------
    # data
    # ------------    
    # load up dataset of u, v vis and images
    
    dataset = EHTIM_Dataset(dset_name = args.dataset,
                            data_path = args.dataset_path,
                            data_path_cont = args.data_path_cont,
                            data_path_imgs = args.data_path_imgs,
                            img_res = args.input_size,
                            pre_normalize = False,
                            )

    nT = 1024

    split_train, split_val = random_split(dataset, [len(dataset)-nT, nT])
    split_u, split_s = random_split(split_train, [len(split_train)-1024, 1024])


    ngpu = torch.cuda.device_count()
    #uv_pts_normalized, uv_pts_dense, vis, vis_dense, img = dataset[10]


    train_loader = CombinedDataLoader(split_s, split_u, batch_size=8, num_workers=args.num_workers)


    val_loader = DataLoader(
            split_val, 
            batch_size=4, 
            num_workers=args.num_workers, 
            drop_last=True)  

    # for i, data in enumerate(val_loader):
    #     print(i,data)


    # ------------
    # model
    # ------------
    mlp_hiddens = [args.mlp_hidden_dim for i in range(args.mlp_layers-1)]
    implicitModel = NF_Model(args,
                                        learning_rate=1e-4,
                                        L_embed=args.L_embed,
                                        input_encoding=args.input_encoding,
                                        sigma=args.sigma,
                                        num_fourier_coeff=args.num_fourier,
                                        batch_size=32,
                                        input_size=args.input_size,
                                        latent_dim=latent_dim,
                                        hidden_dims=mlp_hiddens,
                                        model_checkpoint=args.model_checkpoint,
                                        ngpu=ngpu)

    
    checkpoint_callback = ModelCheckpoint(monitor='train/total_loss',dirpath='')
    # trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
    trainer = pl.Trainer.from_argparse_args(args, 
                                            gpus=args.ngpus,
                                            plugins=DDPPlugin(find_unused_parameters=False),
                                            replace_sampler_ddp=True,
                                            accelerator='ddp',
                                            progress_bar_refresh_rate=20, 
                                            max_epochs=args.m_epochs, 
                                            val_check_interval=100, 
                                            )
                                            
    # ------------
    # training
    # ------------
    print('==Training==')
    print(f'--- loading from {args.model_checkpoint}...')
    trainer.fit(implicitModel, train_loader, val_loader)  
    print(implicitModel)


if __name__ == '__main__':
    cli_main()

