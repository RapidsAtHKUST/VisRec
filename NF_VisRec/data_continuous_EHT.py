from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_ehtim_cont import *
import torch
import random


def load_h5_uvvis(fpath):
    print('--loading h5 file for eht sparse and dense {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_sparse = np.array(F['u_sparse'])
        v_sparse = np.array(F['v_sparse'])
        vis_re_sparse = np.array(F['vis_re_sparse'])
        vis_im_sparse = np.array(F['vis_im_sparse'])
        u_dense = np.array(F['u_dense'])
        v_dense = np.array(F['v_dense'])
        vis_re_dense = np.array(F['vis_re_dense'])
        vis_im_dense = np.array(F['vis_im_dense'])
    print('Done--')
    return u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense


def load_h5_uvvis_cont(fpath):
    print('--loading h5 file for eht continuous {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_cont = np.array(F['u_cont'])
        v_cont = np.array(F['v_cont'])
        vis_re_cont = np.array(F['vis_re_cont'])
        vis_im_cont = np.array(F['vis_im_cont'])
    print('Done--')
    return u_cont, v_cont, vis_re_cont, vis_im_cont



def Transform_function_batch(vis_sparse, uv_sparse):
    batch_size = vis_sparse.size(0)

    for i in range(batch_size):
        if random.random() > 0.5:
            # noise on position
            noise = torch.randn_like(uv_sparse[i])  
            uv_sparse[i] = uv_sparse[i] + noise * 0.0001

        if random.random() > 0.5:
            # noise on vis
            noise = torch.randn_like(vis_sparse[i])  
            rate = random.uniform(0.5, 5)
            vis_sparse[i] = vis_sparse[i] + rate * noise

        if random.random() > 0.5:
            # global position offset
            offset1 = 0.0001 * torch.FloatTensor(1).uniform_(-1, 1)
            uv_sparse[i, :, 0] += offset1
            offset2 = 0.0001 * torch.FloatTensor(1).uniform_(-1, 1)
            uv_sparse[i, :, 1] += offset2

        if random.random() > 0.5:
            # crop
            min_rows = int(vis_sparse.size(1) // 2)
            max_rows = int(vis_sparse.size(1) // 1.3)
            n = random.randint(min_rows, max_rows + 1)
            indices = torch.randperm(vis_sparse.size(1))[:n]
            vis_sparse[i, indices, :] = 0

        if random.random() > 0.3:
            distances = torch.sqrt(uv_sparse[i,:,0]**2 + uv_sparse[i,:,1]**2)
            if random.random() > 0.5:
                # no high
                d = random.uniform(0.2, 0.4)
                selected_indices = (distances > d).nonzero().squeeze()
                vis_sparse[i, selected_indices, :] = 0
            else:
                # no low
                d = random.uniform(0.2, 0.4)
                selected_indices = (distances < d).nonzero().squeeze()
                vis_sparse[i, selected_indices, :] = 0

        # random mask
        lower_mask_rate = 0.4
        upper_mask_rate = 0.6
        num_vectors_to_mask = random.randint(int(vis_sparse.size(1) * lower_mask_rate), 
                                             int(vis_sparse.size(1) * upper_mask_rate))
        masked_indices = torch.randperm(vis_sparse.size(1))[:num_vectors_to_mask]
        vis_sparse[i, masked_indices, :] = 0

    return vis_sparse, uv_sparse


def Transform_function_batch_slight(vis_sparse, uv_sparse):
    batch_size = vis_sparse.size(0)

    for i in range(batch_size):
        if random.random() > 0.9:
            noise = torch.randn_like(uv_sparse[i])  
            uv_sparse[i] = uv_sparse[i] + noise * 0.0001

        if random.random() > 0.8:
            noise = torch.randn_like(vis_sparse[i])  
            rate = random.uniform(0.5, 5)
            vis_sparse[i] = vis_sparse[i] + rate * noise

        if random.random() > 0.9:
            offset1 = 0.0001 * torch.FloatTensor(1).uniform_(-1, 1)
            uv_sparse[i, :, 0] += offset1
            offset2 = 0.0001 * torch.FloatTensor(1).uniform_(-1, 1)
            uv_sparse[i, :, 1] += offset2

        if random.random() > 0.8:
            lower_mask_rate = 0
            upper_mask_rate = 0.5
            num_vectors_to_mask = random.randint(int(vis_sparse.size(1) * lower_mask_rate), 
                                                int(vis_sparse.size(1) * upper_mask_rate))
            masked_indices = torch.randperm(vis_sparse.size(1))[:num_vectors_to_mask]
            vis_sparse[i, masked_indices, :] = 0

    return vis_sparse, uv_sparse


from torch.utils.data import DataLoader

class CombinedDataLoader:
    def __init__(self, supervised_dataset, unsupervised_dataset, batch_size, num_workers):
        self.supervised_loader = DataLoader(
            supervised_dataset,
            batch_size = 8, 
            num_workers=num_workers,
            shuffle=True
        )
        self.unsupervised_loader = DataLoader(
            unsupervised_dataset,
            batch_size = 8, 
            num_workers=num_workers,
            shuffle=True,
        )
        self.Transform_function_batch = Transform_function_batch
        self.Transform_function_batch_slight = Transform_function_batch_slight
    
    def __iter__(self):
        self.supervised_iter = iter(self.supervised_loader)
        self.unsupervised_iter = iter(self.unsupervised_loader)
        return self

    def __next__(self):
        supervised_batch = next(self.supervised_iter)
        unsupervised_batch = next(self.unsupervised_iter)

        vis_sparse, uv_sparse = self.Transform_function_batch(unsupervised_batch[2], unsupervised_batch[0])
        # vis_sparse_aug, uv_sparse_aug = self.Transform_function_batch(supervised_batch[2], supervised_batch[0])
        vis_sparse_s, uv_sparse_s = self.Transform_function_batch_slight(supervised_batch[2], supervised_batch[0])
        uv_coords_combined = torch.cat((uv_sparse_s, unsupervised_batch[0], uv_sparse), dim=0)
        uv_dense_combined = torch.cat((supervised_batch[1], unsupervised_batch[1], unsupervised_batch[1]), dim=0)
        vis_sparse_combined = torch.cat((vis_sparse_s, unsupervised_batch[2], vis_sparse), dim=0)
        visibilities_combined = supervised_batch[3]
        img_combined = torch.cat((supervised_batch[4], unsupervised_batch[4], unsupervised_batch[4]), dim=0)
        labels_combined = torch.cat((supervised_batch[5], unsupervised_batch[5], unsupervised_batch[5]), dim=0)
        return uv_coords_combined, uv_dense_combined, vis_sparse_combined, visibilities_combined, img_combined, labels_combined, supervised_batch[0].shape[0], uv_sparse.shape[0]



def Transform_function(vis_sparse, uv_sparse):

    if random.random() > 0.5:
        noise = np.random.randn(*uv_sparse.shape)   
        uv_sparse = uv_sparse + noise * 0.0001

    if random.random() > 0.5:
        noise = np.random.randn(*vis_sparse.shape)   # Random values from a Gaussian distribution with mean=0 and std_dev=0.05
        vis_sparse = vis_sparse + noise  # Add the noise to the original data

    if random.random() > 0.5:
        offset1 = 0.0001 * np.random.uniform(-1, 1)
        uv_sparse[:, 0] += offset1
        offset2 = 0.0001 * np.random.uniform(-1, 1)
        uv_sparse[:, 1] += offset2

    # if random.random() > 0.5:
    #     # flip v
    #     uv_sparse[:, 1] *= -1

    # if random.random() > 0.5:
    #     # flip u
    #     uv_sparse[:, 0] *= -1

    if random.random() > 0.5:
        # crop
        min_rows = int(vis_sparse.shape[0] // 2)
        max_rows = int(vis_sparse.shape[0] // 1.3)

        n = np.random.randint(min_rows, max_rows + 1)  # max_rows + 1 because the high end is exclusive in numpy's randint function

        indices = np.random.choice(vis_sparse.shape[0], n, replace=False)
        vis_sparse[indices, :] = 0

    if random.random() > 0.3:
        if random.random() > 0.5:
            # no high
            d = random.uniform(0.2, 0.4)
            distances = np.sqrt(uv_sparse[:,0]**2 + uv_sparse[:,1]**2)
            binary_vector = (distances > d).astype(float)
            selected_indices = np.where(distances > d)[0]
            vis_sparse[selected_indices, :] = 0
        else:
            # no low
            d = random.uniform(0.2, 0.4)
            distances = np.sqrt(uv_sparse[:,0]**2 + uv_sparse[:,1]**2)
            binary_vector = (distances < d).astype(float)
            selected_indices = np.where(distances < d)[0]
            vis_sparse[selected_indices, :] = 0


    # random mask
    lower_mask_rate = 0.4
    upper_mask_rate = 0.6
    num_vectors_to_mask_lower = int(vis_sparse.shape[0] * lower_mask_rate)
    num_vectors_to_mask_upper = int(vis_sparse.shape[0] * upper_mask_rate)
    num_vectors_to_mask = np.random.randint(low=num_vectors_to_mask_lower, high=num_vectors_to_mask_upper + 1, size=(1)).item()
    masked_indices = np.random.permutation(vis_sparse.shape[0])[:num_vectors_to_mask]
 
    vis_sparse[masked_indices, :] = 0

    return vis_sparse, uv_sparse


class EHTIM_Dataset(Dataset):
    '''
    EHT-imaged dataset (load precomputed)
    ''' 
    def __init__(self,  
            dset_name = 'Galaxy10', # 'MNIST'
            data_path = '../data/eht_grid_128FC_200im_Galaxy10_DECals_full.h5', 
            data_path_imgs = '../data/Galaxy10_DECals.h5', 
            data_path_cont = '../data/eht_cont_200im_Galaxy10_DECals_full.h5',
            img_res = 200,
            mode = '',
            train_classes = [],
            pre_normalize = False,
            unsupervised = False,
            ):

        self.unsupervised = unsupervised
        self.transform_function = Transform_function
        # get spectral data
        u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense = load_h5_uvvis(data_path)
        print(u_sparse.shape, v_sparse.shape, vis_re_sparse.shape, vis_im_sparse.shape, u_dense.shape, v_dense.shape, vis_re_dense.shape, vis_im_dense.shape)

        uv_sparse = np.stack((u_sparse.flatten(), v_sparse.flatten()), axis=1)
        uv_dense = np.stack((u_dense.flatten(), v_dense.flatten()), axis=1)
        fourier_resolution = int(len(uv_dense)**(0.5))
        self.fourier_res = fourier_resolution

        # rescale uv to (-0.5, 0.5)
        max_base = np.max(uv_sparse)
        uv_dense_scaled = np.rint((uv_dense+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
        self.uv_dense = uv_dense_scaled
        self.vis_re_dense = vis_re_dense
        self.vis_im_dense = vis_im_dense
        # TODO: double check un-scaling if continuous (originally scaled with sparse) 
        # should be ok bc dataset generation was scaled to max baseline, so np.max(uv_sparse)=np.max(uv_cont)
            
        # use sparse continuous data
        if data_path_cont:
            print('using sparse continuous visibility data..')
            u_cont, v_cont, vis_re_cont, vis_im_cont = load_h5_uvvis_cont(data_path_cont)
            uv_cont = np.stack((u_cont.flatten(), v_cont.flatten()), axis=1)
            uv_cont_scaled = np.rint((uv_cont+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_cont_scaled
            self.vis_re_sparse = vis_re_cont
            self.vis_im_sparse = vis_im_cont
            
        # use sparse grid data
        else:
            print('using sparse grid visibility data..')
            uv_sparse_scaled = np.rint((uv_sparse+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_sparse_scaled
            self.vis_re_sparse = vis_re_sparse
            self.vis_im_sparse = vis_im_sparse
        
        # load GT images
        self.img_res = img_res 

        self.train_classes = train_classes
        self.mode = mode 
        self.indices = []
        
        if dset_name == 'MNIST':
            if data_path_imgs:
                from torchvision.datasets import MNIST
                from torchvision import transforms

                transform = transforms.Compose([transforms.Resize((img_res, img_res)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.1307,), (0.3081,)),
                                                ])
                self.img_dataset = MNIST('', train=True, download=True, transform=transform)
            else:  # if loading img data is not necessary
                self.img_dataset = None

        elif dset_name == 'Galaxy10' or 'Galaxy10_DECals':
            if data_path_imgs:
                self.img_dataset = Galaxy10_Dataset(data_path_imgs, None)
                # self.set_mode(mode)
            else:  # if loading img data is not necessary
                self.img_dataset = None

        else:
            print('[ MNIST | Galaxy10 | Galaxy10_DECals ]')
            raise NotImplementedError
            
        # pre-normalize data? (disable for phase loss)
        self.pre_normalize = pre_normalize

    def set_mode(self, mode):
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"
        self.mode = mode
        if self.img_dataset is not None:
            if mode == 'train':
                self.indices = [i for i, (_, label) in enumerate(self.img_dataset) if label in self.train_classes]
            else:
                self.indices = [i for i, (_, label) in enumerate(self.img_dataset) if label not in self.train_classes]
        else:
            self.indices = []
            

    def __getitem__(self, idx):
        # idx = self.indices[idx]
        vis_dense = self.vis_re_dense[:,idx] + 1j*self.vis_im_dense[:,idx]
        vis_real = self.vis_re_sparse[:,idx].astype(np.float32)
        vis_imag = self.vis_im_sparse[:,idx].astype(np.float32)
        if self.pre_normalize == True:
            padding = 50 ## TODO make this actual hyperparam
            real_min, real_max= np.amin(vis_real)-padding, np.amax(vis_real)+padding
            imag_min, imag_max= np.amin(vis_imag)-padding, np.amax(vis_imag)+padding
            vis_real_normed = (vis_real - real_min) / (real_max - real_min)
            vis_imag_normed = (vis_imag - imag_min) / (imag_max - imag_min)
            vis_sparse = np.stack([vis_real_normed, vis_imag_normed], axis=1) 
        else:
            vis_sparse = np.stack([vis_real, vis_imag], axis=1) 

        if self.img_dataset:
            img, label = self.img_dataset[idx]
            img_res_initial = int(torch.numel(img)**(0.5))
            img = img.reshape((img_res_initial,img_res_initial))
            if img_res_initial != self.img_res:
                img = upscale_tensor(img, final_res=self.img_res, method='cubic')
                img = torch.from_numpy(img)
        else:
            img = torch.from_numpy(np.zeros((self.img_res,self.img_res)))
            label = None

        # mask_matrix = np.load('vlba.npy')  
        # # Reshape the matrix to 1x16384, iterating from left to right and bottom to top
        # mask_bool = mask_matrix.astype(bool).flatten()
        # self.uv_sparse = self.uv_dense[mask_bool]
        
        # padding = 0 ## TODO make this actual hyperparam
        # vis_real = vis_dense[mask_bool].real
        # vis_imag = vis_dense[mask_bool].imag
        # real_min, real_max= np.amin(vis_real)-padding, np.amax(vis_real)+padding
        # imag_min, imag_max= np.amin(vis_imag)-padding, np.amax(vis_imag)+padding
        # vis_real_normed = (vis_real - real_min) / (real_max - real_min)
        # vis_imag_normed = (vis_imag - imag_min) / (imag_max - imag_min)
        # vis_sparse = np.stack([vis_real_normed, vis_imag_normed], axis=1) 

        # if self.unsupervised == False:    
        return self.uv_sparse.astype(np.float32), self.uv_dense.astype(np.float32), vis_sparse.astype(np.float32), vis_dense, img, label

        # else:
        #     vis_sparse, uv_sparse == self.transform_function(vis_sparse, self.uv_sparse)
        #     return self.uv_sparse.astype(np.float32), self.uv_dense.astype(np.float32), vis_sparse.astype(np.float32), vis_dense, img, label
     

    def __len__(self):
        return len(self.vis_re_sparse[0,:])

if __name__ == "__main__":
    
    fourier_resolution = 64
    dset_name = 'Galaxy10' #'MNIST'
    idx = 123
    
    data_path =f'../data/eht_grid_128FC_200im_Galaxy10_DECals_full.h5'

    #spectral_dataset = EHTIM_Dataset(data_path)
    #uv_sparse, uv_dense, vis_sparse, vis_dense = spectral_dataset[idx]
    
    im_data_path = '../data/Galaxy10_DECals.h5'
    spectral_dataset = EHTIM_Dataset(dset_name = dset_name,
                                     data_path = data_path,
                                     data_path_imgs = im_data_path,
                                     img_res = 200
                                    )
    uv_sparse, uv_dense, vis_sparse, vis_dense, img = spectral_dataset[idx]
    print(uv_sparse.shape, uv_dense.shape, vis_sparse.shape, vis_dense.shape, img.shape)
    
    # plot data
    vis_amp_sparse = np.linalg.norm(vis_sparse, axis=1)
    vis_amp_dense = np.abs(vis_dense)

    print(uv_sparse.shape)
    plt.scatter(uv_sparse[:,0], uv_sparse[:,1], c=vis_amp_sparse)
    plt.savefig('ehtim_sparse.png')
    print(uv_dense.shape)
    print(uv_dense)
    print(vis_amp_dense.shape)
    print(vis_amp_dense)
    plt.scatter(uv_dense[:,0], uv_dense[:,1], c=vis_amp_dense)
    plt.savefig('ehtim_dense.png')
    
    plt.imshow(img)
    plt.savefig('ehtim_gt_img.png')
    
#    obs_meta = spectral_dataset.get_metadata(idx, dset_name)
#    plt.imshow(obs_meta['gt_img'])
#    plt.savefig('ehtim_gt_img.png')
