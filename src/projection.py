import torch
import pickle
import numpy as np
import random
import math

import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from einops import rearrange, repeat
import scipy.ndimage

import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# import webdataset as wds

image_transforms = []
image_transforms.extend([
    transforms.ToTensor(),      # -> c h w
    torchvision.transforms.Resize(256),
    transforms.Lambda(lambda x: x * 2. - 1.),   # x [0,1] -> [-1, 1]
    ])
            
image_transforms = torchvision.transforms.Compose(image_transforms)        


class ProjectionDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train=None, validation=None,
                 num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if train is not None:
            self.dataset_config = train
        if validation is not None:
            self.dataset_config = validation

    def train_dataloader(self):
        dataset = ProjectionData(path=self.root_dir, type="train", \
                                size=self.dataset_config.image_transforms.size)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, shuffle=(sampler is None), sampler=sampler, num_workers=self.num_workers)
        # return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)


    def val_dataloader(self):
        dataset = ProjectionData(root_dir=self.root_dir, type="val", \
                                size=self.dataset_config.image_transforms.size)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, shuffle=(sampler is None), sampler=sampler, num_workers=self.num_workers)


    def test_dataloader(self):
        dataset = ProjectionData(root_dir=self.root_dir, type="val", \
                                size=self.dataset_config.image_transforms.size)
        return DataLoader(dataset, shuffle=False, num_workers=self.num_workers)

 

class ProjectionData(Dataset):
    """
    view conditioned projection dataset
    """
    def __init__(self, path, type="train", size=256, coords=None) -> None:
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        
        self.DSD, self.DSO = data["DSD"]/1000.0, data["DSO"]/1000.0 # 1.5, 1.0
        self.type = type


        if type == "train" and coords == None:
            self.projs = data["train"]["projections"]   # (50, 256, 256)
            self.angles = data["train"]["angles"]       # (50,)
            self.n_projs = data["numTrain"]             # 50
            coords = torch.stack(torch.meshgrid(
                        torch.linspace(0, self.n_projs-1, self.n_projs),
                        torch.linspace(0, self.n_projs-1, self.n_projs), indexing="ij"), -1)    # [50, 50, 2 ij]    
            self.coords = torch.reshape(coords, [-1, 2])    # [-1, 2 ij]: (target, cond)
                
            # TODO duiplicate [x,x]
               
        elif type == "val" and coords == None:
            self.projs = data["val"]["projections"]     # (50, 256, 256)
            self.angles = data["val"]["angles"]         # (50,)
            self.n_projs = data["numVal"]
            coords = torch.stack(torch.meshgrid(
                        torch.linspace(0, self.n_projs-1, self.n_projs), 
                        torch.linspace(0, 0, 1), indexing="ij"), -1)  
            self.coords = torch.reshape(coords, [-1, 2])    # [-1, 2 ij]: (target, cond)

        elif coords is not None:
            self.projs = data[type]["projections"]     # (50, 256, 256)
            self.angles = data[type]["angles"]         # (50,)
            self.n_projs = len(self.angles)
            self.coords = coords
            # n_projs = len(self.angles)
            # coords = torch.stack(
            #             [torch.linspace(1, n_projs-1, n_projs//2), 
            #             torch.linspace(0, n_projs-2, n_projs//2)], -1)  # [(target,cond)...]
            # self.coords = torch.reshape(coords, [-1, 2])    # [-1, 2 ij]: (target, cond)



    def __len__(self) -> int:
        return self.coords.shape[0]
    

    def __getitem__(self, index):
        data = {}

        target_index, cond_index = self.coords[index].long()
        # target_index = index
        # cond_index = random.sample(range(self.n_projs), k=1)[0]

        target_edge = self.process_im(self.sobel_edge_detection(self.projs[target_index]))
        target_edge = torch.clamp(target_edge, min=-1.0, max=1.0)
        target_im = self.process_im(self.projs[target_index]) 
        target_im = torch.clamp(target_im, min=-1.0, max=1.0)
        target_RT = self.angle2pose(self.DSO, self.angles[target_index])

        cond_edge = self.process_im(self.sobel_edge_detection(self.projs[cond_index]))
        cond_edge = torch.clamp(cond_edge, min=-1.0, max=1.0)
        cond_im = self.process_im(self.projs[cond_index])
        cond_im = torch.clamp(cond_im, min=-1.0, max=1.0)
        cond_RT = self.angle2pose(self.DSO, self.angles[cond_index])

        data["target_edge"] = target_edge
        data["target_image"] = target_im    # target (rotated): (h, w, 3)
        data["target_angle"] = self.angles[target_index]

        data["cond_edge"] = target_edge
        data["cond_image"] = cond_im                 # raw (inital)
        data["cond_angle"] = self.angles[cond_index]
        data["T"] = self.get_T(target_RT, cond_RT, self.DSD)

        return data


    def process_im(self, im):
        x = im[..., np.newaxis]
        x = x.repeat(3, axis=-1)
        return image_transforms(x)


    def angle2pose(self, DSO, angle):
        """
        input
            angle: theta along X-axis
        return
            T: affine transform 
        """
        phi1 = -np.pi/2             # about axis x
        phi2 = 0.0                  # about axis y
        phi3 = angle + np.pi / 2    # about axis z

        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])

        R2 = np.array([[np.cos(phi2), 0.0, np.sin(phi2)], 
                    [0.0, 1.0, 0.0],
                    [-np.sin(phi2), 0.0, np.cos(phi2)]])

        R3 = np.array([[np.cos(phi3), -np.sin(phi3), 0.0],
                    [np.sin(phi3), np.cos(phi3), 0.0],
                    [0.0, 0.0, 1.0]])   
        
        rot = np.dot(np.dot(R3, R2), R1)
        # rot = Rz @ Ry @ Rx

        # (position) translation in the X and Y directions in Object coordinate
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        RT = np.eye(4)
        RT[:-1, :-1] = rot       # rotation matrix (M): orientation of camera
        RT[:-1, -1] = trans      # translation (b); position of camera
        return RT


    def cartesian_to_spherical(self, xyz):
        # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])
       

    def get_T(self, target_RT, cond_RT, DSD):
        dir = np.array([0, 0, DSD])
        R, T = target_RT[:3, :3], target_RT[:3, -1]
        T_target = np.matmul(R, dir) + T

        R, T = cond_RT[:3, :3], cond_RT[:3, -1]
        T_cond = np.matmul(R, dir) + T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()]) # angles in radians 
        return d_T


    def sobel_edge_detection(self, grayscale):
        grayscale = grayscale.astype(np.float32)

        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1], 
                            [0,   0,  0], 
                            [1,   2,  1]], dtype=np.float32)

        # Apply 2D convolution to compute gradients
        edge_x = scipy.ndimage.convolve(grayscale, sobel_x, mode='reflect')
        edge_y = scipy.ndimage.convolve(grayscale, sobel_y, mode='reflect')

        # Compute gradient magnitude
        gamma = 2.0  # Adjust contrast
        edge_magnitude = (edge_x ** 2 + edge_y ** 2) ** (1 / gamma)

        # Normalize to [0, 1] for visualization
        edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-6)
        return edge_magnitude

