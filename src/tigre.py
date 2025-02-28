import torch
import pickle
import os
import sys
import numpy as np

from torch.utils.data import DataLoader, Dataset


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # 1.5; Distance Source Detector      (m)
        self.DSO = data["DSO"]/1000  # 1.0; Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data["nDetector"])  # [256, 256]; number of pixels              (px)
        self.dDetector = np.array(data["dDetector"])/1000  # [0.001, 0.001]; size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # [0.256, 0.256]; total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # [128, 128, 128]; number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"])/1000  # [0.001, 0.001, 0.001]; size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # [0.128, 0.128, 0.128]; total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"])/1000  # [0., 0., 0.]; Offset of image from origin   (m)
        self.offDetector = np.array(data["offDetector"])/1000  # [0., 0.]; Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]    # 0.5; Accuracy of FWD proj   (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]            # parallel, cone   ... (fan TODO)
        self.filter = data["filter"]        # None;


class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, start_view=0, n_views=50, n_rays=1024, type="train", device="cuda"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_rays = n_rays    # 1024
        self.near, self.far = self.get_near_far(self.geo)   
    
        if type == "train":
            # ----------------- along projections 256x256 vx -----------------
            self.projs = torch.tensor(data["train"]["projections"][start_view:start_view+n_views], dtype=torch.float32, device=device) # [50, 256, 256]
            # rays: [numAngles, nDetector_h, nDetector_w, ray] = [50, 256, 256, 8]
            # ray = rays[..., idx] = [rays_o 3d, rays_d 3d, near, far] 
            angles = data["train"]["angles"][start_view:start_view+n_views]    # (50,); sources in (xo,yo) plane

            rays = self.get_rays(angles, self.geo, device)       # [50, 256, 256, 6]; 6=[rays_o, rays_d]
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)  # [50, 256, 256, 8]
            self.n_projs =  len(self.projs)          # data["numTrain"]     # 50
            # coords: coordinates for one projection/view
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"),
                                 -1)                        # [256, 256, 2 ij]
            self.coords = torch.reshape(coords, [-1, 2])    # [256*256, 2 ij]

            # ----------------- along CT images 128x128 px -----------------
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)                # [128, 128, 128]
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)   # [128, 128, 128, 3]
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_projs = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        
    def __len__(self):
        return self.n_projs

    def __getitem__(self, index):
        """
        Input:
            index: idx of a projection -> one view
        Return (train):
            projs: random selected n_rays=1024 pixels from ONE projection
            rays: correspond n_rays=1024 rays [x,y,z]
        """
        if self.type == "train":
            projs_valid = (self.projs[index]>0).flatten()   # one view [256, 256]->[65536]
            coords_valid = self.coords[projs_valid]         # [43915, 2]

            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
            select_coords = coords_valid[select_inds].long()    # [1024, 2]
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]   # [50, 256, 256, 8] -> [1024, 8]
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]] # [50, 256, 256] -> [1024]
            out = {
                "projs":projs,  # [1024]
                "rays":rays,    # [1024, 8]     # random select 1024 rays
            }
        elif self.type == "val":        
            rays = self.rays[index]     # [50, 256, 256, 8] -> [256, 256, 8]
            projs = self.projs[index]   # [50, 256, 256] -> [256, 256, 8] position
            out = {
                "projs":projs,  # [256, 256]    ground-truth value
                "rays":rays,    # [256, 256, 8] positions
            }
        return out

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])  #=>(H, W, num_imgs)
        return voxel
    
    def get_rays(self, angles, geo: ConeGeometry, device):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        W, H = geo.nDetector    # 256, 256
        DSD = geo.DSD
        rays = []
        
        for angle in angles:    # source position in (Xo, Yo) plane
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)     # affine transform: [4,4]
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                # pixel coordinates in the projection
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]  # (u,v) coordinates: [256,256]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                # uu/DSD: normalizing the pixel coordinates by DSD -> homogeneous coordinates?
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)   # (uu, vv, 1) homogeneous coordinates; [256, 256, 3]
                # rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) 
                rays_d = torch.matmul(pose[:3,:3], dirs[..., None]).squeeze().to(device)    # [256, 256, 3]; a rotated point
                rays_o = pose[:3, -1].expand(rays_d.shape)   # origin vectors: [256, 256, 3]; source direction vector in O coordinate
            elif geo.mode == "parallel":        # ???
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] * # direction vectors 
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)

                # import open3d as o3d
                # from src.util.draw_util import plot_rays, plot_cube, plot_camera_pose
                # cube1 = plot_cube(np.zeros((3,1)), geo.sVoxel[...,np.newaxis])
                # cube2 = plot_cube(np.zeros((3,1)), np.ones((3,1))*geo.DSO*2)
                # rays1 = plot_rays(rays_d.cpu().detach().numpy(), rays_o.cpu().detach().numpy(), 2)
                # poseray = plot_camera_pose(pose.cpu().detach().numpy())
                # o3d.visualization.draw_geometries([cube1, cube2, rays1, poseray])
            
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1)) # rays.append([256, 256, 6])

        return torch.stack(rays, dim=0)     # [50, 256, 256, 6]

    def angle2pose(self, DSO, angle):
        """
        input
            angle: theta along X-axis
        return
            T: affine transform 
        """
        phi1 = -np.pi/2   
        phi2 = 0.0
        phi3 = angle + np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        R2 = np.array([[np.cos(phi2), 0.0, np.sin(phi2)], 
                    [0.0, 1.0, 0.0],
                    [-np.sin(phi2), 0.0, np.cos(phi2)]])
        R3 = np.array([[np.cos(phi3), -np.sin(phi3), 0.0],
                    [np.sin(phi3), np.cos(phi3), 0.0],
                    [0.0, 0.0, 1.0]])   
        
        # phi1 = -np.pi/2   
        # phi2 = np.pi / 2
        # phi3 = angle 
        # R1 = np.array([[1.0, 0.0, 0.0],
        #             [0.0, np.cos(phi1), -np.sin(phi1)],
        #             [0.0, np.sin(phi1), np.cos(phi1)]])
        # R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0], 
        #             [np.sin(phi2), np.cos(phi2), 0.0],
        #             [0.0, 0.0, 1.0]])
        # R3 = np.array([[np.cos(phi3), -np.sin(phi3), 0.0],
        #             [np.sin(phi3), np.cos(phi3), 0.0],
        #             [0.0, 0.0, 1.0]])   
        
        rot = np.dot(np.dot(R3, R2), R1)
        # rot = Rz @ Ry @ Rx

        # (position) translation in the X and Y directions in Object coordinate
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot       # rotation matrix (M): orientation of camera
        T[:-1, -1] = trans      # translation (b); position of camera
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        Return:
            near: nearest distance from Source to Origin = DSO - near of CT
            far: farest distence from Srouce to Origin = DSO + far of CT
        """
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far    # 0.9044903320081219 1.0955096679918779
