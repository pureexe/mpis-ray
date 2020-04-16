from torch.utils.data import Dataset
from read_write_model import read_model
from scipy.spatial.transform import Rotation
from PIL import Image
import numpy as np
import os

class RayDataset(Dataset):
    def __init__(self,dataset_dir, scale = 1.0):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir,'dense/images')
        self.sparse_dir = os.path.join(dataset_dir,'dense/sparse')
        if not os.path.exists(self.image_dir) or not os.path.exists(self.sparse_dir):
            raise RuntimeError('Directory dataset \'{}\'maltform!'.format(dataset_dir))
        intrinsics, extrinsics, _ = read_model(self.sparse_dir)
        self.extrinsics_index = []
        self.extrinsics = {}
        self.intrinsics = {}
        self.pixels = None
        self.pixel_loc = None
        scale_x = scale_y = 1.0
        self.image_len = 0
        for i in extrinsics:
            self.extrinsics_index.append(i) #in case extrinsic id not sort.
            extrinsic = extrinsics[i]
            self.extrinsics[i] = {
                'rotation': Rotation.from_quat(extrinsic[1]).as_matrix(),
                'translation': extrinsics[2],
                'intrinsic_id': extrinsic[3]
            }
            image, scale_x, scale_y = self.read_image(extrinsic[4])
            grid = self.get_grid(image.shape)
            image = image.reshape(-1,3)
            self.image_len = image.shape[0]
            if self.pixels is None:
                self.pixels = image
                self.pixel_loc = grid
            else:
                self.pixels = np.vstack(self.pixels,image)
                self.grid = np.vstack(self.pixel_loc,grid)

        for i in intrinsics:
            fx, fy, px, py = self.parse_camera(intrinsics[i]) 
            self.intrinsics[i] = np.array([
                [fx*scale_x,        0.0,    px*scale_x],
                [       0.0, fy*scale_y,    py*scale_y],
                [       0.0,        0.0,            1.0]
            ])

    def __len__(self):
        return self.pixels.shape[0]
    
    def __getitem__(self, idx):
        extrinsic = self.get_extrinsic(idx)
        return {
            'rotation': extrinsic['rotation'],
            'translation': extrinsic['translation']
            'intrinsic': self.intrinsics[extrinsic['intrinsic_id']]
            'position': self.pixel_loc[idx]
            'color': self.pixels[idx]
        }

    def get_extrinsic(self,idx):
        image_id = int(idx / self.image_len)
        extrinsic_id = self.extrinsics_index[image_id]
        return self.extrinsics[extrinsic_id]

    def get_grid(self,shape):
        h,w,c = shape
        p = np.meshgrid(range(h),range(w))
        p = np.dstack(p)
        return p.reshape(-1,2)

    def read_image(self, path, scale):
        image_path = os.path.join(image_dir, path)
        if not os.path.exists(image_path):
            raise RuntimeError("Image {} not found in dataset".format(path))
        image = Image.open(image_path)
        if scale == 1.0:
            return np.array(image), 1.0, 1.0
        width, height = im.size
        new_width = int(width*scale)
        new_height = int(height*scale)
        scale_x = new_width / width
        scale_y = new_height / height
        image = imae.resize((new_width,new_height))
        return np.array(image), scale_x, scale_y

    def parse_camera(self,camera):
        """ parse camrera """
        index, model, width, height, params = cam
        if model == 'PINHOLE': #undistorted photo
            fx, fy, px, py = params
        elif model == 'SIMPLE_RADIAL': #default colmap distort image
            fx, px, py, _ = params
            fy = fx
        else:
            raise NotImplementedError('Camera model {} isn\'t implement yet')
        return fx,fy,px,py