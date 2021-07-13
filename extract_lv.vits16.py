import torch
from itertools import product
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd 
import PIL
import os
from tqdm import tqdm
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
# !export CUDA_VISIBLE_DEVICES='0,1'

class DFDataset(Dataset):
    def __init__(self, path, transform=None, overlap=128):
        self.path = path
        self.training_df = pd.read_parquet(self.path)
        print('Training Samples', len(self.training_df))
        self.training_frames = self.training_df.frames.tolist()
        self.training_imgs = self.training_df.path.tolist()
        self.length = len(self.training_imgs)

        
        self.overlap = overlap
        self.transform = transform

    def __len__(self):
        return self.length

    # Try to turn this into torchvision.transforms so we can use with any dataset and so the input would be one image, and out put would be a flattented list of tiled images.
    def get_patch(self, image, img_file, pid, frame, x, y, save=False):
        #with an 800x600 pixel image, the image's left upper point is (0, 0), the right lower point is (800, 600).
        xx = x + 256
        yy = y + 256
        
        
        patch = image.crop((x, y, xx, yy))
        if save:
            data_root = 'subtile'
            os.makedir(os.path.join(data_root,frame), recursive=True, exists_ok=True)
            patch.save(os.path.join(data_root, frame, f'{pid}.png'))
            
        patch_df = {'path': img_file, 'frame': frame, 'pid': pid, 'x1': x, 'y1': y, 'x2': xx, 'y2': yy }

        
        return (patch, patch_df)
    
    def __getitem__(self, index): 
        img_file = self.training_imgs[index]
        img_file = '/data' + img_file
        
        image = PIL.Image.open(img_file).convert("RGB")

        wx, hy = image.size
        
        perm = product(
            np.arange(0, wx, self.overlap).astype(int),
            np.arange(0, hy, self.overlap).astype(int))
        
        frame = self.training_frames[index]
        ret = [self.get_patch(image, img_file, idx, frame, x, y) for idx, (x, y) in enumerate(perm)]

        imgs = [s[0] for s in ret]
        imdf = [s[1] for s in ret]
        
        imdf = [(i) for i in imdf]
        
#         imdf = [pd.DataFrame(adf) for adf in imdf]
        
        imgs = [self.transform(animage) for animage in imgs]

    
        return imgs, imdf

    
def customcollate(batch):
    '''
    batch: the batch of data used in pytorch data loader class.
    '''
    imgs = [k[0] for k in batch]
    imdf = [k[1] for k in batch]
    all_images = []
    
    for aimg in imgs:
        all_images.extend(aimg)
    
    all_dfs = []
    for adf in imdf:
        all_dfs.extend(adf)
    
    #     imdf = [s for s in imdf]
#     imdf = pd.concat(imdf).reset_index(drop=True)
    sample = torch.stack(all_images)
        
    return sample, all_dfs

training_file = '/data/data2/dev/af3c9242-b676-499a-8910-65addbddb8da.parquet'
# df = pd.read_parquet(training_file)


txf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

dtset = DFDataset(training_file, txf)

dtloader = torch.utils.data.DataLoader(
    dtset,
    batch_size=5,
    collate_fn=customcollate,
    num_workers=5,
)

sdf = []
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

vits16 = torch.nn.DataParallel(vits16)

device = 'cuda'
vits16.to(device)
vits16.eval()

avgpool = False


def save_npy(filename, lv):
    with open(filename, 'wb') as npf:
        np.save(npf, lv)
    return 1

np_save = {}

    
with torch.no_grad():
    for idx, (imgs, imdfs) in enumerate(tqdm(dtloader)):
        imgs = imgs.to(device)
                
        intermediate_output = vits16.module.get_intermediate_layers(imgs, 6)
        output = [x[:, 0] for x in intermediate_output]
        
#         if avgpool:
#             output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
        output = torch.cat(output, dim=-1)
    
        # NPY List
        npimgs = output.detach().cpu().numpy().tolist()
        cc = (imdfs[0]['path']).split('/')[-3]
        frame = (imdfs[0]['frame'])
        
        save_dir='/data/data2/dev/nsvideoraic/output/'
        meta_save_dir='/data/data2/dev/nsvideoraic/meta/'
        
        uniquef = []
        for s, npar in zip(imdfs, npimgs):
            s['cuid'] = cc
            f1 = s['frame']
            uniquef.append(f1)
            
            x1 = s['x1']
            y1 = s['y1']
            
            npy_savedir = os.path.join(save_dir, str(cc), str(f1), str(x1))

            meta_save = os.path.join(meta_save_dir, str(cc))
            os.makedirs(npy_savedir, exist_ok=True)
            os.makedirs(meta_save, exist_ok=True)
            
            npy_save = os.path.join(save_dir, str(cc), str(f1), str(x1), f'{str(y1)}.npy')

            s['npyfile'] = npy_save 
            np_save[npy_save] = npar 

# 
        # Add NPY dest to imdfs
        meta_save = os.path.join(meta_save_dir, str(cc))
    
        # Batch size must be 1, so that the whole batch is just one frame. Else it will store multiple
        all_df = pd.DataFrame(imdfs)

        for fff in uniquef:
            aframedf = all_df[all_df['frame'] == fff]
            aframedf.to_parquet(os.path.join(meta_save, f'frame{str(fff)}.parquet' ))

                

#         sdf.append(all_df)
res = [save_npy(ff, ll) for (ff, ll) in (np_save.items())]
