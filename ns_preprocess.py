# data_preprocess.py
#
# Brian D Goodwin, 2021-03-22
#
# creates raic datapreprocessing class.
#
# UPDATES NEEDED:
# Time dependencies on the maps for change detection.
#
# Need to save out vectors as npy files and move to azure
#
# Need to build appropriate dim tables for each ball tree
# calculation (inside calculate_ball_tree() method)
#
# Ball tree needs to calculated used a discretized approach
#
# Need to add json files to azure blob where each json file is a 
# path to either a tree or a dim table so that the production script
# knows what exists. Note that if it's a map, you need to include the
# zlevel.
#
# calculate_raic_vectors() does not accomodate saving parquet files
# for later retrieval.
#
# Current functions do not support averaging the latent code from 
# multiple transforms of the same image.
#
# need to accomodate a custom downloaded model

import argparse
import os
import pickle
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from math import ceil, log2, log10
from pprint import pprint
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import torch
from itertools import product
from azure.storage.blob import ContainerClient
from PIL import Image
import PIL
from pythaic.raic import latentspace
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

# GLOBAL CONSTANTS
DATA_DIRECTORY = '/data/data2/datasets/video/video-display'
AZURE_STORAGE_ACCOUNT = "https://geospatialraicstorage001.blob.core.windows.net"
SAS_KEY = '?sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2022-02-04T03:39:10Z&st=2021-02-03T19:39:10Z&spr=https&sig=UxEiitFWuWQ0jEiUFrMd5G8vGAnjiF9KH4OGCAoLGAE%3D'
DATA_TYPES = ['video-display']
DOMAIN_TYPES = ['guid']
CUSTOM_FOLDERS = ['nsbit']


class NSFrame(Dataset):
    def __init__(self, indf, transform=None, overlap=128):
        self.indf = indf
        # self.training_df = pd.read_parquet(self.path)
        self.training_df = indf
        print('Training Samples', len(self.training_df))
        self.training_df = indf.sort_values("path")
        self.training_df = self.training_df.head(30)
        print(self.training_df.head())
        
        self.training_frames = self.training_df.frames.tolist()
        self.training_imgs = self.training_df.path.tolist()

        self.length = len(self.training_imgs)

        
        self.overlap = overlap
        self.transform = transform

        # Get Images per batch
        img_file = self.training_imgs[0]
        img_file = '/data' + img_file
        
        image = PIL.Image.open(img_file).convert("RGB")

        wx, hy = image.size
        
        perm = product(
            np.arange(0, wx, self.overlap).astype(int),
            np.arange(0, hy, self.overlap).astype(int))
        
        self.tiles_per_frame = 0

        for idx, (x, y) in enumerate(perm):
            self.tiles_per_frame += 1


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
        
        if self.transform:
            imgs = [self.transform(animage) for animage in imgs]

    
        return imgs, imdf

    
def sorted_nicely(self, l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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

# Create this Second
class NSLatentVector(latentspace.LatentVectors):
    def __init__(self, dataname, dataquality, *args, dsetclass = NSFrame, **kwargs) -> None:
        '''
        Setup model and use functions to calculate RAIC vectors. The default
          model used to calculate vectors is Wideresnet50.
          
        INPUT
        remove_relu (default=False): whether the output vector should have
          the ReLu activation function (forcing the output to be >0) or not 
          (allowing the output to be -inf>x>inf).
        bit_model_name (default=None): can obtain the latent vectors from the 
          BiT-M or -S models by inputting a string with a valid model name.
          Input must be one of: 'BiT-M-R50x1', 'BiT-M-R50x3', 'BiT-M-R101x1', 
          'BiT-M-R101x3', 'BiT-M-R152x2', 'BiT-M-R152x4', 'BiT-S-R50x1', 
          'BiT-S-R50x3', 'BiT-S-R101x1', 'BiT-S-R101x3', 'BiT-S-R152x2', and 
          'BiT-S-R152x4'.
        custommodel (default=None): input a nn.Module class (custom pytorch 
          model) to generate RAIC vectors. The network must not require any
          additionial modifications.
        '''
        
        super(NSLatentVector, self).__init__(*args,**kwargs)
        self.dsetclass = dsetclass
        self.dataname = dataname
        self.dataquality = dataquality

        self.meta_save_dir = os.path.join(DATA_DIRECTORY, 'metatables')
        self.npy_save_dir = os.path.join(DATA_DIRECTORY, 'latents')

    def create_xy_folders(self, imdfs):
        # Latents 
        # print(imdfs[0])
        frame = [int(ss['frame']) for ss in imdfs]
        x = [ss['x1'] for ss in imdfs]
        y = [ss['y1'] for ss in imdfs]
        all_cuid = self.dataname
        # Create save dest directories
        for cf, cx in zip(frame, x):
            # NPY files 
            x_dir = os.path.join(self.npy_save_dir, all_cuid, 'high')
            if not os.path.exists(x_dir):
                os.makedirs(x_dir, exist_ok=True)
                
            meta_dir = os.path.join(self.meta_save_dir, all_cuid, 'high')
            if not os.path.exists(meta_dir):
                os.makedirs(meta_dir, exist_ok=True)

        np_save_paths = [os.path.join(self.npy_save_dir, all_cuid, 'high', f'frame_{cf}.npy') for cf in (frame)]
        
        return np_save_paths

    def get_vectors(self
                    , indf
                    , numWorkers=os.cpu_count()
                    , pathcolumnname = "path"
                    , num_images_per_batch=4
                    , write_parquets = True
                    , parquet_write_interval=100
                    , parquet_output_dir = './output') -> None:
        '''
        Calculate RAIC vectors.
        
        INPUT
        indf: an input data frame containing the columns: path
        numWorkers (default=24): pytorch data loader number of workers.
        pathcolumnname (default='path'): the name of the column in the 
          dataframe that contains the filename (full path) to the image.
        num_images_per_batch (default=512): batch_size
        write_parquets (default=False): whether vectos should be stored in
          a ndarray object or written to parquet files in a specified 
          directory.
        parquet_write_interval (default=100): the frequency at which vectors
          should be written to parquet files. The default is every 100
          batches.
        parquet_output_dir (default = None): the path to save resulting 
          parquet files (dataframes) that will be retrieved after all 
          vectors have been calculated.
          
        VALUE
        returns None; either stores the result in self.latvecs or writes the
          latent vectors to parquet files having the output:
          [xfms columns]: columns from the object provided by __getitem__()
            method as defined in the dataset class.
          [V]: array (numpy) of latent vector
        # '''
        
        # if not isinstance(pathcolumnname,str):
        #     raise ValueError("Path column name must be a string indicating the name of the column containing the path to each image.")
        # elif not set([pathcolumnname]).issubset(indf.columns):
        #     raise ValueError(f"Expected columns: `{pathcolumnname}`.")
        
        if (not parquet_output_dir) and write_parquets:
            raise ValueError("must provide an output directory for parquet files")
        elif write_parquets:
            os.makedirs(parquet_output_dir, exist_ok=True)
            self.parquet_output_dir = parquet_output_dir
        

        txf = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dset = self.dsetclass(indf=indf, transform=txf)
        self.dload = DataLoader(self.dset, batch_size=num_images_per_batch, shuffle=False, collate_fn=customcollate, num_workers=numWorkers, pin_memory=True, drop_last=False)

        with torch.no_grad():
            
            dimdf = []
            alltmpdf = []
            if write_parquets:
                # Will write parqutes in frame-batches, then will need to save them by frame npy files. versus the frame_0/x_0/y_0.npy format
                out = []
                for c,(dat, xfm) in enumerate(tqdm(self.dload)):
                    N = len(xfm)
                    dat = dat.to(self.device)
                    output = self.model(dat)
                    tmp = output.cpu().detach().numpy()

                    npy_files = self.create_xy_folders(xfm)
                    for ss, npy in zip(xfm, npy_files):
                        ss['npypath'] = npy

                    # latarray.append(tmp)
                    dimdf.append(xfm)
                    tmpdf = pd.DataFrame(xfm)
                    alltmpdf.append(pd.concat([tmpdf, pd.DataFrame({'V':[tmp[k,:] for k in range(N)]})],axis=1,sort=False))    


                    out.append(pd.concat([tmpdf, pd.DataFrame({'V':[tmp[k,:] for k in range(N)]})],axis=1,sort=False))
                    
                    if (((c+1)%parquet_write_interval) == 0):
                        out = pd.concat(out).reset_index(drop=True)
                        out.to_parquet(f'{self.parquet_output_dir}/batch_{str(c).zfill(9)}.parquet')
                        out = []
                        
                if (((c+1)%parquet_write_interval)!=0):
                    out = pd.concat(out).reset_index(drop=True)
                    out.to_parquet(f'{self.parquet_output_dir}/batch_{str(c).zfill(9)}.parquet')
                self.dimdf = pd.concat(alltmpdf).reset_index(drop=True)

            else:
                latarray = []
                latdictionary = {}

                for c,(dat, xfm) in enumerate(tqdm(self.dload)):
                    # Batch with (images * subtiles_per_image) size
                    N = len(xfm)
                    dat = dat.to(self.device)
                    output = self.model(dat)

                    tmp = output.cpu().detach().numpy()

                    npy_files = self.create_xy_folders(xfm)
                    for ss, npy in zip(xfm, npy_files):
                        ss['npypath'] = npy

                    # xfm['npypath'] = npy_files
                    
                    latarray.append(tmp)
                    dimdf.append(xfm)
                    tmpdf = pd.DataFrame(xfm)
                    alltmpdf.append(tmpdf)    

                self.dimdf = pd.concat(alltmpdf).reset_index(drop=True)

                # Used for later writing npy files
                self.latvecs = np.concatenate(latarray)
                
                # Split by frames
                # used for later writing metadata files.
                # self.dimdf = dimdf.reset_index(drop=True)

        try:
            torch.cuda.empty_cache()
        except:
            print('No cuda detected; i.e., no cache to clear.')
            
# GUID
# High
# Frame
# /video-display/metatable/GUID/high/frame/*frame.parquet
# /video-display/latents/GUID/high/frame/x/y.npy

# Create this First 
class NSDataPreprocessing():
    def __init__(self
                 , training_df:pd.DataFrame
                 , dataname:str ='17bbd414-4132-464a-bc1e-ab5dd66c7e8f' # GUID
                 , datasourcename:str = None
                 , dataquality='high' # Quality H,M,L
                 , dataframe='frames' # Frames
                 , valid_image_extensions = ['jpeg','jpg','png']
                 , default_dataset = NSFrame
                 , zlevelfilter:int = None
                 ) -> None:
        
        if not isinstance(valid_image_extensions, list):
            raise ValueError('`valid_image_extensions` must be a list of strings.')
        elif not all([isinstance(k,str) for k in valid_image_extensions]):
            raise ValueError('each element in `valid_image_extensions` must be a string')
        
        if datasourcename is None:
            self.datasourcename = dataname
        else:
            self.datasourcename = datasourcename
        
        self.dataname = dataname
        self.dataquality = dataquality
        self.dataframe = dataframe
        
        self.training_df = training_df

        self.img_datapath = os.path.join(DATA_DIRECTORY,self.dataname,self.dataquality)
        # self.datapath = os.path.join(DATA_DIRECTORY,self.dataname,self.dataquality,self.dataframe)
        self.meta_save_dir = os.path.join(DATA_DIRECTORY, 'metatables')
        self.npy_save_dir = os.path.join(DATA_DIRECTORY, 'latents')
 
        self.default_dataset_class = default_dataset
        self.default_dataloader_class = DataLoader # currently no way to change this
        
        self._datamovementcommands = {
            'push_to_azure':[]
            ,'pull_from_azure':[]
        }
    
    def calculate_raic_vectors(self, custom_model = None, bit_m_model = None, remove_ReLu=True,num_images_per_batch=None) -> None:
        if (not custom_model) and (not bit_m_model):
            bit_m_model = 'BiT-M-R101x1'
            print(f'\n\nUsing {bit_m_model} as RAIC model.\n\n')
        
        self.raic = NSLatentVector(self.dataname, self.dataquality, remove_relu=remove_ReLu, bit_model_name = bit_m_model, custommodel = custom_model)
        
            # Let the input be the path to the parquet ifle.

        self.raic.get_vectors(self.training_df, pathcolumnname='input')


    def write_df_chunk_npy(self, df):
        vectors = np.array(df.V.tolist())
        npy_path = df.npypath.values[0]
        frame = df.frame.values[0]

        with open(npy_path, 'wb') as npp:
            np.save(npp, vectors)

        updateddf = df.drop(columns=['V'])
        updateddf.to_parquet(os.path.join(self.meta_save_dir, self.dataname, 'high', f'frame_{frame}.parquet'))

        return 1

    def write_frame_latent_vector_arrays(self, valid_image_extensions=['jpeg','jpg','png']) -> None:
        df = self.raic.dimdf

        # df.npypath.apply(lambda x: os.path.dirname(x)).drop_duplicates().apply(lambda x: os.makedirs(x,exist_ok=True))

        print(f'Example file: {df.npypath.iloc[0]}')

        unique_frames = df.frame.unique()

        ret = [self.write_df_chunk_npy(df[df['frame'] == ss]) for ss in unique_frames]
        # for s in unique_frames:
            # ss = df[df['frame'] == s]
            

        with ThreadPoolExecutor(max_workers=os.cpu_count()*2) as exe:
            exe.map(self.write_df_chunk_npy, df.groupby('frame'))
            

    def write_latent_vector_arrays(self, valid_image_extensions=['jpeg','jpg','png']) -> None:
        '''
        Saved vectors must have the shape (2048,); i.e., (dim,)
        '''
        # df = self.raic.dimdf.assign(path = lambda x: x.path.str.replace(DATA_DIRECTORY, os.path.join(DATA_DIRECTORY,'latents')).str.replace('|'.join(valid_image_extensions),'npy',regex=True))
        df = self.raic.dimdf
        print(f'Example file: {df.path.iloc[0]}')
        
        df.npypath.apply(lambda x: os.path.dirname(x)).drop_duplicates().apply(lambda x: os.makedirs(x,exist_ok=True))
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()*2) as exe:
            exe.map(lambda x: np.save(df.npypath.iloc[x], self.raic.latvecs[x].squeeze()), range(len(df)))
            
        # print(f'Use the following azcopy command to move to azure:\nazcopy cp --recursive=true "{DATA_DIRECTORY}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/*" "{AZURE_STORAGE_ACCOUNT}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/{SAS_KEY}"')
        # print(f'\nUse the following azcopy command to pull from azure:\nazcopy cp --recursive=true "{AZURE_STORAGE_ACCOUNT}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/*{SAS_KEY}" "{DATA_DIRECTORY}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/"')
        
        self._datamovementcommands['push_to_azure'].append(f'azcopy cp --recursive=true "{DATA_DIRECTORY}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/*" "{AZURE_STORAGE_ACCOUNT}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/{SAS_KEY}"')
        self._datamovementcommands['pull_from_azure'].append(f'azcopy cp --recursive=true "{AZURE_STORAGE_ACCOUNT}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/*{SAS_KEY}" "{DATA_DIRECTORY}/latents/{self.dataname}/{self.dataquality}/{self.dataframe}/"')
    
    
    def finalize_save_meta_tables(self, valid_image_extensions=['jpeg','jpg','png']):
        '''
        need localpath,blobpath,locallatentpath,bloblatentpath,localautotrainlatent,blobautotrainlatent,locallatent,bloblatent
        '''
        # try:
        self.dimdf = self.raic.dimdf
        # except:
        #     raise ValueError('Finalizing Dim tables requires that ball-trees were calculated. Future work will resolve this issue.')
        
        # Finalize meta table by taking the batched (can contain multiple frames) and saving by frames.
        meta_dir = os.path.join(self.meta_save_dir, self.dataname, self.dataquality)

        self.uniqueframes = self.dimdf.frame.unique()
        for fr in self.uniqueframes:
            aframedf = self.dimdf[self.dimdf['frame'] == fr]
            frameparquet = os.path.join(meta_dir, f'{str(fr)}.parquet')
            df = aframedf
            df = df.rename(columns={'path':'localpath'})
            df['blobpath'] = df.localpath.str.replace(DATA_DIRECTORY,AZURE_STORAGE_ACCOUNT)
            df['locallatentpath'] = df.localpath.str.replace(DATA_DIRECTORY, os.path.join(DATA_DIRECTORY,'latents')).str.replace('|'.join(valid_image_extensions),'npy',regex=True)
            df['bloblatentpath'] = df.localpath.str.replace(DATA_DIRECTORY, os.path.join(AZURE_STORAGE_ACCOUNT,'latents')).str.replace('|'.join(valid_image_extensions),'npy',regex=True)
            df['localautotrainlatent'] = df.localpath.str.replace(DATA_DIRECTORY, os.path.join(DATA_DIRECTORY,'autotrainlatents')).str.replace('|'.join(valid_image_extensions),'npy',regex=True)
            df['blobautotrainlatent'] = df.localpath.str.replace(DATA_DIRECTORY, os.path.join(AZURE_STORAGE_ACCOUNT,'autotrainlatents')).str.replace('|'.join(valid_image_extensions),'npy',regex=True)
            df['locallatent'] = df.locallatentpath
            df['bloblatent'] = df.bloblatentpath
            df['blobimage'] = df.blobpath
            
            df.to_parquet(frameparquet)     

        self._datamovementcommands['push_to_azure'].append(f'azcopy cp --recursive=true "{DATA_DIRECTORY}/metatable/{self.dataname}/{self.dataquality}/{self.dataframe}/*" "{AZURE_STORAGE_ACCOUNT}/metatable/{self.dataname}/{self.dataquality}/{self.dataframe}/{SAS_KEY}"')
        self._datamovementcommands['pull_from_azure'].append(f'azcopy cp --recursive=true "{AZURE_STORAGE_ACCOUNT}/metatable/{self.dataname}/{self.dataquality}/{self.dataframe}/*{SAS_KEY}" "{DATA_DIRECTORY}/metatable/{self.dataname}/{self.dataquality}/{self.dataframe}/"')
       

    def transfer_raic_data_to_geospatial_container(self,transfer_data=False) -> None:
        if not transfer_data:
            print('\nTo transfer data, set `transfer_data=True`.\n')
            print('IMPORTANT: if transfer_data=True, The following commands will be run using `os.system()` in\n\torder to transfer data to azure blob storage:')
            print('\n\n'.join(self._datamovementcommands['push_to_azure']))
            print('\n\nUse the following to pull the resulting data FROM azure (to the production VM):')
            print('\n\n'.join(self._datamovementcommands['pull_from_azure']))
            return None
        else:
            print('IMPORTANT: Running the following commands using `os.system()`:')
            print('\n\n'.join(self._datamovementcommands['push_to_azure']))
            return [os.system(k) for k in self._datamovementcommands['push_to_azure']]
            

if __name__ == "__main__":
    training_file = "/home/azureuser/data/onlyaf3c.parquet"
    training_df = pd.read_parquet(training_file)

    ns = NSDataPreprocessing(training_df)