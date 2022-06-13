import numpy as np
import torch
#from torchvision import transforms
import torch.utils.data as torchdata
#import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, '../')
from conf import PATH_DATA
from pathlib import Path
import h5py

import flag


rain_const = 28
waterdepth_diff_const = 0.01


def load_dataset_file(filename):
    with h5py.File(filename, 'r') as f:
        dem = f["dem"][()]
        mask = f["mask"][()]
        peak = f["peak"][()]
        start_ts = f["start_ts"][()]
        event_name = f["event_name"][()]
        rainfall_events = []

        for k in filter(lambda x: "rainfall_events"==x[:15],  sorted(f.keys())):
            rainfall_events.append(f[k][()])
        return peak, rainfall_events, start_ts, dem, mask, event_name

def load_catch_dataset(args):
    
    # data_transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    # to be modified because it does not generalize
    base_path = Path(PATH_DATA) / Path("{}/data_npy".format(args.data))

    dataloaders = {}

    it = MyCatchment(args, "train",base_path / Path("train_max.h5"), fix_indexes = args.fix_indexes)
    dataloaders["train"] = torchdata.DataLoader(it, 
                                                batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                shuffle=True,
                                                drop_last=True)
    
    if args.fix_indexes_val == False:
        it2 = MyCatchment(args, "val", base_path / Path("val_max.h5"), fix_indexes = args.fix_indexes)
    else:
        it2 = MyCatchmentVal(args, "val", base_path / Path("val_max.h5"), fix_indexes = args.fix_indexes_val)
    
    dataloaders["val"] = torchdata.DataLoader(it2,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              drop_last=True)

    data_stats = it.data_stats

    return dataloaders, data_stats
                                

def load_catch_dataset_test(args):

    base_path = Path(PATH_DATA) / Path("{}/data_npy".format(args.data))

    dataloaders = {}

    if not(args.data== "toy"):
        it = MyCatchment(args, "test", base_path / Path("test_max.h5"), fix_indexes=True) #, transforms=data_transform)
        dataloaders["test"] = torchdata.DataLoader(it, 
                                                batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                shuffle=False,
                                                drop_last=True)

    data_stats = it.data_stats

    return dataloaders, data_stats


class MyCatchment(torch.utils.data.Dataset):
    
    def __init__(self, args, set, h5file, fix_indexes=False, transforms = None):
        'Initialization'

        # this is not the way it should be, but let us fix it once we go for more than one time step
        
        #self.input_channels = 3 + (args.timestep-1) * 2
        self.input_channels = 14 # 12 rain + dem + start_ts
        self.data_stats = {"input_channels": self.input_channels,
              "output_channels": 1}
        

        self.transforms = transforms
        self.set = set
        
        self.h5file = h5py.File(h5file, "r")
        keys = sorted(self.h5file.keys())
        self.dem = torch.tensor(self.h5file["dem"][()]).float()
        self.dem_mask = torch.tensor(self.h5file["mask"][()]).bool()
        self.start_ts = torch.tensor(self.h5file["start_ts"][()]).float()
        self.peak = torch.tensor(self.h5file["peak"][()]).float()
        self.rainfall_events = []
        for k in filter(lambda x: "rainfall_events"==x[:15], keys ):
            self.rainfall_events.append(torch.tensor(self.h5file[k][()]/rain_const).float())
        
        self.N_events = self.peak.shape[0]
        self.px = self.start_ts.shape[0]
        self.py = self.start_ts.shape[1]
        #self.T_steps = [len(x) for x in self.rainfall_events]
        self.rain_c = args.rain_c
        #self.timestep = args.timestep
        #self.predict_ahead = args.predict_ahead
        #self.ts_out = args.ts_out

        self.sample_type = args.sample_type
        if self.sample_type == "full":
            self.fix_indexes = False
        else:
            self.fix_indexes = fix_indexes
            
        self.tau = args.tau
        self.change_tau = args.change_tau
        self.nx = args.dim_patch
        self.ny = args.dim_patch
        self.num_patch = args.num_patch
        
        self.build_indexes()
        
        if self.fix_indexes:
            self.find_all_fix_indexes()   # returns all possible incexes without random generation
        else:
            self.inds = None

        assert(len(self.rainfall_events)==self.N_events)
        assert(self.dem.shape[0]==self.px)
        assert(self.dem.shape[1]==self.py)        
        assert(self.dem_mask.shape[0]==self.px)
        assert(self.dem_mask.shape[1]==self.py)      

    def build_indexes(self):
        self.indexes_e = []
        self.indexes_t = []

        self.indexes_e.append(np.arange(self.N_events))

        self.indexes_e = np.concatenate(self.indexes_e)
        if self.num_patch > 1 and self.fix_indexes == False:
            self.indexes_e = np.tile(self.indexes_e, self.num_patch)
        self.N_batch_images = len(self.indexes_e)
        
        
    def __exit__(self, *args):
        self.h5file.close()
        
    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample

        if self.fix_indexes:    ### accessing event (tr), T (timestep for each file) and indexes for each file   
            tmp = index // len(self.inds)
            index_e = self.indexes_e[tmp] 
            index_s = index % len(self.inds)  
            index_t = []
        else:
            ### accessing event (tr) and T (timestep for each file)       
            index_t = []
            index_e = self.indexes_e[index] 
            index_s = 0
            
        if self.sample_type == "full":
            xin = self.waterdepth[index_e][index_t: index_t + self.timestep]
            xout = self.waterdepth[index_e][index_t + self.timestep]
            mask = self.dem_mask
            dem = self.dem
            
        if self.sample_type == "single":
            if self.fix_indexes:
                x_p, y_p = self.inds[index_s]
            else:
                x_p, y_p = self.find_patch()
                
            xin = self.start_ts[x_p:x_p+self.nx, y_p:y_p+self.ny]
            xout = self.peak[index_e][x_p:x_p+self.nx, y_p:y_p+self.ny]

            mask = self.dem_mask[x_p:x_p+self.nx, y_p:y_p+self.ny]
            dem = self.dem[x_p:x_p+self.nx, y_p:y_p+self.ny]
        
        #y = (xout - xin[-1]) / waterdepth_diff_const   ###############################
        y = xout
        y = y.unsqueeze(0)
        x = torch.zeros((self.input_channels, *dem.shape))
        
        x[0] = dem.clone()
        for i in range(self.rain_c):    # rainfall channels always 12
            x[1+i, mask] = self.rainfall_events[index_e][i]     # creates channels for rainfall
        x[1+self.rain_c] = xin

        mask = mask.unsqueeze(0).clone()
        #print(x.shape)
        
#         return {'data': x,'y':  y, 'mask': mask}
        return {'data': x,'gt':  y, 'mask': mask, 'index_e':index_e, 'timestep':index_t, 'index_s':index_s, 'x_p': x_p, 'y_p':y_p}  # to access event, timestep, current patch (index_s) and coordinates of patch
        
    def upd_tau(self):
        epoch = flag.epoch
        if epoch > 200 and epoch < 350:
            self.tau = 0.6
        elif epoch >= 350 and epoch < 450:
            self.tau = 0.7
        elif epoch >= 450:
            self.tau = 0.8
    
    def find_patch(self):
        '''
        create patches with at least a fraction of tau non-zero elements
        '''
        #if self.set == "val":
        #    self.tau = 10/(256*256)

        if self.change_tau == True:
            self.upd_tau()

        z_patch = torch.zeros(self.nx, self.ny)
        while (torch.sum(z_patch) < (self.nx*self.ny * self.tau)):   
            x_p = np.random.randint(self.px-self.nx-1)   
            y_p = np.random.randint(self.py-self.ny-1)
            z_patch = self.dem_mask[x_p:(x_p + self.nx), y_p:(y_p + self.ny)]
        return x_p, y_p
    
    def find_all_fix_indexes(self):
        indx =  np.arange(0, self.px-self.nx-1, np.int(self.nx))
        indy = np.arange(0, self.py-self.ny-1, np.int(self.ny))
        x, y = np.meshgrid(indx,indy)
        self.inds = np.array([x.flatten(), y.flatten()]).T
        
    
    def __len__(self):
        'Denotes the total number of samples'
        if self.fix_indexes:
            return self.N_batch_images * len(self.inds)
        else:
            return self.N_batch_images


##### val dataloader

class MyCatchmentVal(torch.utils.data.Dataset):

    def __init__(self, args, set, h5file, fix_indexes=False, transforms=None):
        'Initialization'

        # this is not the way it should be, but let us fix it once we go for more than one time step

        self.input_channels = 3 + (args.timestep - 1) * 2
        self.data_stats = {"input_channels": self.input_channels,
                           "output_channels": 1}

        self.transforms = transforms
        self.set = set

        self.h5file = h5py.File(h5file, "r")
        keys = sorted(self.h5file.keys())
        self.dem = torch.tensor(self.h5file["dem"][()]).float()
        self.dem_mask = torch.tensor(self.h5file["mask"][()])
        self.rainfall_events = []
        for k in filter(lambda x: "rainfall_events" == x[:15], keys):
            self.rainfall_events.append(torch.tensor(self.h5file[k][()] / rain_const).float())
        in_memory = False
        self.waterdepth = []
        for k in filter(lambda x: "waterdepth" == x[:10], keys):
            if in_memory:
                self.waterdepth.append(self.h5file[k][()])
            else:
                self.waterdepth.append(self.h5file[k])
        self.N_events = len(self.waterdepth)
        self.px = self.waterdepth[0].shape[1]
        self.py = self.waterdepth[0].shape[2]
        self.T_steps = [len(x) for x in self.rainfall_events]
        self.timestep = args.timestep

        self.sample_type = args.sample_type
        if self.sample_type == "full":
            self.fix_indexes = False
        else:
            self.fix_indexes = fix_indexes

        self.tau = args.tau
        self.nx = args.dim_patch
        self.ny = args.dim_patch
        self.num_patch = args.num_patch

        self.build_indexes()

        if self.fix_indexes:
            self.find_all_fix_indexes()  # returns all possible incexes without random generation
        else:
            self.inds = None

        assert (len(self.rainfall_events) == self.N_events)
        for i in range(self.N_events):
            assert (self.rainfall_events[i].shape[0] == self.waterdepth[i].shape[0] == self.T_steps[i])
        assert (self.dem.shape[0] == self.px)
        assert (self.dem.shape[1] == self.py)
        assert (self.dem_mask.shape[0] == self.px)
        assert (self.dem_mask.shape[1] == self.py)

        if self.timestep > 2:
            raise NotImplementedError('Timestep = 2 has not been implemented')

    def build_indexes(self):
        self.indexes_t = []
        self.indexes_e = []
        v = 0
        for i, nt in enumerate(self.T_steps):
            nv = nt - self.timestep
            ## repeating 11 timesteps (3000-6000 timesteps)
            ## by adding 11 to nv in indexes_t
            if nv > 0:
                t = np.arange((nv))
                ## append the repeated timesteps of(3000-6000)
                t = np.append(t, np.arange(10, 21))
                ## sort the list
                t.sort()
                self.indexes_t.append(t)

                self.indexes_e.append(np.ones([nv+11], dtype=int) * i)
                v += (nv+11)
        self.indexes_t = np.concatenate(self.indexes_t)
        self.indexes_e = np.concatenate(self.indexes_e)
        if self.num_patch > 1 and self.fix_indexes == False:
            self.indexes_t = np.tile(self.indexes_t, self.num_patch)
            self.indexes_e = np.tile(self.indexes_e, self.num_patch)
        self.N_batch_images = len(self.indexes_t)

        # Here a shuffling would be easy

    def __exit__(self, *args):
        self.h5file.close()

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample

        if self.fix_indexes:  ### accessing event (tr), T (timestep for each file) and indexes for each file
            tmp = index // len(self.inds)
            index_e = self.indexes_e[tmp]
            index_t = self.indexes_t[tmp]
            index_s = index % len(self.inds)
        else:
            ### accessing event (tr) and T (timestep for each file)
            index_e = self.indexes_e[index]
            index_t = self.indexes_t[index]
            index_s = 0

        if self.sample_type == "full":
            xin = torch.from_numpy(self.waterdepth[index_e][index_t: index_t + self.timestep]).float()
            xout = torch.from_numpy(self.waterdepth[index_e][index_t + self.timestep]).float()
            mask = self.dem_mask
            dem = self.dem

        if self.sample_type == "single":
            if self.fix_indexes:
                x_p, y_p = self.inds[index_s]
            else:
                x_p, y_p = self.find_patch()

            xin = torch.from_numpy(self.waterdepth[index_e][index_t: index_t + self.timestep, x_p:x_p + self.nx,
                                   y_p:y_p + self.ny]).float()
            xout = torch.from_numpy(
                self.waterdepth[index_e][index_t + self.timestep, x_p:x_p + self.nx, y_p:y_p + self.ny]).float()
            mask = self.dem_mask[x_p:x_p + self.nx, y_p:y_p + self.ny]
            dem = self.dem[x_p:x_p + self.nx, y_p:y_p + self.ny]

        # y = (xout - xin[-1]) / waterdepth_diff_const   ###############################
        y = xout
        y = y.unsqueeze(0)
        x = torch.zeros((self.input_channels, *dem.shape))

        x[0] = dem.clone()
        x[1] = self.rainfall_events[index_e][index_t]
        x[2:2 + self.timestep] = xin
        if self.timestep > 1:
            x[-self.timestep + 1:] = torch.diff(xin, axis=0) / waterdepth_diff_const
        mask = mask.unsqueeze(0).clone()

        #         return {'data': x,'y':  y, 'mask': mask}
        return {'data': x, 'gt': y, 'mask': mask, 'index_e': index_e, 'timestep': index_t, 'index_s': index_s,
                'x_p': x_p, 'y_p': y_p}  # to access event, timestep, current patch (index_s) and coordinates of patch

    def find_patch(self):
        '''
        create patches with at least a fraction of tau non-zero elements
        '''
        if self.set == "val":
            factor = 10 #self.nx * self.ny * self.tau

        z_patch = torch.zeros(self.nx, self.ny)
        while (torch.sum(z_patch) < (factor)):
            x_p = np.random.randint(self.px - self.nx - 1)
            y_p = np.random.randint(self.py - self.ny - 1)
            z_patch = self.dem_mask[x_p:(x_p + self.nx), y_p:(y_p + self.ny)]
        return x_p, y_p

    def find_all_fix_indexes(self):
        indx = np.arange(0, self.px - self.nx - 1, self.nx)
        indy = np.arange(0, self.py - self.ny - 1, self.ny)
        x, y = np.meshgrid(indx, indy)
        self.inds = np.array([x.flatten(), y.flatten()]).T

    def __len__(self):
        'Denotes the total number of samples'
        if self.fix_indexes:
            return self.N_batch_images * len(self.inds)
        else:
            return self.N_batch_images  # * self.num_patch


