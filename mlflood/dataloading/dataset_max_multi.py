import numpy as np
import torch
from torch._C import Value
from torch.nn.functional import pad
import torch.utils.data as torchdata
from conf import PATH_GENERATED
from pathlib import Path
import h5py
from torch.utils.data import TensorDataset, DataLoader

print(h5py.__version__)
from conf import rain_const, waterdepth_diff_const

def normalize(x):
    
    return x/waterdepth_diff_const

def unnormalize(x):
    
    return x*waterdepth_diff_const
    

def load_dataset(catchment_kwargs):
    
    catchment_num = catchment_kwargs['num']
    train_dataset = []
    for i in catchment_num:
        a = MyCatchment(PATH_GENERATED / Path(str(i)+"-train.h5"), 
                                tau = catchment_kwargs["tau"] ,
                                timestep = catchment_kwargs["timestep"],
                                sample_type = catchment_kwargs["sample_type"],
                                dim_patch = catchment_kwargs["dim_patch"],
                                fix_indexes = catchment_kwargs["fix_indexes"],
                                normalize_output = catchment_kwargs["normalize_output"],
                                use_diff_dem = catchment_kwargs["use_diff_dem"],
                                num_patch = catchment_kwargs["num_patch"],
                                predict_ahead = catchment_kwargs["predict_ahead"])
        train_dataset.append(a)
    
    train_sets = torch.utils.data.ConcatDataset(train_dataset)
    
    catchment_num_val = catchment_kwargs['val']
    valid_dataset = []

    for j in catchment_num_val:
        b = MyCatchment(PATH_GENERATED / Path(j+"-val.h5"), 
                                tau = catchment_kwargs["tau"] ,
                                timestep = catchment_kwargs["timestep"],
                                sample_type = catchment_kwargs["sample_type"],
                                dim_patch = catchment_kwargs["dim_patch"],
                                fix_indexes = catchment_kwargs["fix_indexes"],
                                normalize_output = catchment_kwargs["normalize_output"],
                                use_diff_dem = catchment_kwargs["use_diff_dem"],
                                num_patch = catchment_kwargs["num_patch"],
                                predict_ahead = catchment_kwargs["predict_ahead"])
        valid_dataset.append(b)
    
    val_sets = torch.utils.data.ConcatDataset(valid_dataset)
    
    return train_sets, val_sets

def load_test_dataset(catchment_kwargs):
    
    catchment_num = catchment_kwargs['num']
    dataset = MyCatchment(PATH_GENERATED / Path(catchment_num+"-test.h5"), 
                                tau = catchment_kwargs["tau"] ,
                                timestep = catchment_kwargs["timestep"],
                                sample_type = catchment_kwargs["sample_type"],
                                dim_patch = catchment_kwargs["dim_patch"],
                                fix_indexes = catchment_kwargs["fix_indexes"],
                                normalize_output = catchment_kwargs["normalize_output"],
                                use_diff_dem = catchment_kwargs["use_diff_dem"],
                                num_patch = catchment_kwargs["num_patch"],
                                predict_ahead = catchment_kwargs["predict_ahead"])
    return dataset



def dataloader_args(train_dataset, valid_dataset, catchment_num = "toy", batch_size = 8):
    
    dataloaders_train = torchdata.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
    
    dataloaders_valid = torchdata.DataLoader(valid_dataset, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)
    
    return dataloaders_train, dataloaders_valid

def dataloader_args_test(train_dataset, catchment_num = "toy", batch_size = 8):
    
    dataloaders_train = torchdata.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True)
    
    return dataloaders_train


def pad_borders(waterdepth, dem, mask, border_size):
    
    dem = pad(dem, 4*[border_size], mode='constant', value=-1)
    mask = pad(mask, 4*[border_size], mode='constant', value=False) 
    dem[mask==False] = -1
    waterdepth = pad(waterdepth, 4*[border_size], mode='constant', value=0)
    return waterdepth, dem, mask

def build_diff_dem(dem):
    
    px, py = dem.shape
    dx1 = torch.unsqueeze(torch.diff(dem, prepend=torch.tensor([[-1]*py]), dim=0), 0)
    dx2 = torch.unsqueeze(torch.diff(dem, append=torch.tensor([[-1]*py]), dim=0), 0)
    dy1 = torch.unsqueeze(torch.diff(dem, prepend=torch.tensor([[-1]]*px), dim=1), 0)
    dy2 = torch.unsqueeze(torch.diff(dem, append=torch.tensor([[-1]]*px), dim=1), 0)
    return torch.cat([dx1,dx2,dy1,dy2])

class MyCatchment(torch.utils.data.Dataset):
    
    def __init__(self,  h5file, tau=0.5, upsilon=0, timestep=1, sample_type="single", dim_patch=64, fix_indexes=False, border_size=0, normalize_output = False, use_diff_dem=True, num_patch = 10, predict_ahead = 0):
        '''
        Initialization
        '''

        # this is not the way it should be, but let us fix it once we go for more than one time step
        if not(sample_type in ["single",  "full"]):
            raise ValueError("Unknown sample_type")
        self.timestep = timestep
        self.border_size = border_size
        self.do_pad = True
        self.normalize_output = normalize_output
        self.use_diff_dem = use_diff_dem
        self.predict_ahead = predict_ahead

        print(f"Load file: {h5file}")
        self.h5file = h5py.File(h5file, "r")
        keys = sorted(self.h5file.keys())
        self.dem = self.pad_borders(torch.tensor(self.h5file["dem"][()]).float(), -1)
        self.dem_mask = self.pad_borders(torch.tensor(self.h5file["mask"][()]).bool(), False)
        self.dem[self.dem_mask==False] = -1

        self.peak = self.pad_borders(torch.tensor(self.h5file["peak"][()]).float(), -1)
        self.start_ts = self.pad_borders(torch.tensor(self.h5file["start_ts"][()]).float(), -1)
        
        self.diff_dem = build_diff_dem(self.dem)
        
        self.rainfall_events = []
        for k in filter(lambda x: "rainfall_events"==x[:15], keys ):
            self.rainfall_events.append(torch.tensor(self.h5file[k][()]).float())
        self.input_channels = 14 
        self.data_stats = {"input_channels": self.input_channels,
              "output_channels": 1}

        self.N_events = self.peak.shape[0]
        self.px = self.start_ts.shape[0]
        self.py = self.start_ts.shape[1]
        self.T_steps = [len(x) for x in self.rainfall_events]
        

        self.sample_type = sample_type
        if self.sample_type == "full":
            self.fix_indexes = False
        else:
            self.fix_indexes = fix_indexes
            
        self.tau = tau
        self.upsilon = upsilon
        self.nx = dim_patch
        self.ny = dim_patch
        self.num_patch = num_patch
        
        self.curr_index = 0
        self.x_p = 0
        self.y_p = 0

        self.build_indexes()

        assert(len(self.rainfall_events)==self.N_events)
        assert(self.dem.shape[0]==self.px)
        assert(self.dem.shape[1]==self.py)        
        assert(self.dem_mask.shape[0]==self.px)
        assert(self.dem_mask.shape[1]==self.py)
        if not(self.sample_type == "full"): 
            assert(self.nx<=self.px)
            assert(self.ny<=self.py)
        
    def pad_borders(self, x, value):
        
        return pad(x, 4*[self.border_size], mode='constant', value=value)
    
    def build_indexes(self):
        '''
        Builds indexes for timesteps, events to be returned
        '''

        self.indexes_t = []
        self.indexes_e = []

        self.indexes_e.append(np.arange(self.N_events))
        self.indexes_e = np.concatenate(self.indexes_e)

        # for more than 1 patch per timestep
        if self.num_patch > 1 and self.fix_indexes == False:
            self.indexes_e = np.tile(self.indexes_e, self.num_patch)
        self.N_batch_images = len(self.indexes_e)

        if self.fix_indexes:
            self.inds = self.get_all_fix_indexes()   # returns all possible incexes without random generation
        else:
            self.inds = 0                  # cannot return None        
        # Here a shuffling would be easy
        
    def __exit__(self, *args):

        self.h5file.close()

    def __getitem__(self, index):
        '''
        Generates one sample of data by accessing accessing event (index_e), 
        timestep for each file (index_t) and indexes (index_s) for each file 
        
        if fix_indexes = True, data is generated sequentially
        '''

        if self.fix_indexes:      
            tmp = index // len(self.inds)
            index_e = self.indexes_e[tmp] 
            index_t = self.indexes_t[tmp]
            index_s = index % len(self.inds)  
            
            self.curr_index = index_s
            self.x_p, self.y_p = self.inds[index_s]
            
        else:     
            index_e = self.indexes_e[index] 
            index_t = []
            index_s = None
        
        # select input data. based on current index_e, index_t and number of timesteps
        xin = self.start_ts
        xout = self.peak[index_e]
        mask = self.dem_mask
        dem = self.dem
        diff_dem = self.diff_dem
        if self.sample_type == "single":

            if self.fix_indexes:
                x_p, y_p = self.inds[index_s]    # finds adjacent coordinates
            else:
                x_p, y_p = self.find_patch(mask, xout, index_t)
            xin, mask, dem, diff_dem, xout = self.crop_to_patch(x_p, y_p, xin, mask, dem, diff_dem, xout)

        outputs = self.build_output(xout, xin, mask)
        
        inputs = self.build_inputs(xin, self.rainfall_events[index_e], mask, dem, diff_dem)
        
        return inputs, outputs   
    
    def crop_to_patch(self, x_p, y_p, xin, mask, dem, diff_dem, xout=None):
        '''
        Crops a patch of xin, mask, dem, xout based on current coordinates x_p, y_p
        '''

        xin = xin[x_p:x_p+self.nx, y_p:y_p+self.ny]
        diff_dem = diff_dem[:, x_p:x_p+self.nx, y_p:y_p+self.ny]
        mask = mask[x_p:x_p+self.nx, y_p:y_p+self.ny]
        dem = dem[x_p:x_p+self.nx, y_p:y_p+self.ny]
        if xout is not None:
            xout = xout[x_p:x_p+self.nx, y_p:y_p+self.ny]
        
        return xin, mask, dem, diff_dem, xout
    
    def build_output(self, xout, xin, mask):
        '''
        Builds ground truth and returns  [ground truth, mask]
        
        Optional: ground truth can be normalized
        If border_size, ground truth is cropped accordingly
        '''
        
        if self.normalize_output:
            y = normalize(xout - xin[-1])  
        else: 
            y = xout
        if self.border_size:
            border = self.border_size
            y = y[border:-border, border:-border]
            mask = mask[border:-border, border:-border]
        y = y.unsqueeze(0)
        mask = mask.unsqueeze(0)
        outputs = torch.cat([y, mask])        
        #return outputs
        
        return y
    
    def build_inputs(self, xin, rainfall_events, mask, dem, diff_dem):

        '''
        Builds input channels for the model and returns (inputs, mask)
        Channels are: dem, rainfall, wd@(timestep-1)
        
        Optional: if timestep>2, the number of channels is increased accordingly and normalizes
        '''
        rain_ch = rainfall_events.shape[0]
        rain_ch = 12
        rain_t = torch.zeros((rain_ch, *dem.shape))
        for i in range(rain_ch):    
            rain_t[i, mask] = rainfall_events[i]     # creates channels for rainfall
        
        d = 4 if self.use_diff_dem else 0
        x = torch.zeros((self.input_channels, *dem.shape))
        x[0] = dem.clone()
        if self.use_diff_dem:
            x[1:5] = diff_dem
        x[1+d : 1+d + rain_ch] = rain_t
        ## remove xin and add mask as feature
        x[1+d + rain_ch] = mask #xin
        mask = mask.unsqueeze(0).clone()
        
        return (x, mask)

    def find_patch(self, mask, xout, index_t=None):
        '''
        create patches with:
        - at least a fraction of tau non-zero elements in mask AND
        - at least a average water depth value within those elements larger than upsilon (in m) 
        '''

        z_patch = torch.zeros(self.nx, self.ny)
        xout_patch = torch.zeros(self.nx, self.ny)
        border = self.border_size if self.do_pad else 0
        bool_wd = True

        while ((torch.sum(z_patch) < (self.nx*self.ny * self.tau)) or bool_wd): 
            
            x_p = np.random.randint(0, self.px-self.nx+1)
            y_p = np.random.randint(0, self.py-self.ny+1)
            
            xout_patch = xout[x_p:(x_p + self.nx+border*2), y_p:(y_p + self.ny+border*2)]  
            
            # compute z_patch and bool_wd for while statement
            z_patch = mask[x_p:(x_p + self.nx+border*2), y_p:(y_p + self.ny+border*2)]
            # check the average wd in the patch > upsilon
            #bool_wd = (torch.sum(xout_patch)/torch.sum(mask)) < self.upsilon
            a = xout_patch > 0.2
            bool_wd = len(xout_patch[a]) < 1 # the patch has atleast 1000 pixels with wd > 20 cm  
            
        return x_p, y_p

    def get_all_fix_indexes(self, non_full=False):
        '''
        use this function if every patch is padded with border = border_size
        finds indeces to create adjacent overlapping patches
        the patches need to move steps = nx - border_size*2 to guarantee complete reconstruction of the catchment
        nx = ny = dim_patch
        px = waterdepth.shape[0]
        py = waterdepth.shape[1]
        '''

        border = self.border_size if self.do_pad else 0
        if non_full:
            mx = self.px - 2*border 
            my = self.py - 2*border 
        else:
            mx = self.px - self.nx + 1 
            my = self.py - self.ny + 1
        indx =  np.arange(0, mx, self.nx-border*2)    
        indy = np.arange(0, my, self.ny-border*2)         
        x, y = np.meshgrid(indx,indy)
        
        return np.array([x.flatten(), y.flatten()]).T

    def denormalize_output(self, y):
        
        if self.normalize_output:
            return unnormalize(y)
        
        return y
        
    
    def __len__(self):
        '''
        Denotes the total number of samples
        '''

        if self.fix_indexes:
            return self.N_batch_images * len(self.inds)
        else:
            return self.N_batch_images