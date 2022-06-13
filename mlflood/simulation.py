import numpy as np
from skimage.filters import median
import h5py
from conf import PATH_GENERATED, rain_const
from pathlib import Path
from skimage.transform import resize

viscosity = 1

def create_random_dem(px, py, seed=None):
    """Create a random terrain."""
    rnd_array = np.random.rand(32, (32*py)//px)
    dem = median(median(rnd_array))
    dem = dem - np.min(dem)
    dem = dem/ np.maximum(np.max(dem), 1e-5)
    dem = resize(dem, [px, py], anti_aliasing=True)
    return dem



def one_step(cwd, rf, dem, test=False):
    """One step of simulation."""

    # 1. Compute relative flow from terrain
    # Border conditions
    dem_wb = -1*np.ones([dem.shape[0]+2, dem.shape[1]+2])
    dem_wb[1:-1, 1:-1] = dem

    # Diff of terrain
    dx = np.diff(dem_wb, axis=0)[:,1:-1]
    dy = np.diff(dem_wb, axis=1)[1:-1,:]
    # Build a array of value
    v = np.array([viscosity*np.ones(dem.shape), 
                dx[:-1,:], 
                - dx[1:,:],
                dy[:,:-1],
                - dy[:,1:]
                ])
    # # Relu
    # l = np.maximum(v, 0)
    l = np.maximum(np.exp(v) - 0.9,0)
    
    # Split the flow
    p = l / np.sum(l, axis=0, keepdims=True)
    if test:
        np.testing.assert_allclose(np.sum(p, axis=0), 1)

    # 2. Move the water around
    leakage = 0
    flows = (cwd+rf) * p

    # Left-over value
    nwd = flows[0]
    # Receive from right edge
    nwd[:-1,:] += flows[1,1:,:]
    leakage += np.sum(flows[1,0,:]) 
    # Receive from left edge
    nwd[1:,:] += flows[2,:-1,:]
    leakage += np.sum(flows[2,-1,:]) 
    # Receive from bottom edge
    nwd[:,:-1] += flows[3,:,1:]
    leakage += np.sum(flows[3,:,0]) 
    # Receive from top edge
    nwd[:,1:] += flows[4,:,:-1]
    leakage += np.sum(flows[4,:,-1]) 
    return nwd, leakage

def simulate(rainfall, dem):
    """Perform one simulation from zero conditions."""

    cwd = np.zeros(dem.shape)
    leakage = []
    wd = []
    wd.append(cwd)
    for rf in rainfall:
        cwd, l = one_step(cwd, rf, dem)
        wd.append(cwd)
        leakage.append(l)
    return np.array(wd), np.array(leakage)


def create_sims_dataset(nx = 256, ny = 512, splits=[30, 5, 10], genname="sims"):
    """Create the sims datasets."""
  
    def make_one_simulation(T, dem):
        """helping function for one simulation (wrapper around simulate)."""

        # produce rainfall
        t = T/10+np.random.rand()*T/3
        var = T/5+np.random.rand()*T/2
        x = np.arange(T-1)
        rainfall = np.exp(- (x-t)**2 /var)/28
        # simulate
        wd, leakage = simulate(rainfall, dem)
        rainfall = np.concatenate([rainfall, np.array([0])])
        leakage = np.concatenate([np.array([0]), leakage])
        
        # convert to float 32
        return wd.astype(np.float32), rainfall.astype(np.float32), leakage.astype(np.float32)


    for N_events, suffix in zip(splits, ["train", "val", "test"]):
        name = genname+"-"+suffix
        with h5py.File( PATH_GENERATED / Path(name+".h5"), 'w') as f:
            # Dem
            dem = create_random_dem(nx, ny).astype(np.float32)
            dset_dem = f.create_dataset("dem", [nx, ny])
            dset_dem[()] = dem

            # Mask
            dset_mask = f.create_dataset("mask", [nx, ny])
            dset_mask[()] = np.ones(dem.shape, np.float32)  

            dset_event_name = f.create_dataset("event_name", [N_events],dtype=h5py.string_dtype(encoding='utf-8'))
            dsets_rainfall_events = []
            dsets_timestep = []
            dsets_leakage = []
            dsets = []

            for i in range(N_events):
                # Timesteps
                T = np.random.randint(35, 50)
                dsets_timestep.append(f.create_dataset("timesteps_{}".format(i), [T]))
                dsets_timestep[-1][()] = np.arange(T)
                # Perform simulations
                waterdepth, rainfall, leakage = make_one_simulation(T, dem)
                # Rainfall
                dsets_rainfall_events.append(f.create_dataset("rainfall_events_{}".format(i),  [T]))
                dsets_rainfall_events[-1][()] = rainfall
                # Waterdepth
                dsets.append(f.create_dataset("waterdepth_{}".format(i), [T, nx, ny]))
                dsets[-1][()] = waterdepth
                # Leakage
                dsets_leakage.append(f.create_dataset("leakage_{}".format(i), [T]))
                dsets_leakage[-1][()] = leakage
                
                
                
                

