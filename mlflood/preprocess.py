from tqdm import tqdm
import json
from pathlib import Path
import h5py
import numpy as np
from mlflood.conf import PATH_GENERATED, PATH_TOY, PATH_709, PATH_709_new, PATH_684, PATH_709wall, PATH_709wall_0threshold, PATH_709_0threshold

physical_const = 12 * 1000

def folder_split(catchment_num):
    if (catchment_num == "709"  or catchment_num == "709_new") :
        train_folders = ['tr5_1', 'tr20_1', 'tr50_1', 'tr2_2', 
                    'tr10_2', 'tr20_2', 'tr50_2', 'tr5_3', 
                    'tr10_3', 'tr100_3']

        val_folders = ['tr100_2', 'tr2_3']

        test_folders = ['tr2_1', 'tr10_1', 'tr100_1', 'tr5_2','tr20_3', 'tr50_3']
    elif catchment_num == "toy":
        train_folders = ["tr5_1", "tr20_1", "tr50_1", "tr2_2", "tr10_2", "tr20_2", 
                        "tr50_2", "tr5_3", "tr10_3", "tr100_3"]   
        val_folders = ["tr100_2", "tr2_3"]
        test_folders = []
    elif (catchment_num == "684"  or catchment_num == "709wall" or catchment_num == "709wall_0threshold" or catchment_num == "709_0threshold") :
        train_folders = ["tr100_1"]   
        val_folders = []
        test_folders = []
        
    else:
        raise NotImplementedError()
    return train_folders, val_folders, test_folders


def build_dataset(catchment_num):
    # Location of dataset_npy folder
    if catchment_num == "toy":
        path_dataset = PATH_TOY
    elif catchment_num == "709":
        path_dataset = PATH_709
    elif catchment_num == "709_new":
        path_dataset = PATH_709_new
    elif catchment_num == "709wall_0threshold":
        path_dataset = PATH_709wall_0threshold
    elif catchment_num == "709wall":
        path_dataset = PATH_709wall
    elif catchment_num == "709_0threshold":
        path_dataset = PATH_709_0threshold
    else:
        path_dataset = PATH_684


    # Dem location:
    path_dem = path_dataset/ Path("dem_norm.npy")
    path_mask = path_dataset/ Path("dem_mask.npy")
    path_json = path_dataset/ Path("rainfall.json")

    with open(path_json, "r") as f:
        d = json.load(f)

    dt = d["dt"]
    dict_rains = d["dict_rains"]

    train_folders, val_folders, test_folders = folder_split(catchment_num)

    for folders, suffix in zip([train_folders, val_folders, test_folders], ["train", "val", "test"]):
        name = catchment_num+"-"+suffix
        if len(folders)==0:
            continue
        
        with h5py.File( PATH_GENERATED / Path(name+".h5"), 'w') as f:

            dem = np.load(path_dem)

            nx, ny = dem.shape
            N_events = len(folders)

           ## # should be removed for 709, where some timesteps are different
#             T_steps = []
#             for k,v in dict_rains.items():
#                 T_steps.append(len(v))
#             np.testing.assert_allclose(np.array(T_steps), T_steps[0])
#             T_steps = T_steps[0]
            
            dset_dem = f.create_dataset("dem", [nx, ny])
            dset_dem[()] = dem

            dset_mask = f.create_dataset("mask", [nx, ny])
            dset_mask[()] = np.load(path_mask)
            dset_event_name = f.create_dataset("event_name", [N_events],dtype=h5py.string_dtype(encoding='utf-8'))

            dsets_rainfall_events = []
            dsets_timestep = []
            dsets = []
            for i, folder  in tqdm(enumerate(folders), total=N_events):
                dset_event_name[i] = folder
                p = path_dataset / Path(folder)
                nt = len(list(p.glob("*.npy")))
                dsets_timestep.append(f.create_dataset("timesteps_{}".format(i), [nt]))
                dsets_timestep[-1][()] = np.arange(0, 300*nt, 300)
                dsets_rainfall_events.append(f.create_dataset("rainfall_events_{}".format(i),  [nt]))
                dsets.append(f.create_dataset("waterdepth_{}".format(i), [nt, nx, ny]))
                onefilename = next(p.glob("*.npy")).name
                lenend = len(onefilename.split("_")[-1])
                sim_file = onefilename[:-lenend]
                for j in range(nt):
                    dsets[-1][j] = np.load(p / Path(sim_file + str(j*dt) + ".npy"))
                    if j<len(dict_rains[folder]):
                        dsets_rainfall_events[-1][j] = dict_rains[folder][j] / physical_const
                    else:
                        dsets_rainfall_events[-1][j] = 0