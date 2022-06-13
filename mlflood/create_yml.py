import yaml
import os

folder_name = 'exp_yml'
if not os.path.exists(folder_name): 
    os.mkdir(folder_name) 
    
### Catchment settings
catchment_kwargs = {}
catchment_kwargs["num"] = "709"
catchment_kwargs["tau"] = 0.5
catchment_kwargs["timestep"]= 5      # for timestep >1 use CNN rolling or Unet
catchment_kwargs["sample_type"]="single"
catchment_kwargs["dim_patch"]=256
catchment_kwargs["fix_indexes"]=False
catchment_kwargs["border_size"] = 0
catchment_kwargs["normalize_output"] = False
catchment_kwargs["use_diff_dem"] = False
catchment_kwargs["num_patch"] = 10      # number of patches to generate from a timestep
catchment_kwargs["predict_ahead"] = 12  # how many timesteps ahead to predict; default value 0 for just predicting the next timestep

with open('exp_yml/default_catchment_kwargs.yml', 'w') as file:
    documents = yaml.dump(catchment_kwargs, file)