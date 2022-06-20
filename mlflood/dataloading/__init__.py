from dataloading.dataset_multi import load_dataset as load_catch_multi
from dataloading.dataset_multi import dataloader_args as dl_args_multi

from dataloading.dataset import load_dataset as load_catch_709
from dataloading.dataset import dataloader_args as dl_args_709

# from dataloading.dataset_max_709_stef import load_dataset as load_catch_709_max
# from dataloading.dataset_max_709_stef import dataloader_args as dl_args_709_max

from dataloading.dataset_max_709 import load_dataset as load_catch_709_max
from dataloading.dataset_max_709 import dataloader_args as dl_args_709_max

from dataloading.dataset_max_multi import load_dataset as load_multi_max
from dataloading.dataset_max_multi import dataloader_args as dl_args_multi_max

def get_dataloaders(args, catchment_kwargs):
    
    if args.task == "wd_ts" and args.data == "multi":
        train, val =  load_catch_multi(catchment_kwargs)
        return dl_args_multi(train, val, catchment_num=catchment_kwargs["num"], batch_size=args.batch_size)
    
    if args.task == "wd_ts" and args.data == "709":
        train, val = load_catch_709(catchment_kwargs)
        return dl_args_709(train, val, catchment_num=catchment_kwargs["num"], batch_size=args.batch_size)
    
    if args.task == "max_depth" and args.data == "multi":
        train, val = load_multi_max(catchment_kwargs)
        return dl_args_multi_max(train, val, catchment_num=catchment_kwargs["num"], batch_size=args.batch_size)
    
    if args.task == "max_depth" and args.data == "709":
        train, val = load_catch_709_max(catchment_kwargs)
        return dl_args_709_max(train, val, catchment_num=catchment_kwargs["num"], batch_size=args.batch_size)
    else:
        raise NotImplementedError
