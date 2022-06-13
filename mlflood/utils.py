import os

def metric_flatten(func):
    def inner(y1, y2, *args, **kwargs):
        mask = y1[:,1] > 0

        y1 = y1[:,0]
        y2 = y2[:,0]
        y1 = y1[mask]
        y2 = y2[mask]
        return func(y1, y2, *args, **kwargs)
    inner.__name__ = "f_"+func.__name__
    return inner

def new_log(catchment_dict, base_path, base_name, args=None):

    folder_path = os.path.join(base_path, args.data)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    folder_path = os.path.join(folder_path, base_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    previous_runs = os.listdir(folder_path)
    n_exp = len(previous_runs)

    experiment_folder = os.path.join(folder_path, "experiment_{}".format(n_exp))

    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        args_dict.update(catchment_dict)
        with open(os.path.join(experiment_folder, "args" + '.txt'), 'w') as f:
            sorted_names = sorted(args_dict.keys(), key=lambda x: x.lower())
            for key in sorted_names:
                value = args_dict[key]
                f.write('%s:%s\n' % (key, value))

    return experiment_folder

def to_device(sample, device=None):
    
    import torch
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sampleout = {}
    sampleout['gt'] = sample[1].to(device=device, dtype=torch.float)
    sampleout['mask'] = sample[0][1].to(device=device, dtype=torch.float)
    sampleout['data'] = sample[0][0].to(device=device, dtype=torch.float)

    return sampleout

def to_device_eval(sample, device= None):
    
    import torch
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sample = sample.to(device=device, dtype=torch.float)

    return sample
