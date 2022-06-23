import numpy as np
from tqdm import tqdm
import torch
from poutyne import Model, SKLearnMetrics
from dataloading.dataset_max_709 import unnormalize, normalize, pad_borders
from utils import metric_flatten
from models.Baseline import Baseline
# from models.PerfectGraph import PerfectGraph
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, median_absolute_error
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from utils import to_device_eval


### Some model to help the evaluation

def base_model(device=None, border_size=0, normalize_output=False, use_diff_dem=True, ts_ahead=0):
    model = Model(Baseline(border_size=border_size, normalize_output=normalize_output, use_diff_dem=use_diff_dem, ts_ahead=ts_ahead), None, "mse", 
            epoch_metrics=[ SKLearnMetrics(metric_flatten(r2_score)), 
                              SKLearnMetrics(metric_flatten(explained_variance_score)), 
                              SKLearnMetrics(metric_flatten(mean_squared_error)), 
                              SKLearnMetrics(metric_flatten(median_absolute_error))
                            ],
            device=device)
    return model

############################################################################################################################################################
##################################################################  Data Visualization ############################################################################ 


def plot_pie(wd, lims, labels):
    '''
    Pie plot of waterdepth values at a specific timestep
    
    Inputs
    wd: tensor of waterdepth values at a specific timestep
    lims: tuple of values we want to split the data into
    labels: labels for the plot
    '''
    
    if len(labels) == 5:
        sizes = [torch.sum(wd<lims[0]), 
                 torch.sum((wd>lims[0]) & (wd<lims[1])), 
                 torch.sum((wd>lims[1]) & (wd<lims[2])), 
                 torch.sum((wd>lims[2]) & (wd<lims[3])), 
                 torch.sum(wd>lims[3])]
    else:
        sizes = [
                 torch.sum((wd>lims[0]) & (wd<lims[1])), 
                 torch.sum((wd>lims[1]) & (wd<lims[2])), 
                 torch.sum((wd>lims[2]) & (wd<lims[3])), 
                 torch.sum(wd>lims[3])]
    fig1, ax1 = plt.subplots(figsize = [18,6])
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    
    
# Waterdepth evolution
def plot_wd_evol(wd_time, lims, labels):
    '''
    plots evolution over time of waterdepth values
    
    Inputs
    wd_time: tensor of waterdepth values
    lims: tuple of values we want to split the data into
    labels: labels for the plot
    '''
    
    wd_evolution = []
    for i in range(len(wd_time)):
        wd = wd_time [i]
        
        if len(labels) == 5:
            sizes = [torch.sum(wd<lims[0]), 
                     torch.sum((wd>lims[0]) & (wd<lims[1])), 
                     torch.sum((wd>lims[1]) & (wd<lims[2])), 
                     torch.sum((wd>lims[2]) & (wd<lims[3])), 
                     torch.sum(wd>lims[3])]
            sizes = sizes/(np.sum(sizes))
            wd_evolution.append(sizes)
        else:
            sizes = [ 
                     torch.sum((wd>lims[0]) & (wd<lims[1])), 
                     torch.sum((wd>lims[1]) & (wd<lims[2])), 
                     torch.sum((wd>lims[2]) & (wd<lims[3])), 
                     torch.sum(wd>lims[3])]
            sizes = sizes/(np.sum(sizes))
            wd_evolution.append(sizes)   
            
    a_df = pd.DataFrame(data=wd_evolution, columns=labels)
    
    fig, ax = plt.subplots(figsize = [12,6])
    for j in range(len(labels)):
        plt.plot(a_df[labels[j]], label = labels[j])
    plt.legend()
    plt.ylabel('Normalized waterdepth')
    plt.xlabel('Timesteps')
    plt.title('Evolution of Water Depth')
    plt.show()
    
    
############################################################################################################################################################
##################################################################  MAE and MSE  ############################################################################ 


# def evaluate_dataset_1step(model, dataset):
#     mse, _ = model.evaluate_dataset(dataset)
#     return unnormalize(mse)

def predict_batch(model, dataset):
    """
    MSE 1 step for all batches.
    """

    outputs = []
    for x, y in tqdm(dataset):
        y_pred = model.predict([[x]], verbose=0).squeeze()
        mask = y_pred[1] > 0
        y_pred = y_pred[0]
        y = y.detach().cpu().numpy()[0]
        if dataset.normalize_output:
            dim = x[0].shape[0] - dataset.timestep
            xin = x[0][dim].detach().cpu().numpy()
            y_pred = unnormalize(y_pred) + xin
            y = unnormalize(y) + xin
        y = y[mask]
        y_pred = y_pred[mask]
        outputs.append((y_pred, y))
    
    return outputs

def mse(p):
    """
    Compute MSE
    """

    mses = []
    for v1, v2, mask in p:
        mses.append(np.mean((v1[:,mask]-v2[:,mask])**2, axis=1))
    
    return mses

def mses2plot(mses):
    """
    Transforms the compute MSEs into curves for plots
    """

    tmax = max([len(e) for e in mses])
    n_mses = np.zeros([tmax])
    m_mses = np.zeros([tmax])
    s_mses = [[] for _ in range(tmax)]
    for i, e in enumerate(mses):
        t = len(e)
        m_mses[:t] += e
        n_mses[:t] += 1
        for j in range(t):
            s_mses[j].append(e[j])
    m_mses = m_mses/n_mses
    s_mses = np.array([np.std(np.array(e)) for e in  s_mses])
    s_mses = s_mses/np.sqrt(n_mses)
    
    return m_mses, s_mses
    
def mse_from_predicted_dataset(predictions_ag):
    """
    Given the prediction for a dataset, compute the MSE for each event and each timestep.
    """

    mse = []
    for event in predictions_ag:
        mse_event = []
        for y, y_pred, mask in zip(*event):
            mse_event.append(np.mean((y[:,mask]-y_pred[:,mask])**2))
        mse.append(mse_event)
    
    return mse

def mae_event(pred: np.array, gd: np.array, mask: Optional[np.array]=None) -> np.array:
    """MAE for an event.
    
    Return a 1D array of MAE corresponding ot each timestep.
    """

    if mask is not None:
        pred = pred[:, mask]
        gd = gd[:, mask]
    
    return np.mean(abs(pred-gd), axis=1)

def mae_event_upd(pred: np.array, gt: np.array, mask: Optional[np.array]=None, save_folder = None, ev_name = 'ev0', model_nm = "None") -> np.array:
    """MAE for an event.
    
    Return a 1D array of MAE corresponding ot each timestep.
    """
    
    c = pred[:, mask]
    d = gt[:, mask]
    diff = abs(c-d)

    loss_com = []
    for i in range(gt.shape[0]):
        t = np.mean(diff[i,:])
        loss_com.append(t)

    # mean mae calculation for the event
    mean_mae = 0
    for j in range(len(loss_com)):
        mean_mae = mean_mae + loss_com[j]
    mean_mae = mean_mae/len(loss_com)
    print("Mean mae for the event is: ", mean_mae, "m")

    t = np.arange(len(loss_com))
    plt.plot(t, loss_com, label=model_nm)
    plt.legend()
    plt.ylabel('Absolute Error (m)')
    plt.xlabel('Timesteps')
    if save_folder:
        filename = save_folder + 'Mae_' + ev_name +'.png'
        plt.savefig(filename, dpi=1200)
    plt.show()

def mse_event(pred: np.array, gd: np.array, mask: Optional[np.array]=None) -> np.array:
    """MSE for an event.
    
    Return a 1D array of MSE corresponding ot each timestep.
    """

    if mask is not None:
        pred = pred[:, mask]
        gd = gd[:, mask]
    
    return np.mean((pred-gd)**2, axis=1)


def split_values(pred: np.array, gd: np.array, lims: tuple=(0.2, 1)) -> List[Tuple[np.array, np.array]]:
    """Split the value of pred and gd according to lims. 
    
    This function should be applied after the application of the mask.
    
    Return a list of splits: [(pred 1d array, gd 1d array), ...]
    """

    vmin = min(np.min(gd), np.min(pred))
    if vmin < 0:
        print(ValueError(f"Negative waterdept value: {vmin}"))
    vmax = np.max(gd)*2
    if vmax < lims[-1]:
        print("Some bins are empty")
    
    lims = [vmin, *lims, vmax]
    splits = []
    for vmin, vmax in zip(lims[:-1], lims[1:]):
        selection = np.logical_and(gd>=vmin, gd<vmax)
        splits.append((pred[selection], gd[selection]))
    assert sum([len(split[0]) for split in splits]) == gd.size
    
    return splits

def boxplot_mae(pred, gd, mask, lims: tuple=(0.2, 1), pred_ts = None):
    """
    Compute the mae for a boxplot
    pred_ts indicates how many timesteps ahead we are looking at
    """

    if pred_ts:
        pred = pred[pred_ts, mask]
        gd = gd[pred_ts, mask]
    else:
        pred = pred[:, mask]
        gd = gd[:, mask]
    splits = split_values(pred, gd, lims=lims)
    
    return [np.abs(split[0]-split[1]).flatten() for split in splits]


############################################################################################################################################################
##################################################################  Predictions ############################################################################ 

def predict_dataset(model, dataset, start_ts=None, ar = True):
    """
    Predict all events in a dataset.
    """

    predictions_ag = []

    for i in tqdm(range(dataset.N_events)):
        predictions_ag.append(predict_event(model, dataset, i, start_ts=start_ts, ar = ar))
    
    return  predictions_ag


def predict_event(model, dataset, event_num, arch, start_ts=None, ar = True, T = None):
    """
    Predict a full event using overlapping patches. The fuction returns a reconstructed catchment minus a border size of 10 to remove border effect between patches
    
    This function will split the prediction if `dataset.sample_type == "single"`. 
    Alternatively, it will use the full frame to predict if `dataset.sample_type == "full"`.
    
    ar: auto regressively
    start_ts: start at timestep start_ts (if None starts at 0.)
    """
    
    

    def create_inputs(dataset, x_p, y_p, xin, rainfall, mask, dem, diff_dem, xout):
            xin, mask, dem, diff_dem, xout = dataset.crop_to_patch(x_p, y_p, xin, mask, dem, diff_dem, xout)
            inputs = dataset.build_inputs(xin, rainfall, mask, dem, diff_dem)
            return inputs, xout
        
    def crop_overlapping_patches(y_pred, patch_dim):
        '''
        crops a patch to fit it for the overlapping reconstruction
        '''
        cropped = y_pred[10: 
                         int(patch_dim/2) + 10, 
                         10: 
                         int(patch_dim/2) + 10]
        return cropped


    patch_dim = dataset.nx
    inds = dataset.get_all_fix_indexes(non_full=False) 
    b = dataset.border_size
    
    xin = dataset.start_ts.clone()
    rainfall = dataset.rainfall_events[event_num].clone()
    dem = dataset.dem
    diff_dem = dataset.diff_dem
    mask = dataset.dem_mask
    recons_pred_full = torch.zeros(dataset.px - 2*b, dataset.py - 2*b)
    recons_gt_full = torch.zeros(dataset.px - 2*b, dataset.py - 2*b)
    target = dataset.peak[event_num]

    for inds_count, (x_p, y_p) in enumerate(inds):

        inputs_ts, xout = create_inputs(dataset, x_p, y_p, xin, rainfall, mask, dem, diff_dem, target)
        (data, mask1) = inputs_ts
        print(data.shape)
        data = data.unsqueeze(dim=0)
        data = to_device_eval(data)
        y_pred = model(data)['y_pred'].squeeze().detach().cpu()
        y_true = xout.squeeze().detach().cpu()

        plt.imshow(y_pred, cmap = 'Blues')
        plt.show()

#         recons_pred_full[x_p:x_p+patch_dim, y_p:y_p + patch_dim] = y_pred
#         recons_gt_full[x_p:x_p+patch_dim, y_p:y_p + patch_dim] = y_true

        recons_pred_full[x_p + 10:
                     x_p + 10 + (int(patch_dim/2)),
                     y_p + 10:
                     y_p + 10 + (int(patch_dim/2))] = crop_overlapping_patches(y_pred, patch_dim)
        recons_gt_full[x_p + 10:
                    x_p + 10 + (int(patch_dim/2)),
                    y_p + 10:
                    y_p + 10 + (int(patch_dim/2))] = crop_overlapping_patches(y_true, patch_dim)
        
    recons_mask_full = mask.clone()

    return recons_pred_full.numpy(), recons_gt_full.numpy(), recons_mask_full.numpy()



############################################################################################################################################################
##################################################################  Plots ############################################################################ 

#################  Quantitative plots ##################################

def plot_maes(maes, labels, start_ts=0, save_folder = None, name = 'autoregressive', title = None):
    '''
    Plot MAEs for different models. This function is valid for both 1timestep ahead and autoregressive mode
    Inputs: 
        - array of MAEs (each computed with mae_event())
        - array of labels
    '''  

    nt = len(maes[0])
    t = start_ts + np.arange(nt)
    for mae, label in zip(maes, labels):
        plt.plot(t, mae, label=label)   
    plt.legend()
    plt.tight_layout()
    plt.ylabel('Absolute Error (cm)')
    plt.xlabel('Timesteps')
    if title:
        plt.title(title, fontsize = 14 ,fontweight="bold")
    if save_folder:
        filename = save_folder + 'Mae_' + name +'.png'
        plt.savefig(filename, dpi=1200)
    plt.show()
    
def singleboxplot(data, ticks, labels, colors, save_folder = None, name = 'autoregressive', title = "Boxplots for models cnn, graph and baseline "):
    
    '''
    Plot Boxplot for single model. This function is valid for both 1timestep ahead and autoregressive mode
    Inputs: 
        - data: array of data (each computed with boxplot_mae())
        - array of xticks of lenght 5
        - array of labels of len(data)
        - array of colors of lenght 5
    '''  
        
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    assert len(data)==len(labels)
    
    nbox = len(data)
#     nticks = len(data)
    plt.figure(figsize=(15, 10))
    flierprops = dict(marker='d', markerfacecolor='black', markersize=4, linestyle='none', markeredgecolor='black')
    for i, (d, label, c) in enumerate(zip(data, labels, colors)):
        nticks = len(d)
        s = 2/(4*nticks+3)
#         assert nticks == len(d)
        bpl = plt.boxplot(d, positions=np.array(range(nticks))*2.0-2*s+4*s*i, notch=True, patch_artist=True, flierprops=flierprops, sym='', widths=3*s)
        set_box_color(bpl, c) 

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=c, label=label)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    # plt.ylim(0, 8)
    plt.tight_layout()
    if title:
        plt.title(title, fontsize = 14 ,fontweight="bold")
    if save_folder:
        filename = save_folder + 'Boxplot_' + name +'.png'
        plt.savefig(filename, dpi=1200)
    plt.show()
    
def multiboxplot(data, ticks, labels, colors, save_folder = None, name = 'autoregressive', title = "Boxplots for models cnn, graph and baseline "):
    
    '''
    Plot Boxplots for multiple models. This function is valid for both 1timestep ahead and autoregressive mode
    Inputs: 
        - data: array of data (each computed with boxplot_mae())
        - array of xticks of lenght 5
        - array of labels of len(data)
        - array of colors of lenght 5
    '''  
        
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    
    assert len(data)==len(labels)
    
    nbox = len(data)
#     nticks = len(data)
    plt.figure(figsize=(nbox*2,5))
    flierprops = dict(marker='d', markerfacecolor='black', markersize=4, linestyle='none', markeredgecolor='black')
    for i, (d, label, c) in enumerate(zip(data, labels, colors)):
        nticks = len(d)
        s = 2/(4*nticks+3)
#         assert nticks == len(d)
        bpl = plt.boxplot(d, positions=np.array(range(nticks))*2.0-2*s+4*s*i, notch=True, patch_artist=True, flierprops=flierprops, sym='', widths=3*s)
        set_box_color(bpl, c) 

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=c, label=label)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    # plt.ylim(0, 8)
    plt.tight_layout()
    if title:
        plt.title(title, fontsize = 14 ,fontweight="bold")
    if save_folder:
        filename = save_folder + 'Multiboxplots_' + name +'.png'
        plt.savefig(filename, dpi=1200)
    plt.show()
    
#################  Qualitative plots ##################################    

def plot_answer_sample(pred, gt, mask, ts, zoom=None, show_diff=True, global_scale=False, save_folder = None, model_name = '_1ts_ahead_cnn'):
    
    '''
    Visualize a full or zoomed in reconstruction for 1 timestep, for a single model. This function is valid for both 1timestep ahead and autoregressive mode
    Inputs: 
        - predictions
        - aground truth
        - mask
        - zoom: array of coordinates  
    '''  
        
        
    import matplotlib.colors as colors    
    cmin = 0.01
    cmap = "hot_r"
    cmap = "gist_heat_r"
#     cmap='seismic'
    if zoom is not None:
        pred = pred[:, zoom[0]:zoom[1], zoom[2]:zoom[3]]
        gt = gt[:, zoom[0]:zoom[1], zoom[2]:zoom[3]]
        mask = mask[zoom[0]:zoom[1], zoom[2]:zoom[3]]
    
    if show_diff:
        fig, axs = plt.subplots(2,3, figsize=(18,10))
        axs = [axs[0][0], axs[0][1], axs[0][2], axs[1][0], axs[1][1], axs[1][2]]
    else:
        fig, axs = plt.subplots(1,3, figsize=(18,5))
        
    
    diff = np.abs(gt[ts] - pred[ts])

    if global_scale:
        vmax_abs = max(np.max(gt), cmin)
        vmax_diff = max(np.max(np.abs(gt-pred)), cmin)
    else:
        vmax_abs = max(np.max(gt[ts]), cmin)
        vmax_diff = max(np.max(diff), cmin)
        
    pos0 = axs[0].imshow(pred[ts], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_abs))
    pos1 = axs[1].imshow(gt[ts], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_abs))
    pos2 = axs[2].imshow(diff, cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_diff))  
    if show_diff and ts>0:
        pos3 = axs[3].imshow(pred[ts] - gt[ts-1], cmap="seismic", vmin=-vmax_diff, vmax = vmax_diff)  
        pos4 = axs[4].imshow(gt[ts] - gt[ts-1], cmap="seismic", vmin=-vmax_diff, vmax = vmax_diff)  

    fig.suptitle("Visualization for model {} and timestep {} ".format(model_name.split('_')[-1], ts), fontsize = 14 ,fontweight="bold")

    axs[0].set_title('Reconstructed Output',fontweight="bold")
    axs[1].set_title('Reconstructed GT ',fontweight="bold")
    axs[2].set_title('Absolute Error',fontweight="bold")

    cbar0 = fig.colorbar(pos0, ax=axs[0], extend='both', fraction=0.136, pad=0.02)
    cbar1 = fig.colorbar(pos1, ax=axs[1], extend='both', fraction=0.136, pad=0.02)
    cbar2 = fig.colorbar(pos2, ax=axs[2], extend='both', fraction=0.136, pad=0.02)
                         
    if show_diff and ts>0:
        axs[3].set_title('Diff pred',fontweight="bold")
        axs[4].set_title('Diff GT',fontweight="bold")
        cbar3 = fig.colorbar(pos3, ax=axs[3], extend='both', fraction=0.136, pad=0.02)
        cbar4 = fig.colorbar(pos4, ax=axs[4], extend='both', fraction=0.136, pad=0.02)    
        
    if save_folder:
        if zoom:
            model_name = model_name + '_zoom'
        filename = save_folder + 'Reconstruction_' +  model_name + '.png'
        plt.savefig(filename, dpi=1200)
    plt.show()

    
############################################################################################################################################################
##################################################################  Movies ############################################################################ 

# def numpy2movie(predictions, ground_t, fps=10):
    
#     '''
#     Plots all predicted timesteps in validation data
#     '''
#     import matplotlib.pyplot as plt
#     from moviepy.editor import VideoClip
#     from moviepy.video.io.bindings import mplfig_to_npimage
#     duration = len(predictions)/fps
#     fig = plt.figure(figsize=(12,4))
#     vmax= np.max(ground_t)
#     vmin = np.min(ground_t)
#     vmax_d = np.max(np.abs(predictions-ground_t))
#     def make_frame(t):

#         i = int(np.floor(t*fps))
#         ax1 = plt.subplot(131)
#         ax1.clear()
#         plt.imshow(ground_t[i], vmin=vmin, vmax=vmax)
#         plt.colorbar()
#         plt.title("Ground truth")
#         plt.xticks([])
#         plt.yticks([])
        
#         ax2 = plt.subplot(132)
#         ax2.clear()
#         plt.imshow(predictions[i], vmin=vmin, vmax=vmax)
#         plt.colorbar()     

#         plt.title("Prediction")
#         plt.xticks([])
#         plt.yticks([])
        
#         ax3 = plt.subplot(133)
#         ax3.clear()
#         plt.imshow(np.abs(predictions[i]-ground_t[i]), vmin=0, vmax=vmax_d)
#         plt.colorbar()
#         plt.title("Difference")
#         plt.xticks([])
#         plt.yticks([])
        

#         return mplfig_to_npimage(fig)

#     animation = VideoClip(make_frame, duration=duration)
    
#     return animation

def numpy2movie(predictions, ground_t, fps=1, save_folder=None):
    
    '''
    Plots all predicted timesteps in validation data
    '''

    import matplotlib.pyplot as plt
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    import matplotlib.colors as colors
    duration = len(predictions)/fps
    fig = plt.figure(figsize=(12,4))
    vmax= np.max(ground_t)
    # vmin = np.min(ground_t)
    vmax_d = np.max(np.abs(predictions-ground_t))
    #cmap = "inferno_r"
    cmap = "gist_heat_r"
    cmin = 0.01
    def make_frame(t):
    
        i = int(np.floor(t*fps))
        ax1 = plt.subplot(131)
        ax1.clear()
        plt.imshow(ground_t[i], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax))
        plt.colorbar()
        plt.title("Ground truth")
        plt.xticks([])
        plt.yticks([])
        
        ax2 = plt.subplot(132)
        ax2.clear()
        plt.imshow(predictions[i], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax))
        plt.colorbar()     

        plt.title("Prediction")
        plt.xticks([])
        plt.yticks([])
        
        ax3 = plt.subplot(133)
        ax3.clear()
        plt.imshow(np.abs(predictions[i]-ground_t[i]), cmap=cmap, norm=colors.LogNorm(vmin = cmin, vmax = vmax_d))
        plt.colorbar()
        plt.title("Difference")
        plt.xticks([])
        plt.yticks([])
        

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)

    if save_folder:
        filename = save_folder + 'movie.mp4'
        animation.write_videofile(filename, fps=fps)
    
    return animation



def mat2movie(mat, fps=10):
    
    '''
    Plots all predicted timesteps in validation data
    '''

    import matplotlib.pyplot as plt
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    duration = len(mat)/fps
    vmax= np.max(mat)
    vmin = np.min(mat)
    fig = plt.figure()
    ax1 = plt.gca()
    def make_frame(t):
        ax1.clear()
        i = int(np.floor(t*fps))
        ax1.imshow(mat[i], vmin=vmin, vmax=vmax)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    
    return animation

