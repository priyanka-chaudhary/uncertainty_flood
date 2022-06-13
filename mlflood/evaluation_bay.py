import numpy as np
from tqdm import tqdm
import torch
from poutyne import Model, SKLearnMetrics
from dataset import unnormalize, normalize, pad_borders
from utils import metric_flatten
from models.Baseline import Baseline
# from models.PerfectGraph import PerfectGraph
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, median_absolute_error
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from utils import to_device_eval

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    
    yhat = x[time_step][mask_cnn]
    test_targets = y[time_step][mask_cnn]
    
    if ax is None :
        fig , ax = plt.subplots(figsize=(8,5))
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    
    plt.xlabel("Predicted water depth (m)",fontsize=18)
    plt.ylabel("Real water depth (m)", fontsize=18)
    plt.title("Scatter plot errors", fontsize=20)
#     plt.plot([0, 6], [0, 6], 'black', label= 'diagonal')

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def predict_event_bay(model, dataset, event_num, arch, start_ts=None, ar = True, T = None, ts_out = 0):
    """
    Predict a full event.
    
    This function will split the prediction if `dataset.sample_type == "single"`. 
    Alternatively, it will use the full frame to predict if `dataset.sample_type == "full"`.
    
    ar: auto regressively
    start_ts: start at timestep start_ts (if None starts at 0.)
    """

    patch_dim = dataset.nx
    # inds = dataset.get_all_fix_indexes() 
    b = dataset.border_size
    timestep = dataset.timestep
    tsh = dataset.predict_ahead

    if start_ts is None:
        start_ts = timestep-1
    assert start_ts >= timestep-1
    # assert dataset.normalize_output == False
    if T is None:
        T = len(dataset.rainfall_events[event_num]) - start_ts - 1 - tsh
    
    index_t = start_ts - timestep + 1
    xin = dataset.waterdepth[event_num][index_t : index_t +timestep].clone()
    rainfall = dataset.rainfall_events[event_num][index_t : ].clone()
    dem = dataset.dem
    diff_dem = dataset.diff_dem
    mask = dataset.dem_mask
    recons_pred_full = torch.zeros(T, dataset.px - 2*b, dataset.py - 2*b)    
    recons_pred_full_sigma = torch.zeros(T, dataset.px - 2*b, dataset.py - 2*b) 

    for t in tqdm(range(T)):
        recons_pred_full[t], recons_pred_full_sigma[t] = predict_next_ts_bay(dataset, model, xin, rainfall[t:t+timestep+tsh], mask, dem, diff_dem, ts_out)
        recons_pred_full[t] = recons_pred_full[t] * mask
        recons_pred_full_sigma[t] = recons_pred_full_sigma[t] * mask
        
        if ar:
            if timestep>1:
                xin[:-1] = xin[1:].clone()
            xin[-1] = dataset.pad_borders(recons_pred_full[t].clone(), 0)
        else:
            xin = dataset.waterdepth[event_num][index_t+t+1: index_t+t+timestep+1].clone()
  
    if b:
        recons_gt_full = dataset.waterdepth[event_num][start_ts+tsh +1:start_ts+tsh+T +1, b : -b, b: -b].clone()
        recons_mask_full = mask[b : -b, b : -b ].clone()
    else:
        recons_gt_full = dataset.waterdepth[event_num][start_ts+tsh +1:start_ts+tsh+T +1, : , :].clone()
        recons_mask_full = mask.clone()

    return recons_pred_full.numpy(), recons_gt_full.numpy(), recons_mask_full.numpy(), recons_pred_full_sigma.numpy()


def predict_next_ts_bay(dataset, model, xin, rainfall, mask, dem, diff_dem, ts_out):

    """Predict the next timestep for unet arch. C x H x W
    C = channels, H = height, W = width
    """

    b = dataset.border_size
    normalize_output = dataset.normalize_output

    def create_inputs(dataset, x_p, y_p, xin, rainfall, mask, dem, diff_dem):
        xin, mask, dem, diff_dem, _ = dataset.crop_to_patch(x_p, y_p, xin, mask, dem, diff_dem)
        inputs = dataset.build_inputs(xin, rainfall, mask, dem, diff_dem)
        return inputs

    nx = dataset.nx
    ny = dataset.ny

    inds = dataset.get_all_fix_indexes(non_full=True) 
    recons_pred = torch.zeros(dataset.px - 2*b, dataset.py - 2*b)   
    recons_pred_sigma = torch.zeros(dataset.px - 2*b, dataset.py - 2*b)   
    for inds_count, (x_p, y_p) in enumerate(inds):

        inputs_ts = create_inputs(dataset, x_p, y_p, xin, rainfall, mask, dem, diff_dem)

        # stef
        (data, mask1) = inputs_ts
        data = data.unsqueeze(dim=0)
        data = to_device_eval(data)
        y_pred = model(data)

        sigma = y_pred['sigma'].detach().cpu()
        y_pred = y_pred['y_pred'].squeeze().detach().cpu()

        pred_cnn_ts = y_pred * mask1.squeeze().numpy()
        sigma = sigma * mask1.squeeze().numpy()
    
        
        x_p2 = x_p + (nx-b*2) 
        y_p2 = y_p + (ny-b*2)
        if normalize_output:
            px, py = y_pred.shape
            y_pred = xin[-1, x_p+b: x_p+b+ px, y_p+b:y_p + b + py] + unnormalize(pred_cnn_ts)
        
        if ts_out:  
            pred_cnn_ts = pred_cnn_ts.squeeze()[0]
            sigma = sigma.squeeze()[0]

            
        recons_pred[x_p:x_p2, y_p:y_p2] = pred_cnn_ts
        recons_pred_sigma[x_p:x_p2, y_p:y_p2] = sigma
        
        
    
    return recons_pred, recons_pred_sigma



def plot_answer_sample_bay(pred, pred_sigma, gt, mask, ts, zoom=None, show_diff=True, global_scale=False, save_folder = None, model_name = '_1ts_ahead_cnn'):
    
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
        pred_sigma = pred_sigma[:,zoom[0]:zoom[1], zoom[2]:zoom[3]]
        gt = gt[:, zoom[0]:zoom[1], zoom[2]:zoom[3]]
        mask = mask[zoom[0]:zoom[1], zoom[2]:zoom[3]]
    
    if show_diff:
        fig, axs = plt.subplots(2,3, figsize=(18,10))
        axs = [axs[0][0], axs[0][1], axs[0][2], axs[1][0], axs[1][1], axs[1][2]]
    else:
        fig, axs = plt.subplots(1,4, figsize=(18,5))
        
    
    diff = np.abs(gt[ts] - pred[ts])

    if global_scale:
        vmax_abs = max(np.max(gt), cmin)
        vmax_diff = max(np.max(np.abs(gt-pred)), cmin)
    else:
        vmax_abs = max(np.max(gt[ts]), cmin)
        vmax_diff = max(np.max(diff), cmin)
        
    pos0 = axs[0].imshow(pred[ts], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_abs))
    pos1 = axs[1].imshow(gt[ts], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_abs))
    pos2 = axs[2].imshow(pred_sigma[ts], cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_diff))  
    pos3 = axs[3].imshow(diff, cmap=cmap, norm=colors.LogNorm(vmin =cmin, vmax = vmax_diff)) 
    if show_diff and ts>0:
        pos3 = axs[3].imshow(pred[ts] - gt[ts-1], cmap="seismic", vmin=-vmax_diff, vmax = vmax_diff)  
        pos4 = axs[4].imshow(gt[ts] - gt[ts-1], cmap="seismic", vmin=-vmax_diff, vmax = vmax_diff)  

    fig.suptitle("Visualization for model {} and timestep {} ".format(model_name.split('_')[-1], ts), fontsize = 14 ,fontweight="bold")

    axs[0].set_title('Reconstructed Output',fontweight="bold")
    axs[1].set_title('Reconstructed GT ',fontweight="bold")
    axs[2].set_title('Sigma',fontweight="bold")
    axs[3].set_title('Absolute Error',fontweight="bold")

    cbar0 = fig.colorbar(pos0, ax=axs[0], extend='both', fraction=0.136, pad=0.02)
    cbar1 = fig.colorbar(pos1, ax=axs[1], extend='both', fraction=0.136, pad=0.02)
    cbar2 = fig.colorbar(pos2, ax=axs[2], extend='both', fraction=0.136, pad=0.02)
    cbar3 = fig.colorbar(pos3, ax=axs[2], extend='both', fraction=0.136, pad=0.02)
                         
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
    
    
    
    
    # Combine errors

def boxplot_single(time_step, pred_cnn, gt_cnn, mask_cnn):
    yhat = pred_cnn[time_step][mask_cnn]
    test_targets = gt_cnn[time_step][mask_cnn]
    
    a = [yhat, test_targets]
    a = np.array(a).T
    a_df = pd.DataFrame(data=a, columns=['yhat', 'test_targets'])
    a_df_sorted = a_df.sort_values(by=['test_targets'])
    a_df_sorted.reset_index(drop=True, inplace=True)
    a_df_error = 100 * np.abs(a_df_sorted.yhat - a_df_sorted.test_targets)    # predicting in meter, so need to be multiplied by 100

    data = np.array([a_df_error[a_df_sorted.test_targets<0.1], 
            a_df_error[(a_df_sorted.test_targets>0.10) & (a_df_sorted.test_targets<0.20)], 
            a_df_error[(a_df_sorted.test_targets>0.20) & (a_df_sorted.test_targets<0.50)], 
            a_df_error[(a_df_sorted.test_targets>0.50) & (a_df_sorted.test_targets<1)],
            a_df_error[a_df_sorted.test_targets>1]], dtype="object")
    fig, ax = plt.subplots(figsize = [8,5])
    ax.set_ylabel('Absolute Error (cm)')   
    ax.set_xlabel('Water Depth') 
    ax.boxplot(data, showfliers=False)                                #'Hide Outlier Points'
    ax.set_xticklabels(['0-10 cm', '10-20 cm', '20-50cm', '50-100cm','>100cm'])

    plt.show()