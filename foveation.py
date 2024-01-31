#### Standard Library Imports
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
from IPython.core import debugger
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
breakpoint = debugger.set_trace

#### Local imports
from utils.input_args_parser import add_flash_lidar_scene_args
from toflib.input_args_utils import add_eval_coding_args
from toflib import tof_utils, tirf, tirf_scene, coding, coding_utils
from research_utils import plot_utils, np_utils, io_ops, improc_ops
import helpers


def fovea_window(window_size,histo,nz_indi):
    '''
    Make a window around each transient
    '''
    win = window_size // 2
    histo = np.squeeze(histo)
    n_tbins = histo.shape[-1]
    fovea_data = np.zeros((histo.shape[0],histo.shape[1],window_size))
    win_t_start = np.zeros((histo.shape[0],histo.shape[1]))
    for i in range(histo.shape[0]):
        for j in range(histo.shape[1]):
            ind = nz_indi[i,j]
            if ind-win <= 0:
                fovea_data[i,j,:] = histo[i,j,0:window_size]
                win_t_start[i,j] = 0
            elif ind+win >= n_tbins:
                fovea_data[i,j,:] = histo[i,j,n_tbins-window_size:n_tbins]
                win_t_start[i,j] = n_tbins-window_size
            else:
                fovea_data[i,j,:] = histo[i,j,ind-win:ind+win]
                win_t_start[i,j] = ind-win

    return(fovea_data,win_t_start.astype(np.int))


def fovea_window_threshold(window_size,histo,nz_indi,**kwargs):
    '''
    Make a window around each transient
    '''
    win = window_size // 2
    histo = np.squeeze(histo)
    n_tbins = histo.shape[-1]
    fovea_data = np.zeros((histo.shape[0],histo.shape[1],window_size))
    win_t_start = np.zeros((histo.shape[0],histo.shape[1]))
    for i in range(histo.shape[0]):
        for j in range(histo.shape[1]):
            ind = nz_indi[i,j]
            if ind-win <= 0:
                fovea_data[i,j,:] = histo[i,j,0:window_size]
                win_t_start[i,j] = 0
            elif ind+win >= n_tbins:
                fovea_data[i,j,:] = histo[i,j,n_tbins-window_size:n_tbins]
                win_t_start[i,j] = n_tbins-window_size
            else:
                fovea_data[i,j,:] = histo[i,j,ind-win:ind+win]
                win_t_start[i,j] = ind-win



    #Create signal zero every where except where the window lies

    try:
        if 'threshold' in kwargs:
            num_pixels_cheated = 0
            index = []
            win_t_start = win_t_start.astype(int)
            full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],kwargs['n_tbins']))
            for i in range(fovea_data.shape[0]):
                for j in range(fovea_data.shape[1]):
                    if np.max(fovea_data[i,j,:]) < kwargs['threshold']:
                        index.append((i,j))
                        num_pixels_cheated += 1
                        full_sig[i,j,:] = histo[i,j,:]
                        win_t_start[i,j] = 0
                    else:
                        full_sig[i,j,win_t_start[i,j]:win_t_start[i,j]+fovea_data.shape[-1]] = fovea_data[i,j,:]
            # full_sig[full_sig == 0] = np.random.randint(low=0, high=3, size=np.sum(full_sig == 0))
    except ValueError as e:
        print(fovea_data.shape[-1])


    return(fovea_data,win_t_start,full_sig,num_pixels_cheated,index)

def fovea_window_flow(window_size,histo,nz_indi,push_pull):
    '''
    Make a window around each transient
    '''
    win = window_size // 2
    histo = np.squeeze(histo)
    n_tbins = histo.shape[-1]
    fovea_data = np.zeros((histo.shape[0],histo.shape[1],window_size))
    win_t_start = np.zeros((histo.shape[0],histo.shape[1]))
    for i in range(histo.shape[0]):
        for j in range(histo.shape[1]):
            ind = nz_indi[i,j] + push_pull[i,j] #50*((push_pull[i,j]//(push_pull[i,j]+1e-6)).astype(int))
            if ind-win <= 0:
                fovea_data[i,j,:] = histo[i,j,0:window_size]
                win_t_start[i,j] = 0
            elif ind+win >= n_tbins:
                fovea_data[i,j,:] = histo[i,j,n_tbins-window_size:n_tbins]
                win_t_start[i,j] = n_tbins-window_size
            else:
                fovea_data[i,j,:] = histo[i,j,ind-win:ind+win]
                win_t_start[i,j] = ind-win 

    return(fovea_data,win_t_start.astype(np.int))

def gen_full_sig(fovea_data,win_t_start,nt_bins):
    '''
    Generate the full signal from the fovea window, assume default num bins
    '''
    #Create signal zero every where except where the window lies
    full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],nt_bins))
    try:
        for i in range(fovea_data.shape[0]):
            for j in range(fovea_data.shape[1]):
                full_sig[i,j,win_t_start[i,j]:win_t_start[i,j]+fovea_data.shape[-1]] = fovea_data[i,j,:]
        # full_sig[full_sig == 0] = np.random.randint(low=0, high=3, size=np.sum(full_sig == 0))
    except ValueError as e:
        print(fovea_data.shape[-1])
    return(full_sig)

def gen_full_sig_ds(fovea_data,win_t_start,nt_bins,full_bin_res,ds_bin_res):
    '''
    Generate the full signal from the downsampled fovea window 
    '''
    start_depth = win_t_start * full_bin_res
    win_bin_depths = np.arange(0,nt_bins,1) * ds_bin_res
    
    #Create signal zero every where except where the window lies
    full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],nt_bins))
    for i in range(fovea_data.shape[0]):
        for j in range(fovea_data.shape[1]):
            idx = (np.abs(win_bin_depths-start_depth[i,j])).argmin()
            full_sig[i,j,idx:idx+fovea_data.shape[-1]] = fovea_data[i,j,:]

    return(full_sig)

def quantize(dmap,num_bins):
    hist, bins = np.histogram(dmap.flatten(), bins=num_bins)
    quant_dmap_bins = np.digitize(dmap, bins,right=True)
    quant_dmap_depths = bins[quant_dmap_bins]
    return(quant_dmap_bins,quant_dmap_depths,bins)



def downsample_hist(histo,desired_nbins):
    '''
    Take in c_vals and return it with downsampled bins
    '''
    c_vals = np.squeeze(histo)
    start_res = c_vals.shape[-1]
    new_bins = np.linspace(0,start_res,desired_nbins,dtype=int)
    low_res_hist = np.zeros((c_vals.shape[0],c_vals.shape[1],desired_nbins))
    # for i in range(c_vals.shape[0]):
    #     for j in range(c_vals.shape[1]):
    #         for k in range(desired_nbins-1):
    #             low_res_hist[i,j,k] = np.sum(c_vals[i,j,new_bins[k]:new_bins[k+1]])
    for k in range(desired_nbins-1):
        low_res_hist[:,:,k] = np.sum(c_vals[:,:,new_bins[k]:new_bins[k+1]],2)

    return(low_res_hist)

# def fovea_downsample(histo,desired_nbins):
#     start_res = histo.shape[-1]
#     new_bins = np.linspace(0,start_res,desired_nbins,dtype=int)
#     low_res_hist = np.zeros((histo.shape[0],histo.shape[1],desired_nbins))
#     # for i in range(histo.shape[0]):
#     #     for j in range(histo.shape[1]):
#     #         for k in range(desired_nbins-1):
#     #             low_res_hist[i,j,k] = np.sum(histo[i,j,new_bins[k]:new_bins[k+1]])
#     for k in range(desired_nbins-1):
#         low_res_hist[:,:,k] = np.sum(histo[:,:,new_bins[k]:new_bins[k+1]],2)

#     return(low_res_hist)

class BoundingBoxSelector:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.coords = None
        self.rect = None  # To store the Rectangle patch

        # Create a RectangleSelector
        self.rs = RectangleSelector(
            self.ax, self.on_select, drawtype='box', useblit=True,
            button=[1],  # Left mouse button to draw a box
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

        # Connect the 'key_press_event' to the on_key_press function
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show(block=True)  # Use block=True to keep the plot open

    def on_select(self, eclick, erelease):
        if self.rect:
            # Clear the previous bounding box
            self.rect.remove()

        x_min, y_min = int(eclick.xdata), int(eclick.ydata)
        x_max, y_max = int(erelease.xdata), int(erelease.ydata)

        # Store all coordinate locations inside the bounding box in an array
        self.coords = [(int(x), int(y)) for x in range(x_min, x_max) for y in range(y_min, y_max)]

        # Draw the new bounding box on the image
        self.rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)

        plt.draw()

    def on_key_press(self, event):
        if event.key == 'enter':
            plt.close()

def get_bb_coords(image):
    bbox_selector = BoundingBoxSelector(image)
    # Return the bounding box coordinates
    return bbox_selector.coords


def depth_replacement(low_histo,high_histo,coords):
    low_histo = low_histo.squeeze()
    high_histo = high_histo.squeeze()
    factor = high_histo.shape[-1]//low_histo.shape[-1]
    out = np.repeat(low_histo,factor,axis=2)//factor
    if out.shape[-1] != high_histo.shape[-1]:
        n = high_histo.shape[-1]-out.shape[-1]
        for i in range(n): out = np.dstack((out,out[:,:,-1]))
    for coord in coords:
        y,x = coord
        out[x,y,:] = high_histo[x,y,:]
    return out


def rand_pixel_per_bin(hist_bins, quantized_image,num_pix_per_bin,smartChoice):
    # Ensure hist_bins and quantized_image have the same number of bins
    assert len(hist_bins) == (np.max(quantized_image) + 1), "Number of bins in hist_bins must match the quantized_image"

    # Get unique quantization bin values
    unique_bins = np.unique(quantized_image)

    # Initialize lists to store randomly chosen x and y coordinates per bin
    coords = np.zeros((num_pix_per_bin,2,len(unique_bins)),dtype=int)

    # Loop through each quantization bin
    for i,bin_value in enumerate(unique_bins):
        # Create a mask for the current bin
        bin_mask = (quantized_image == bin_value)

        # Get indices of non-zero elements in the bin mask
        non_zero_indices = np.argwhere(bin_mask)

        # Randomly choose one pixel location for the current bin
        if len(non_zero_indices) > 0:
            if smartChoice:
                random_index = np.linspace(np.min(non_zero_indices),np.max(non_zero_indices),num=num_pix_per_bin)
            else:
                random_index = np.random.choice(len(non_zero_indices),size=num_pix_per_bin,replace=True)
            random_pixel_location = (non_zero_indices[random_index])
            coords[:,:,i] = random_pixel_location


    return coords

def spatial_windows(fovea_data,win_t_start,coords,n_tbins,quant_bins):


    '''
    Generate the full signal from the fovea window
    '''
    #Create signal zero every where except where the window lies
    full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],n_tbins))
    try:
        for i in range(quant_bins.max()+1):
                bin_windows = fovea_data[coords[:,0,i],coords[:,1,i],:]
                bin_starts = win_t_start[coords[:,0,i],coords[:,1,i]]
                for j in range(len(bin_starts)):
                    full_sig[coords[j,0,i],coords[j,1,i],bin_starts[j]:bin_starts[j]+fovea_data.shape[2]] = bin_windows[j]
                
                # full_sig[inds[0],inds[1],bin_starts[i]:bin_starts[i]+fovea_data.shape[-1]] = bin_windows[i]
        # full_sig[full_sig == 0] = np.random.randint(low=0, high=3, size=np.sum(full_sig == 0))
    
    except ValueError as e:
        print(e)
    return(full_sig)

def spatial_average(depth_img,coords,quant_bins,checkmin):
    prev_d = 0
    filled_depth = depth_img.copy()
    for i in range(quant_bins.max()+1):
        bin_inds = np.where(quant_bins==i)

        d_vals = np.sort(depth_img[coords[:,0,i],coords[:,1,i]])
        # print(d_vals)
        
        if i == 0:
            avg_dval = np.min((depth_img[depth_img>0]))
        else:
            # avg_dval = d_vals[5]
            avg_dval = np.min(d_vals)
        # if avg_dval.any() < prev_d:
        #     loop = True
        #     k = 0
        #     while loop:
        #         avg_dval = depth_img[k]
        #         k+=1
        #         if avg_dval.any() > prev_d:
        #             loop = False
        k = 0
        if checkmin:
            while avg_dval < prev_d:
                avg_dval = d_vals[k]
                k+=1
        
      
        filled_depth[bin_inds[0],bin_inds[1]] = avg_dval
        prev_d = avg_dval
        print(prev_d)
    return(filled_depth)
        
