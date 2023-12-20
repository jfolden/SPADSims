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

def gen_full_sig(fovea_data,win_t_start,nt_bins):
    '''
    Generate the full signal from the fovea window, assume default num bins
    '''
    #Create signal zero every where except where the window lies
    full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],nt_bins))
    for i in range(fovea_data.shape[0]):
        for j in range(fovea_data.shape[1]):
            full_sig[i,j,win_t_start[i,j]:win_t_start[i,j]+fovea_data.shape[-1]] = fovea_data[i,j,:]

    return(full_sig)

def gen_full_sig_ds(fovea_data,win_t_start,nt_bins,full_bin_res,ds_bin_res):
    '''
    Generate the full signal from the downsampled fovea window 
    '''
    
    #Create signal zero every where except where the window lies
    full_sig = np.zeros((fovea_data.shape[0],fovea_data.shape[1],nt_bins))
    for i in range(fovea_data.shape[0]):
        for j in range(fovea_data.shape[1]):
            full_sig[i,j,win_t_start[i,j]:win_t_start[i,j]+fovea_data.shape[-1]] = fovea_data[i,j,:]

    return(full_sig)



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
    