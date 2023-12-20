import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
from skimage.transform import resize
from sklearn import preprocessing
from datetime import datetime
breakpoint = debugger.set_trace

#### Local imports
from utils.input_args_parser import add_flash_lidar_scene_args
from toflib.input_args_utils import add_eval_coding_args
from toflib import tof_utils, tirf, tirf_scene, coding, coding_utils
from research_utils import plot_utils, np_utils, io_ops, improc_ops 
from research_utils.timer import Timer 
from datasets import FlashLidarSceneData
from simulate_flash_lidar_scene import get_scene_fname
from PIL import Image
import eval_coding_utils
import helpers
import foveation
import sim



parser = argparse.ArgumentParser(description='Parser for flash lidar simulation.')
add_flash_lidar_scene_args(parser)
add_eval_coding_args(parser)
parser.add_argument('--save_results', default=False, action='store_true', help='Save result images.')
parser.add_argument('--save_data_results', default=False, action='store_true', help='Save results data.')
parser.add_argument('--win_size', default=500, type=int, help='The size of the foveation window')
parser.add_argument('--local_scale', default=False, action='store_true', help='Use Generalized or Local scale - set to true to use local')
parser.add_argument('--mem_fovea', default=False, action='store_true', help='Enable Memory Foveation')
parser.add_argument('--depth_fovea', default=False, action='store_true', help='Enable Depth Foveation')
parser.add_argument('--ds_bins', default=32, type=int, help='Number of bins to downsample to')





args = parser.parse_args()
# if (args.win_size % 2) != 0:
#     args.win_size += 1

num_files_to_sim = 5
all_files = np.genfromtxt('NyuFiles_mod.txt',dtype='str')
fnames = all_files[np.random.randint(0,len(all_files),num_files_to_sim)]
# fnames[0] = "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\dining_room_0019\\00216.h5"
fnames = ["C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0113\\00641.h5",
               "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\office_0004\\00431.h5",
               "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\office_0021\\00021.h5",
               "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\printer_room_0001\\00116.h5",
               "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0078\\00366.h5"]

# fnames_depth = ["F:\\Research\\Collision_Prediction\\SPAD\\1\\depth\\29.png"]
# fnames = ["F:\\Research\\Collision_Prediction\\SPAD\\1\\rgb\\29.png"]


curr_date = datetime.now()
daymonthyear = curr_date.strftime("%m_%d_%Y")


monomodel = helpers.loadModel()

for P in range(num_files_to_sim):
    plt.close('all')

    fnames_split0 = os.path.split(fnames[P])
    fnames_split1 = os.path.split(fnames_split0[0]) 

    out_data_base_dirpath = 'F:/Research/compressive-spad-lidar-cvpr22/data/nyu_results/{}/{}/{}'.format(daymonthyear,fnames_split1[1],fnames_split0[1][:-3])# "win{}".format(args.win_size)
    #NYU Load
    [rgb_gt,depth_gt,h,w] = helpers.load_nyu(fnames[P])
    #Carla Load
    ## rgb_gt = np.asarray(Image.open(fnames[P]))
    ## depth_gt = np.asarray(Image.open(fnames[P])).astype(np.double)
    # depth_gt = 0.2989 * depth_gt[:,:,0] + 0.5870 * depth_gt[:,:,1]+ 0.1140 * depth_gt[:,:,2]

    print("Simming: "+fnames[P])

    [predDepth_gt_raw,predDepth_gt_color] = helpers.monoDepth(monomodel,rgb_gt)

    # abs_errors_ = np.abs(depth_gt- predDepth_gt_raw)*1000
    print('gt_min:{},gt_max:{}, pred_min: {}, pred_max: {}'.format(np.min(depth_gt),np.max(depth_gt),np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    print('gt_mean:{}, pred_mean: {},'.format(np.mean(depth_gt),np.mean(predDepth_gt_raw)))
    # plt.imshow(abs_errors_)
    # plt.show()
    # input('WHEEZEE1')

    #check if pred_depth > max NYU Depth if so norm to min and max of NYU
    # if np.max(predDepth_gt_raw) > 10.0:
    # predDepth_gt_raw = (predDepth_gt_raw-np.min(0))/(np.max(predDepth_gt_raw)-np.min(predDepth_gt_raw))
    # predDepth_gt_raw = np.clip(norm_curve(predDepth_gt_raw),a_min=0,a_max=10) #clip for a linear curve
    ### NORMALIZE
    # predDepth_gt_raw = (predDepth_gt_raw-1e-4)/(15-1e-4)
    # predDepth_gt_raw = norm_curve(predDepth_gt_raw)


    # predDepth_gt_raw = (predDepth_gt_raw - np.min(predDepth_gt_raw))*((np.max(depth_gt)-np.min(depth_gt))/(np.max(predDepth_gt_raw)-np.min(predDepth_gt_raw)))+np.min(depth_gt)
    print('gt_min:{},gt_max:{}, pred_min: {}, pred_max: {}'.format(np.min(depth_gt),np.max(depth_gt),np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    print('gt_mean:{}, pred_mean: {},'.format(np.mean(depth_gt),np.mean(predDepth_gt_raw)))

    if args.local_scale:
        local_curve = helpers.local_scale(depth_gt,predDepth_gt_raw,n_pts=10)
        predDepth_gt_raw = local_curve(predDepth_gt_raw)
    else:
        gen_scale_data = np.load("poly3_gen_scale_all.npz")
        gen_curve = np.poly1d(gen_scale_data['curve'])
        gen_max = gen_scale_data['max']
        gen_min = gen_scale_data['min']
        predDepth_gt_raw = np.clip(predDepth_gt_raw,a_min=0,a_max=10) 
        predDepth_gt_raw = (predDepth_gt_raw-gen_min)/(gen_max-gen_min)
        predDepth_gt_raw = gen_curve(predDepth_gt_raw)


    # predDepth_gt_raw = preprocessing.minmax_scale(predDepth_gt_raw,(0,np.max(depth_gt)))
    # abs_errors_ = np.abs(depth_gt- predDepth_gt_raw)*1000
    # print('gt_mean:{}, pred_meanPostNorm: {}'.format(np.mean(depth_gt),np.mean(predDepth_gt_raw)))
    # plt.imshow(abs_errors_)
    # plt.show()
    # input('WHEEZEE2')

    data_gt = sim.simSPAD(args,rgb=rgb_gt,depth=depth_gt)
    data_gt["mono_raw"] = predDepth_gt_raw
    data_gt["mono_color"] = predDepth_gt_color
#### Getting transients by simming the entire depth is not good, v slow, can change.
    # data_pred = sim.simSPAD(args,rgb=rgb_gt,depth=predDepth_gt_raw) # Add error metric function to always compare agaisnt GT
    
    [_,pred_nzindi] = tof_utils.depthmap2tirf(resize(predDepth_gt_raw,(args.n_rows,args.n_cols)),n_tbins = args.n_tbins, delta_depth = 10 / (args.n_tbins-1)) #Maximum depth in NYU is 10m)
    # questionmark = np.nonzero(data_pred['nonzero_ind'] - pred_nzindi)
    # print(questionmark)

    if args.mem_fovea:
        [fovea_data,win_t_start] = foveation.fovea_window(window_size=args.win_size,histo=data_gt['c_vals'],nz_indi=pred_nzindi)#data_pred['nonzero_ind']
        fovea_full = foveation.gen_full_sig(fovea_data,win_t_start,args.n_tbins)
        decoded_depths_windows = eval_coding_utils.decode_peak(data_gt['coding_obj'], fovea_full, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*data_gt['tbin_depth_res']
        abs_errors_win = np.abs(decoded_depths_windows.squeeze() - resize(depth_gt,(args.n_rows,args.n_cols)))*1000
        win_errors = np_utils.calc_error_metrics(abs_errors_win, delta_eps = data_gt['tbin_res']*1000)
        np_utils.print_error_metrics(win_errors)
        data_fovea = data_gt.copy()
        data_fovea.update(c_vals = fovea_full, decoded_depths = decoded_depths_windows,abs_depth_errors=abs_errors_win,error_metrics=win_errors)

    if args.depth_fovea:
        (rep_tau_ds, rep_freq_ds, tbin_res_ds, t_domain_ds, max_depth_ds, tbin_depth_res_ds) = tof_utils.calc_tof_domain_params(args.ds_bins, max_depth=10)
        ds_sim = foveation.downsample_hist(data_gt['c_vals'],args.ds_bins)
        fovea_box = foveation.get_bb_coords(data_gt['depth_imgs'])
        replaced = foveation.depth_replacement(ds_sim,data_gt['c_vals'],fovea_box)
        depths_ds = eval_coding_utils.decode_peak(data_gt['coding_obj'], ds_sim, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*tbin_depth_res_ds
        depths_rep = eval_coding_utils.decode_peak(data_gt['coding_obj'], replaced, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*(data_gt['tbin_depth_res'])
        if args.mem_fovea:
            n_winbin = int(10/(args.ds_bins/args.win_size))
            (rep_tau_ds_win, rep_freq_ds_win, tbin_res_ds_win, t_domain_ds_win, max_depth_ds_win, tbin_depth_res_ds_win) = tof_utils.calc_tof_domain_params(n_winbin, max_depth=10)
            ds_win = foveation.downsample_hist(fovea_data,args.ds_bins)
            ds_win = foveation.gen_full_sig(ds_win,win_t_start,n_winbin)
            win_rep = foveation.depth_replacement(ds_sim,ds_win,fovea_box)
            depths_ds_win = eval_coding_utils.decode_peak(data_gt['coding_obj'], win_rep, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*tbin_depth_res_ds_win
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.title("Memory Foveation")
        plt.imshow(data_fovea['decoded_depths'])

        plt.subplot(2,2,2)
        plt.title("1/32 Res Histogram")
        plt.imshow(depths_ds)

        plt.subplot(2,2,3)
        plt.title("Full Res Inpainting")
        plt.imshow(depths_rep)

        plt.subplot(2,2,4)
        plt.title("Depth Foveation")
        plt.imshow(depths_ds_win)
        
        plt.show()
        input("YEE")
        


    # sim.plotResults(data_fovea)

    # sim.plotResults(data_gt)
    # sim.plotResults(data_pred)

    # sim.saveResults(data_gt,True,True,out_data_base_dirpath,False)
    # sim.saveResults(data_gt,saveData=True,savePlots=True,out_data_base_dirpath=out_data_base_dirpath,isFoveated=True)
    # sim.saveResults(data_fovea,saveData=True,savePlots=True,out_data_base_dirpath=out_data_base_dirpath,isFoveated=True)






    # [fovea_data,win_t_start] = foveation.fovea_window(window_size=20,histo=c_vals,nz_indi=nz_ind)
    # fovea_full = foveation.gen_full_sig(fovea_data,win_t_start,n_tbins)
    # ds_value = 200
    # (rep_tau_ds, rep_freq_ds, tbin_res_ds, t_domain_ds, max_depth_ds, tbin_depth_res_ds) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_path_length)

    # ds_full = foveation.downsample_hist(c_vals,ds_value)
    # ds_fovea = foveation.fovea_downsample(fovea_data,ds_value)
    # ds_fovea = foveation.gen_full_sig(ds_fovea,win_t_start,n_tbins)

    # decoded_depths_windows = eval_coding_utils.decode_peak(coding_obj, fovea_full, coding_id, rec_algo, pw_factor)*tbin_depth_res

    # abs_depth_errors_window = np.abs(decoded_depths_windows.squeeze() - depths[P])*1000
    # error_metrics = np_utils.calc_error_metrics(abs_depth_errors, delta_eps = tbin_depth_res*1000)
    # error_metrics_window = np_utils.calc_error_metrics(abs_depth_errors, delta_eps = tbin_depth_res*1000)
    # np_utils.print_error_metrics(error_metrics)
    # print('Window Error Metrics: ')
    # np_utils.print_error_metrics(error_metrics)

helpers.close_model(monomodel)