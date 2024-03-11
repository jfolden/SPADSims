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
from tqdm import tqdm

#### Local imports
from utils.input_args_parser import add_flash_lidar_scene_args, add_transient_sim_args
from toflib.input_args_utils import add_eval_coding_args
from toflib import tof_utils, tirf, tirf_scene, coding, coding_utils
from research_utils import plot_utils, np_utils, io_ops, improc_ops 
from research_utils.timer import Timer 
from datasets import FlashLidarSceneData
from simulate_flash_lidar_scene import get_scene_fname
from PIL import Image
import eval_coding_utils
import helpers
from helpers import RunningAverageDict
import foveation
import sim
import scipy
import scipy.io



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
parser.add_argument('--eval', default=False, action='store_true', help='Calculate Error Metrics')
parser.add_argument('--running_avg', default=False, action='store_true', help='Calculate metrics with a running average, or per image')
parser.add_argument('--n_imgs', default=1, type=int, help='Total number of images to sim')
parser.add_argument('--save_percent', default=100, type=int, help='percent of total images to save expressed as an integer')
parser.add_argument('--dataset', default='lindel', type=str, help='what dataset to run')






args = parser.parse_args()
# if (args.win_size % 2) != 0:
#     args.win_size += 1

num_files_to_sim = args.n_imgs
num_images_to_save = int(np.ceil(num_files_to_sim * args.save_percent / 100))

all_files = np.genfromtxt('NyuFiles_val.txt',dtype='str')
fnames = all_files[np.random.randint(0,len(all_files),num_files_to_sim)]
# fnames[0] = "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\dining_room_0019\\00216.h5"
# fnames = ["C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0113\\00641.h5",
#                "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\office_0004\\00431.h5",
#                "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\office_0021\\00021.h5",
#                "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\printer_room_0001\\00116.h5",
#                "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0078\\00366.h5"]

# fnames_depth = ["F:\\Research\\Collision_Prediction\\SPAD\\1\\depth\\29.png"]
# fnames = ["F:\\Research\\Collision_Prediction\\SPAD\\1\\rgb\\29.png"]
# fnames = ["C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\val\\official\\00636.h5"]
# val/00355, val/00034, val/00315, val/00431, val/01170
fnames = ["D:/datasets/lindell_2018/data/captured/elephant.mat",
          "D:/datasets/lindell_2018/data/captured/checkerboard.mat",
          "D:/datasets/lindell_2018/data/captured/hallway.mat",
          "D:/datasets/lindell_2018/data/captured/kitchen.mat",
          "D:/datasets/lindell_2018/data/captured/lamp.mat",
          "D:/datasets/lindell_2018/data/captured/roll.mat",
          "D:/datasets/lindell_2018/data/captured/stairs_ball.mat",
          "D:/datasets/lindell_2018/data/captured/stairs_walking.mat",
          "D:/datasets/lindell_2018/data/captured/stuff.mat"]


curr_date = datetime.now()
daymonthyear = curr_date.strftime("%m_%d_%Y")


monomodel = helpers.loadModel()
if args.running_avg:
    gt_sim_metrics_run = helpers.RunningAverageDict()
    mem_fovea_metrics_run = helpers.RunningAverageDict()
    depth_fovea_metrics_run = helpers.RunningAverageDict()
    ds_sim_metrics_run = helpers.RunningAverageDict()
    ds_v_dfovea_metrics_run = helpers.RunningAverageDict()


for P in tqdm(range(num_files_to_sim),desc="Total Sim"):
    plt.close('all')
    gt_sim_metrics = helpers.RunningAverageDict()
    mem_fovea_metrics = helpers.RunningAverageDict()
    depth_fovea_metrics = helpers.RunningAverageDict()
    ds_sim_metrics = helpers.RunningAverageDict()
    ds_v_dfovea_metrics = helpers.RunningAverageDict()

    fnames_split0 = os.path.split(fnames[P])
    # fnames_split1 = os.path.split(fnames_split0[0])
    nyu_val_number = fnames_split0[1][:-4] #Name of data 
    if args.dataset == 'lindel':
        #load Lindel mats
        [rgb,histo,numbins,h,w] = helpers.load_mat(fnames[P])
        args.n_tbins = numbins
        args.max_depth = scipy.constants.c*((26*10**-12)*numbins)/2
        print(args.n_tbins)
        print(args.max_depth)

    else:
        break
    
    out_data_base_dirpath = 'D:/real_results/lindel/{}/{checks}'.format(daymonthyear,
                                                                                                            checks="win{}_dsbins{}/".format(args.win_size,args.ds_bins) 
                                                                                                            if (args.mem_fovea and args.depth_fovea) 
                                                                                                            else "")# "win{}".format(args.win_size)
    out_data_dirpath = os.path.join(out_data_base_dirpath,'{checks}_{}/{}'.format(curr_date.strftime("%H%M"),nyu_val_number,checks="local_scale" if args.local_scale else "noScale"))
    

    print("Simming: "+fnames[P])
    rgb,histo = rgb[0],histo[0]
    [predDepth_gt_raw,predDepth_gt_color] = helpers.monoDepth(monomodel,rgb)
    predDepth_gt_raw = resize(predDepth_gt_raw,(args.n_rows,args.n_cols))
    lindel_outs = scipy.io.loadmat("D:/datasets/lindell_2018/code/results_captured/elephant_FusionDenoise_10_0.mat")
    lindel_gt = lindel_outs['out']
    # abs_errors_ = np.abs(depth_gt- predDepth_gt_raw)*1000
    # print('gt_min:{},gt_max:{}, pred_min: {}, pred_max: {}'.format(np.min(depth_gt),np.max(depth_gt),np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    # print('gt_mean:{}, pred_mean: {},'.format(np.mean(depth_gt),np.mean(predDepth_gt_raw)))
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

    print('pred_min: {}, pred_max: {}'.format(np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    if args.local_scale:
        curve = helpers.local_scale(lindel_gt,predDepth_gt_raw,n_pts=25)
        predDepth_gt_raw = curve(predDepth_gt_raw)
    # else:
    #     gen_scale_data = np.load("poly3_gen_scale_all.npz")
    #     curve = np.poly1d(gen_scale_data['curve'])
    #     gen_max = gen_scale_data['max']
    #     gen_min = gen_scale_data['min']
    #     predDepth_gt_raw = np.clip(predDepth_gt_raw,a_min=0,a_max=10)
    #     predDepth_gt_raw = (predDepth_gt_raw-gen_min)/(gen_max-gen_min)
    #     predDepth_gt_raw = curve(predDepth_gt_raw)
    print('gt_min:{}, gt_max:{}, \n pred_min: {}, pred_max: {}'.format(np.min(lindel_gt),np.max(lindel_gt),np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    # print('pred_min: {}, pred_max: {}'.format(np.min(predDepth_gt_raw),np.max(predDepth_gt_raw)))
    print('gt_mean:{}, pred_mean: {},'.format(np.mean(lindel_gt),np.mean(predDepth_gt_raw)))
    
    # #Bad Scaling no cookie
    # if (np.max(predDepth_gt_raw) > 10) or (np.min(predDepth_gt_raw)<0):
    #     continue

    # predDepth_gt_raw = preprocessing.minmax_scale(predDepth_gt_raw,(0,np.max(depth_gt)))
    # abs_errors_ = np.abs(depth_gt- predDepth_gt_raw)*1000
    # print('gt_mean:{}, pred_meanPostNorm: {}'.format(np.mean(depth_gt),np.mean(predDepth_gt_raw)))
    # plt.imshow(abs_errors_)
    # plt.show()
    # input('WHEEZEE2')

    data_gt = sim.simSPAD(args,rgb=rgb,depth=lindel_gt,histo=histo)
    
    #Update data_gt dict with experiment information
    data_gt.update(mono_raw = predDepth_gt_raw, mono_color = predDepth_gt_color)

    
    # [_,pred_nzindi] = tof_utils.depthmap2tirf(predDepth_gt_raw,n_tbins = args.n_tbins, delta_depth = args.max_depth / (args.n_tbins-1)) #Maximum depth in NYU is 10m)

    [_,pred_nzindi] = tof_utils.depthmap2tirf(lindel_gt,n_tbins = args.n_tbins, delta_depth = args.max_depth / (args.n_tbins-1)) #Maximum depth in NYU is 10m)


    if args.mem_fovea:
        [fovea_data,win_t_start] = foveation.fovea_window(window_size=args.win_size,histo=data_gt['c_vals'],nz_indi=pred_nzindi)#data_pred['nonzero_ind']
        fovea_full = foveation.gen_full_sig(fovea_data,win_t_start,args.n_tbins)
        decoded_depths_windows = eval_coding_utils.decode_peak(data_gt['coding_obj'], fovea_full, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*data_gt['tbin_depth_res']
        # abs_errors_win = np.abs(decoded_depths_windows.squeeze() - resize(depth_gt,(args.n_rows,args.n_cols)))*1000
        # win_errors = np_utils.calc_error_metrics(abs_errors_win, delta_eps = data_gt['tbin_res']*1000)
        # np_utils.print_error_metrics(win_errors)
        data_mem = data_gt.copy()
        data_mem.update(c_vals = fovea_full, decoded_depths = decoded_depths_windows, window_size = args.win_size,win_t_start = win_t_start,fovea_data=fovea_data)

    if args.depth_fovea:
        (rep_tau_ds, rep_freq_ds, tbin_res_ds, t_domain_ds, max_depth_ds, tbin_depth_res_ds) = tof_utils.calc_tof_domain_params(args.ds_bins, max_depth=args.max_depth)
        print("DS_depthres {}".format(tbin_depth_res_ds))
        ds_sim = foveation.downsample_hist(data_gt['c_vals'],args.ds_bins)
        # fovea_box = foveation.get_bb_coords(data_gt['depth_imgs'])
        # replaced = foveation.depth_replacement(ds_sim,data_gt['c_vals'],fovea_box)
        depths_ds = eval_coding_utils.decode_peak(data_gt['coding_obj'], ds_sim, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*tbin_depth_res_ds
        # depths_rep = eval_coding_utils.decode_peak(data_gt['coding_obj'], replaced, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*(data_gt['tbin_depth_res'])
        if args.mem_fovea:
            n_winbin = int((args.max_depth*args.ds_bins)/(args.win_size*data_gt['tbin_depth_res']))
            (rep_tau_ds_win, rep_freq_ds_win, tbin_res_ds_win, t_domain_ds_win, max_depth_ds_win, tbin_depth_res_ds_win) = tof_utils.calc_tof_domain_params(n_winbin, max_depth=args.max_depth)
            print("DSWindow_depthres {}".format(tbin_depth_res_ds_win))

            ds_win = foveation.downsample_hist(fovea_data,args.ds_bins)
            ds_win = foveation.gen_full_sig_ds(ds_win,win_t_start,n_winbin,data_gt['tbin_depth_res'],tbin_depth_res_ds_win)
            # win_rep = foveation.depth_replacement(ds_sim,ds_win,fovea_box)
            depths_ds_win = eval_coding_utils.decode_peak(data_gt['coding_obj'], ds_win, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*tbin_depth_res_ds_win
        data_ds = data_gt.copy()
        data_ds.update(c_vals = ds_win, decoded_depths = depths_ds_win, c_vals_ds = ds_sim,decoded_depths_ds=depths_ds,window_size=args.win_size, ds_nbins = args.ds_bins)
        
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.title("Memory Foveation")
        # plt.imshow(data_mem['decoded_depths'],vmin=0, vmax=10)

        # plt.subplot(1,3,2)
        # plt.title("1/32 Res Histogram")
        # plt.imshow(depths_ds,vmin=0, vmax=10)

        # # plt.subplot(2,2,3)
        # # plt.title("Full Res Inpainting")
        # # plt.imshow(depths_rep)

        # plt.subplot(1,3,3)
        # plt.title("Depth Foveation")
        # plt.imshow(depths_ds_win,vmin=0, vmax=10)
        
        # plt.show()
        # # input("YEE")
        


    # sim.plotResults(data_mem)

    # sim.plotResults(data_gt)
    # sim.plotResults(data_pred)
    if args.eval:
        gt_sim_metrics.update(helpers.compute_metrics(depth_gt,data_gt['decoded_depths']))
        data_gt.update(metrics=gt_sim_metrics)
        if args.mem_fovea:
            mem_fovea_metrics.update(helpers.compute_metrics(depth_gt,data_mem['decoded_depths']))
            data_mem.update(metrics=mem_fovea_metrics)
        if args.depth_fovea:
            depth_fovea_metrics.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths']))
            ds_sim_metrics.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths_ds']))
            data_ds.update(metrics=depth_fovea_metrics,metrics_ds=ds_sim_metrics)
            ds_v_dfovea_metrics.update(helpers.compute_metrics(data_ds['decoded_depths'],data_ds['decoded_depths_ds']))
        if args.running_avg:
            gt_sim_metrics_run.update(helpers.compute_metrics(depth_gt,data_gt['decoded_depths']))
            mem_fovea_metrics_run.update(helpers.compute_metrics(depth_gt,data_mem['decoded_depths']))
            depth_fovea_metrics_run.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths']))
            ds_sim_metrics_run.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths_ds']))
            ds_v_dfovea_metrics_run.update(helpers.compute_metrics(data_ds['decoded_depths'],data_ds['decoded_depths_ds']))


    if args.save_results:
        if P % (100 / args.save_percent) == 0 and P // (100 / args.save_percent) < num_images_to_save:
            sim.saveResults(data_gt,saveData=True,savePlots=1,out_data_base_dirpath=out_data_dirpath,exper_type="fullRes",file_name=nyu_val_number)
            if args.mem_fovea:
                sim.saveResults(data_mem,saveData=True,savePlots=2,out_data_base_dirpath=out_data_dirpath,exper_type="memoryFovea",file_name=nyu_val_number)
            if args.depth_fovea:
                sim.saveResults(data_ds,saveData=True,savePlots=3,out_data_base_dirpath=out_data_dirpath,exper_type="depthFovea",file_name=nyu_val_number)



if args.running_avg:
        running_avg_fpath = os.path.join(os.path.split(out_data_dirpath)[0], "running_avg"+'.npz')
        def r(m): return m
        np.savez(running_avg_fpath,
                 gt_sim = {k: r(v) for k, v in gt_sim_metrics_run.get_value().items()},
                 mem_fovea = {k: r(v) for k, v in mem_fovea_metrics_run.get_value().items()} ,
                 depth_fovea = {k: r(v) for k, v in depth_fovea_metrics_run.get_value().items()},
                 ds_sim = {k: r(v) for k, v in ds_sim_metrics_run.get_value().items()},
                 ds_v_dfovea = {k: r(v) for k, v in ds_v_dfovea_metrics_run.get_value().items()})




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