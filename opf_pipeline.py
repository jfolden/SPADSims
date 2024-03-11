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
from helpers import RunningAverageDict
import foveation
import sim
import cv2



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
parser.add_argument('--fovea_threshold', default=False, action='store_true', help='Enable thresholding of fovea window')




args = parser.parse_args()
# if (args.win_size % 2) != 0:
#     args.win_size += 1

num_files_to_sim = args.n_imgs
num_images_to_save = int(np.ceil(num_files_to_sim * args.save_percent / 100))

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
# carla_gt_fnames = ["F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene3.npz"]
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene1.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene2.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene3.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene4.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene5.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene6.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene7.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene8.npz",
# #           "F:\\Research\\Collision_Prediction\\foveation_data\\combined_scene9.npz"]

# fnames = ["D:\\carla_results_480x640\\combined_scene3"]#,
# #         #   "D:\\carla_results\\combined_scene1",
# #         #   "D:\\carla_results\\combined_scene2",
# #         #   "D:\\carla_results\\combined_scene3",
# #         #   "D:\\carla_results\\combined_scene4",
# #         #   "D:\\carla_results\\combined_scene5",
# #         #   "D:\\carla_results\\combined_scene6",
# #         #   "D:\\carla_results\\combined_scene7",
# #         #   "D:\\carla_results\\combined_scene8",
# #         #   "D:\\carla_results\\combined_scene9"]
# fnames = ["D:\\carla_results_480x640\\scene0","D:\\carla_results_480x640\\scene7"]
# carla_gt_fnames = ["D:\\carla_GT\\scene0.npz","D:\\carla_GT\\scene7.npz"]#, ["D:\\carla_GT\\scene2.npz"]
fnames = ["D:\\carla_SPADSim_results_480x640\\scene0","D:\\carla_SPADSim_results_480x640\\scene2","D:\\carla_SPADSim_results_480x640\\scene7","D:\\carla_SPADSim_results_480x640\\scene9","D:\\carla_SPADSim_results_480x640\\scene5"]
carla_gt_fnames = ["D:\\carla_GT\\scene0.npz","D:\\carla_GT\\scene2.npz","D:\\carla_GT\\scene7.npz","D:\\carla_GT\\scene9.npz","D:\\carla_GT\\scene5.npz"]#, ["D:\\carla_GT\\scene2.npz"]

# fnames = ["D:\carla_GT\scene0"]

curr_date = datetime.now()
daymonthyear = curr_date.strftime("%m_%d_%Y")


# monomodel = helpers.loadModel()
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

    fnames_split0 = os.path.split(carla_gt_fnames[P])
    # fnames_split1 = os.path.split(fnames_split0[0])
    nyu_val_number = fnames_split0[1][:-4]

    
    out_data_base_dirpath = 'D:/carla_DepthFoveation/{}'.format(nyu_val_number)
    os.makedirs(out_data_base_dirpath, exist_ok=True)
    
    # out_data_dirpath = os.path.join(out_data_base_dirpath,'{checks}_{}/{}'.format(curr_date.strftime("%H%M"),nyu_val_number,checks="local_scale" if args.local_scale else "general_scale"))
    
    #Carla Load
    [rgb_gt,flow_gt,depth_gt] = helpers.load_carla_data(carla_gt_fnames[P])
    rgb_gt = rgb_gt.astype(np.uint8)

    scene_img_paths = helpers.get_npy_paths(fnames[P])
    depths_n_codes = scene_img_paths[-2:]
    del scene_img_paths[-2:]


    sim_depths = np.load(depths_n_codes[0],mmap_mode='r')
    sim_codes = np.load(depths_n_codes[1],allow_pickle=True)
    sim_codes = sim_codes.item()
    # if nyu_val_number == 'scene0':
    #     rgb_gt = rgb_gt[4:24].astype(np.uint8)
    #     flow_gt = flow_gt[4:24]
    #     depth_gt = depth_gt[4:24]
    #     sim_depths = sim_depths[4:24]
    #     scene_img_paths = scene_img_paths[4:24]
    # elif nyu_val_number == 'scene7':
    #     rgb_gt = rgb_gt[85:105].astype(np.uint8)
    #     flow_gt = flow_gt[85:105]
    #     depth_gt = depth_gt[85:105]
    #     sim_depths = sim_depths[85:105]
    #     scene_img_paths = scene_img_paths[85:105]
    print(scene_img_paths)

    

    #Deal with the max value
    # depth_gt[depth_gt>200] = 0
    depth_gt[depth_gt==1000] = 0
    args.max_depth = np.max(depth_gt)

    key_frames_list = []
    depth_frames = np.zeros_like(depth_gt)
    depthDS_frames = np.zeros_like(depth_gt)
    depthFovea_frames = np.zeros_like(depth_gt)
    prev_pixel_arr = np.zeros(flow_gt.shape,np.int)
    prev_dmap_arr = np.zeros_like(depth_gt)
    edges_arr = np.zeros_like(depth_gt,dtype=bool)
    push_pulls = []
    flow3d = np.zeros((flow_gt.shape[0],flow_gt.shape[1],flow_gt.shape[2],3))
    win_start_arr = []
    win_start_old = []
    num_pixels_cheated = []
    index_for_viz = []
    # args.max_depth = 200
    print("Simming: "+fnames[P])
    for i in tqdm(range(rgb_gt.shape[0]),desc="Scene Foveation"):
        # frame_flow = flow_gt[i]
        frame_flow = helpers.calculate_optical_flow(rgb_gt[i-1],rgb_gt[i])
        prev_pixels = helpers.calc_prev_pixel_cv2(frame_flow)
        prev_pixel_arr[i] = prev_pixels
        #First Keyframe
        if i == 0:# or i % 10==0:
            key_frame_hist = np.load(scene_img_paths[i],mmap_mode='r')
            key_frame_depth = sim_depths[i]
            np.save(os.path.join(out_data_base_dirpath,f'hist_{i:04d}.npy'),key_frame_hist.astype(np.int16))
            depth_frames[i] = key_frame_depth
            key_frames_list.append(i)
            flow3d[i] = np.stack((frame_flow[:,:,0],frame_flow[:,:,1],(key_frame_depth/sim_codes['tbin_depth_res']).astype(int)),axis=-1)
            index_for_viz.append([(0,0)])
            print("saved first keyframe")
            continue

        
        if args.mem_fovea:
            frame_depth = sim_depths[i]
            frame_hist = np.load(scene_img_paths[i],mmap_mode='r').squeeze()
            # frame_flow = flow_gt[i]
            
            # edges = cv2.Canny(frame_depth.astype(np.uint8),0,10)
            # edges = cv2.dilate(edges, np.ones((3,3),dtype=np.uint8), iterations=4).astype(bool)
            # edges_arr[i] = edges
            # frame_flow[frame_flow[:,:,0]>frame_depth.shape[0]] = frame_depth.shape[0]
            # frame_flow[frame_flow[:,:,1]>frame_depth.shape[1]] = frame_depth.shape[0]
            
            # if np.max(frame_flow) > 15.0:
            #     key_frame_hist = frame_hist
            #     key_frame_depth = frame_depth
            #     np.save(os.path.join(out_data_base_dirpath,f'hist_{i:04d}.npy'),key_frame_hist.astype(np.int16))
            #     print(f"saved keyframe at location{i}")
            #     depth_frames[i] = key_frame_depth
            #     key_frames_list.append(i)
            #     continue
            # else:
                

            #create dmap based on preivous pixels and produce the transients
            # prev_dmap = depth_frames[i-1,prev_pixels[:,:,1],prev_pixels[:,:,0]]
            prev_dmap = cv2.remap(depth_frames[i-1],prev_pixels,None,interpolation=cv2.INTER_LINEAR)
            # print(np.max(prev_dmap))
            # print(np.max(depth_gt))
            [_,nzindi] = tof_utils.depthmap2tirf(prev_dmap,n_tbins = args.n_tbins, delta_depth = args.max_depth / (args.n_tbins-1))
            # push_pull = (prev_dmap//sim_codes['tbin_depth_res']-flow3d[i-1,:,:,2]).astype(int)
            push_pull = (flow3d[i-1,:,:,2]-flow3d[i-2,:,:,2]).astype(int)
            push_pulls.append(push_pull)
            if args.fovea_threshold:
                 [fovea_data,win_t_start,fovea_full,npc_full,idx_for_viz] = foveation.fovea_window_threshold(window_size=args.win_size,histo=frame_hist,nz_indi=nzindi,threshold=50,n_tbins=args.n_tbins)
                 num_pixels_cheated.append(npc_full)
                 index_for_viz.append(idx_for_viz)
            else:
                [fovea_data,win_t_start] = foveation.fovea_window(window_size=args.win_size,histo=frame_hist,nz_indi=nzindi)
                fovea_full = foveation.gen_full_sig(fovea_data,win_t_start,args.n_tbins)
            fovea_full = fovea_full.astype(np.float32)
            win_start_old.append(win_t_start)
            # [fovea_data,win_t_start] = foveation.fovea_window_flow(window_size=args.win_size,histo=frame_hist,nz_indi=nzindi,push_pull=push_pull)
            # win_start_arr.append(win_t_start)

            # fovea_full[edges] = frame_hist[edges]
            decoded_depths_windows = eval_coding_utils.decode_peak(sim_codes['coding_obj'], fovea_full, sim_codes['coding_id'], sim_codes['rec'], sim_codes['pw_factor'])*sim_codes['tbin_depth_res']#(args.max_depth/args.n_tbins)#(200/2000)
            flow3d[i] = np.stack((frame_flow[:,:,0],frame_flow[:,:,1],(decoded_depths_windows*sim_codes['tbin_depth_res']).astype(int)),axis=-1)
            np.save(os.path.join(out_data_base_dirpath,f'hist_{i:04d}.npy'),fovea_full.astype(np.int16))
            depth_frames[i] = decoded_depths_windows
            prev_dmap_arr[i] = prev_dmap

        if args.depth_fovea:
            (rep_tau_ds, rep_freq_ds, tbin_res_ds, t_domain_ds, max_depth_ds, tbin_depth_res_ds) = tof_utils.calc_tof_domain_params(args.ds_bins, max_depth=args.max_depth)
            print("DS_depthres {}".format(tbin_depth_res_ds))
            ds_sim = foveation.downsample_hist(frame_hist,args.ds_bins)
            # fovea_box = foveation.get_bb_coords(data_gt['depth_imgs'])
            # replaced = foveation.depth_replacement(ds_sim,data_gt['c_vals'],fovea_box)
            depths_ds = eval_coding_utils.decode_peak(sim_codes['coding_obj'], ds_sim, sim_codes['coding_id'], sim_codes['rec'], sim_codes['pw_factor'])*sim_codes['tbin_depth_res']
            # depths_rep = eval_coding_utils.decode_peak(data_gt['coding_obj'], replaced, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*(data_gt['tbin_depth_res'])
            
            if args.mem_fovea:
                n_winbin = int((np.max(depth_gt)*args.ds_bins)/(args.win_size*sim_codes['tbin_depth_res']))
                (rep_tau_ds_win, rep_freq_ds_win, tbin_res_ds_win, t_domain_ds_win, max_depth_ds_win, tbin_depth_res_ds_win) = tof_utils.calc_tof_domain_params(n_winbin, max_depth=args.max_depth)
                print("DSWindow_depthres {}".format(tbin_depth_res_ds_win))

                ds_win = foveation.downsample_hist(fovea_data,args.ds_bins)
                ds_win = foveation.gen_full_sig_ds(ds_win,win_t_start,n_winbin,sim_codes['tbin_depth_res'],tbin_depth_res_ds_win)
                # win_rep = foveation.depth_replacement(ds_sim,ds_win,fovea_box)
                depths_ds_win = eval_coding_utils.decode_peak(sim_codes['coding_obj'], ds_win, sim_codes['coding_id'], sim_codes['rec'], sim_codes['pw_factor'])*sim_codes['tbin_depth_res']

                np.save(os.path.join(out_data_base_dirpath,f'depthFovea_hist_{i:04d}.npy'),ds_win.astype(np.int16))
                np.save(os.path.join(out_data_base_dirpath,f'downSamp_hist_{i:04d}.npy'),ds_sim.astype(np.int16))
                depthDS_frames[i] = depths_ds
                depthFovea_frames[i] = depths_ds_win
            # data_ds = data_gt.copy()
            # data_ds.update(c_vals = ds_win, decoded_depths = depths_ds_win, c_vals_ds = ds_sim,decoded_depths_ds=depths_ds,window_size=args.win_size, ds_nbins = args.ds_bins)
        


    np.save(os.path.join(out_data_base_dirpath,'depths.npy'),depth_frames)
    np.save(os.path.join(out_data_base_dirpath,'depths_ds.npy'),depthDS_frames)
    np.save(os.path.join(out_data_base_dirpath,'depths_fovea.npy'),depthFovea_frames)
    np.save(os.path.join(out_data_base_dirpath,'prev_pixels.npy'),prev_pixel_arr)
    np.save(os.path.join(out_data_base_dirpath,'prev_dmaps.npy'),prev_dmap_arr)
    np.save(os.path.join(out_data_base_dirpath,'edges.npy'),edges_arr)
    np.save(os.path.join(out_data_base_dirpath,'pp.npy'),push_pulls)
    # np.save(os.path.join(out_data_base_dirpath,'win_start.npy'),win_start_arr)
    np.save(os.path.join(out_data_base_dirpath,'win_start_old.npy'),win_start_old)
    np.save(os.path.join(out_data_base_dirpath,'flow3d.npy'),flow3d)
    np.save(os.path.join(out_data_base_dirpath,'npc.npy'),num_pixels_cheated)
    np.save(os.path.join(out_data_base_dirpath,'index_viz.npy'),index_for_viz)
    print("saved depths")

        # data_gt = sim.simSPAD(args,rgb=rgb_gt[i],depth=depth_gt[i])

    
    # for i in range(1,rgb_gt.shape[0]):
        # prev_pixels = helpers.calc_prev_pixel(flow_gt[i])


    # [_,pred_nzindi] = tof_utils.depthmap2tirf(resize(predDepth_gt_raw,(args.n_rows,args.n_cols)),n_tbins = args.n_tbins, delta_depth = 10 / (args.n_tbins-1)) #Maximum depth in NYU is 10m)


    # if args.mem_fovea:
    #     [fovea_data,win_t_start] = foveation.fovea_window(window_size=args.win_size,histo=data_gt['c_vals'],nz_indi=pred_nzindi)#data_pred['nonzero_ind']
    #     fovea_full = foveation.gen_full_sig(fovea_data,win_t_start,args.n_tbins)
    #     decoded_depths_windows = eval_coding_utils.decode_peak(data_gt['coding_obj'], fovea_full, data_gt['coding_id'], args.rec[0], data_gt['pw_factor'])*data_gt['tbin_depth_res']
    #     abs_errors_win = np.abs(decoded_depths_windows.squeeze() - resize(depth_gt,(args.n_rows,args.n_cols)))*1000
    #     win_errors = np_utils.calc_error_metrics(abs_errors_win, delta_eps = data_gt['tbin_res']*1000)
    #     np_utils.print_error_metrics(win_errors)
    #     data_mem = data_gt.copy()
    #     data_mem.update(c_vals = fovea_full, decoded_depths = decoded_depths_windows,abs_depth_errors=abs_errors_win,error_metrics=win_errors, window_size = args.win_size)


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
    # if args.eval:
    #     gt_sim_metrics.update(helpers.compute_metrics(depth_gt,data_gt['decoded_depths']))
    #     data_gt.update(metrics=gt_sim_metrics)
    #     if args.mem_fovea:
    #         mem_fovea_metrics.update(helpers.compute_metrics(depth_gt,data_mem['decoded_depths']))
    #         data_mem.update(metrics=mem_fovea_metrics)
    #     if args.depth_fovea:
    #         depth_fovea_metrics.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths']))
    #         ds_sim_metrics.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths_ds']))
    #         data_ds.update(metrics=depth_fovea_metrics,metrics_ds=ds_sim_metrics)
    #         ds_v_dfovea_metrics.update(helpers.compute_metrics(data_ds['decoded_depths'],data_ds['decoded_depths_ds']))
    #     if args.running_avg:
    #         gt_sim_metrics_run.update(helpers.compute_metrics(depth_gt,data_gt['decoded_depths']))
    #         mem_fovea_metrics_run.update(helpers.compute_metrics(depth_gt,data_mem['decoded_depths']))
    #         depth_fovea_metrics_run.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths']))
    #         ds_sim_metrics_run.update(helpers.compute_metrics(depth_gt,data_ds['decoded_depths_ds']))
    #         ds_v_dfovea_metrics_run.update(helpers.compute_metrics(data_ds['decoded_depths'],data_ds['decoded_depths_ds']))


    # if args.save_results:
    #     if P % (100 / args.save_percent) == 0 and P // (100 / args.save_percent) < num_images_to_save:
    #         sim.saveResults(data_gt,saveData=False,savePlots=1,out_data_base_dirpath=out_data_dirpath,exper_type="fullRes",file_name=nyu_val_number)
            # if args.mem_fovea:
            #     sim.saveResults(data_mem,saveData=False,savePlots=2,out_data_base_dirpath=out_data_dirpath,exper_type="memoryFovea",file_name=nyu_val_number)
            # if args.depth_fovea:
            #     sim.saveResults(data_ds,saveData=False,savePlots=3,out_data_base_dirpath=out_data_dirpath,exper_type="depthFovea",file_name=nyu_val_number)



# if args.running_avg:
#         running_avg_fpath = os.path.join(os.path.split(out_data_dirpath)[0], "running_avg"+'.npz')
#         def r(m): return m
#         np.savez(running_avg_fpath,
#                  gt_sim = {k: r(v) for k, v in gt_sim_metrics_run.get_value().items()},
#                  mem_fovea = {k: r(v) for k, v in mem_fovea_metrics_run.get_value().items()} ,
#                  depth_fovea = {k: r(v) for k, v in depth_fovea_metrics_run.get_value().items()},
#                  ds_sim = {k: r(v) for k, v in ds_sim_metrics_run.get_value().items()},
#                  ds_v_dfovea = {k: r(v) for k, v in ds_v_dfovea_metrics_run.get_value().items()})




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

# helpers.close_model(monomodel)