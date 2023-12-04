'''
to run the stuff do this
python simmer.py -sbr 1.0 -n_rows 240 -n_cols 320 -nphotons 1000 -n_tbins 2000 -coding Identity --account_irf --save_data_results --save_results


'''
import argparse
import os
import sys
sys.path.append('./tof-lib')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger
from skimage.transform import resize
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
import eval_coding_utils
import helpers
import foveation

parser = argparse.ArgumentParser(description='Parser for flash lidar simulation.')
add_flash_lidar_scene_args(parser)
add_eval_coding_args(parser)
parser.add_argument('--save_results', default=False, action='store_true', help='Save result images.')
parser.add_argument('--save_data_results', default=False, action='store_true', help='Save results data.')
args = parser.parse_args()
max_path_length = args.max_transient_path_len

## Get coding ids and reconstruction algos and verify their lengths
coding_ids = args.coding
rec_algos_ids = args.rec
pw_factors = np_utils.to_nparray(args.pw_factors)
n_coding_schemes = len(coding_ids)
(coding_scheme_ids, rec_algos_ids, pw_factors) = eval_coding_utils.generate_coding_scheme_ids(coding_ids, rec_algos_ids, pw_factors)

## Set signal and sbr levels at which the MAE will be calculated at
(signal_levels, sbr_levels, nphotons_levels) = eval_coding_utils.parse_signalandsbr_params(args)
(X_sbr_levels, Y_nphotons_levels) = np.meshgrid(sbr_levels, nphotons_levels)
n_nphotons_lvls = len(nphotons_levels)
n_sbr_lvls = len(sbr_levels)

# Parse input args
n_tbins = args.n_tbins
n_rows = args.n_rows
n_cols = args.n_cols

## Set rep frequency depending on the domain of the simulated transient
(rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_path_length)

num_files_to_sim = 1
all_files = np.genfromtxt('NyuFiles.txt',dtype='str')
fnames = all_files[np.random.randint(0,len(all_files),num_files_to_sim)]
# fnames = all_files[0:num_files_to_sim]

## Create GT gaussian pulses for each coding. Different coding may use different pulse widths
pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=0, t_domain=t_domain)

## initialize coding strategies
coding_list = coding_utils.init_coding_list(coding_ids, n_tbins, args, pulses_list=pulses_list)



#init storage vars for images
rgbs = []
depths = []
transient_imgs = []
curr_date = datetime.now()
daymonthyear = curr_date.strftime("%m_%d_%Y")
for P in range(num_files_to_sim):
    fnames_split0 = os.path.split(fnames[P])
    fnames_split1 = os.path.split(fnames_split0[0]) 

    out_data_base_dirpath = 'F:/Research/compressive-spad-lidar-cvpr22/data/nyu_results/{}/{}/{}'.format(daymonthyear,fnames_split1[1],fnames_split0[1][:-3])

    [rgb,depth,h,w] = helpers.load_nyu(fnames[P])
    print("Simming: "+fnames[P])
    rgb = rgb.transpose((-2,-1,-3)).astype(np.float64)
    rgb = resize(rgb,(n_rows,n_cols,3))
    lumi = tof_utils.rgb2Lumi(rgb)
    depth = resize(depth,(n_rows,n_cols))
    (min_depth_val, max_depth_val) = plot_utils.get_good_min_max_range(depth[depth < max_depth])
    (min_depth_val, max_depth_val) = (min_depth_val*1000, max_depth_val*1000)
    delta_depth = 10 / (n_tbins-1) #Maximum depth in NYU is 10m
    # delta_depth = max_depth_val /(n_tbins-1)
    # delta_depth = np.max(depth) / (n_tbins-1) #Set max depth to the max of the image

    (min_depth_error_val, max_depth_error_val ) = (0, 110)
    #convert depth map to transient img
    [transients,nz_ind] = tof_utils.depthmap2tirfLumi(depth,n_tbins,delta_depth,lumi)
    print(nz_ind)
    # transients = tof_utils.depthmap2tirf(depth,n_tbins,delta_depth)

    #append to storage
    transient_imgs.append(transients)
    rgbs.append(rgb)
    depths.append(depth)

    #create the sim objects
    transient_obj = tirf.TemporalIRF(pulses_list[0].apply(transient_imgs[P]), t_domain=t_domain)
    scene_obj = tirf_scene.ToFScene(transient_obj, rgbs[P])



    for i in range(n_coding_schemes):
        coding_id = coding_ids[i]
        pw_factor = pw_factors[i]
        coding_obj = coding_list[i]
        rec_algo = rec_algos_ids[i]
        coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo=rec_algo, pw_factor=pw_factor, account_irf=args.account_irf)
        for j in range(n_nphotons_lvls):
            for k in range(n_sbr_lvls):
                ## Simulate a dtof image
                curr_mean_sbr = X_sbr_levels[j, k]
                curr_mean_nphotons = Y_nphotons_levels[j, k]
                transient_img_sim = scene_obj.dtof_sim(mean_nphotons=curr_mean_nphotons, mean_sbr=curr_mean_sbr)
                ## Encode
                c_vals = coding_obj.encode(transient_img_sim)
                [fovea_data,win_t_start] = foveation.fovea_window(window_size=100,histo=c_vals,nz_indi=nz_ind)
                print(fovea_data.shape)
                print(win_t_start)
                # Estimate depths
                decoded_depths = eval_coding_utils.decode_peak(coding_obj, c_vals, coding_id, rec_algo, pw_factor)*tbin_depth_res
                ## Calc error metrics
                abs_depth_errors = np.abs(decoded_depths.squeeze() - depths[P])*1000
                error_metrics = np_utils.calc_error_metrics(abs_depth_errors, delta_eps = tbin_depth_res*1000)
                np_utils.print_error_metrics(error_metrics)
                ## Plot depths and depth errors
                plt.clf()
                plt.subplot(2,2,1)
                img=plt.imshow(improc_ops.gamma_tonemap(rgbs[P], gamma=1/4))
                plot_utils.remove_ticks()
                plt.title("RGB Image")
                plt.subplot(2,2,2)
                img=plt.imshow(depths[P]*1000, vmin=min_depth_val, vmax=max_depth_val)
                plot_utils.remove_ticks()
                plot_utils.set_cbar(img)
                plt.title("Depth Image")
                plt.subplot(2,2,3)
                img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
                plot_utils.remove_ticks()
                plot_utils.set_cbar(img)
                plt.title("Absolute depth errors")
                plt.subplot(2,2,4)
                img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
                plot_utils.remove_ticks()
                plot_utils.set_cbar(img)
                plt.title("Decoded Depths")
                plt.show()

                plt.figure()
                img = plt.imshow(np.sum(np.squeeze(c_vals),2),cmap='gray')
                plt.title("Grayscale from Histogram")
                plt.show()

                # out_data_base_dirpath = 'F:/Research/compressive-spad-lidar-cvpr22/data/nyu_results'
                if(args.save_data_results):
                    print("Saving result Data, this may take a while......")
                    out_data_dirpath = os.path.join(out_data_base_dirpath, 'np-{:.2f}_sbr-{:.2f}'.format(curr_mean_nphotons, curr_mean_sbr))
                    os.makedirs(out_data_dirpath, exist_ok=True)
                    out_fname_base = coding_params_str
                    np.savez(os.path.join(out_data_dirpath, out_fname_base+'.npz')
                                , decoded_depths=decoded_depths
                                , abs_depth_errors=abs_depth_errors
                                , error_metrics=error_metrics
                                , c_vals=c_vals
                                , rep_freq=rep_freq
                                , rep_tau=rep_tau
                                , tbin_res=tbin_res
                                , t_domain=t_domain
                                , max_depth=max_depth
                                , tbin_depth_res=tbin_depth_res
                                , Cmat=coding_obj.C
                                , Cmat_decoding=coding_obj.decoding_C
                                , rgb_imgs = rgbs
                                , transients = transient_imgs
                                , depth_imgs = depths 
                    )
					
                if(args.save_results):
                    print("Saving Results...")
                    sim_params_str = '{}_np-{:.2f}_sbr-{:.2f}'.format(P, curr_mean_nphotons, curr_mean_sbr)
                    out_dirpath = os.path.join(out_data_dirpath, sim_params_str)
                    coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo, pw_factor)

                    plt.figure()
                    plot_utils.update_fig_size(height=5, width=6)
                    img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
                    plot_utils.remove_ticks()
                    plot_utils.set_cbar(img, cbar_orientation='horizontal')
                    plot_utils.save_currfig(dirpath=out_dirpath, filename=coding_params_str+'_depths', file_ext='png')
                    #plt.show()

                    plt.figure()
                    plot_utils.update_fig_size(height=5, width=6)
                    img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
                    plot_utils.remove_ticks()
                    plot_utils.set_cbar(img, cbar_orientation='horizontal')
                    plot_utils.save_currfig(dirpath=out_dirpath, filename=coding_params_str+'_deptherrs', file_ext='png')
                    #plt.show()

                    plt.figure()
                    img = plt.imshow(np.sum(np.squeeze(c_vals),2),cmap='gray')
                    plot_utils.remove_ticks()
                    plt.title("Grayscale from Histogram")
                    plot_utils.save_currfig(dirpath=out_dirpath, filename=coding_params_str+'_histGS', file_ext='png')

                if(args.save_results):
                    plt.figure()
                    plot_utils.update_fig_size(height=5, width=6)
                    img=plt.imshow(improc_ops.gamma_tonemap(rgbs[P], gamma=1/4))
                    plot_utils.remove_ticks()
                    plot_utils.save_currfig(dirpath=out_data_base_dirpath, filename='nyu'+'_rgb', file_ext='png')
                    #plt.show()

                    plt.figure()
                    plot_utils.update_fig_size(height=5, width=6)
                    img=plt.imshow(depths[P]*1000, vmin=min_depth_val, vmax=max_depth_val)
                    plot_utils.remove_ticks()
                    # plot_utils.set_cbar(img, cbar_orientation='horizontal')
                    plot_utils.save_currfig(dirpath=out_data_base_dirpath, filename='nyu'+'_depths', file_ext='png')
                    #plt.show()

