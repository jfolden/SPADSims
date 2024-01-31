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

def simSPAD(args,**kwargs):
    max_depth_input = args.max_depth
    max_path_length = 2*max_depth_input

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
    # print("Number of total bins = {} ".format(n_tbins))

    ## Set rep frequency depending on the domain of the simulated transient
    (rep_tau, rep_freq, tbin_res, t_domain, max_depth, tbin_depth_res) = tof_utils.calc_tof_domain_params(n_tbins, max_path_length=max_path_length)
    # print(f'tbindepth_res{tbin_depth_res}')
    # print(f'max_depth:{max_depth}')
    ## Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors*tbin_res, mu=0, t_domain=t_domain)

    ## initialize coding strategies
    coding_list = coding_utils.init_coding_list(coding_ids, n_tbins, args, pulses_list=pulses_list)




    rgb = kwargs['rgb']
    depth = kwargs['depth']
   
    rgb = rgb.astype(np.float64)
    rgb = resize(rgb,(n_rows,n_cols,3))
    # print('RGB:{}'.format(rgb.shape))
    lumi = tof_utils.rgb2Lumi(rgb)

    depth = resize(depth,(n_rows,n_cols))
    (min_depth_val, max_depth_val) = plot_utils.get_good_min_max_range(depth[depth < max_depth])
    (min_depth_val, max_depth_val) = (min_depth_val*1000, max_depth_val*1000)
    delta_depth = max_depth_input / (n_tbins-1) #Maximum depth in NYU is 10m
    # delta_depth = max_depth_val / (n_tbins-1)
    # print(np.max(depth))
    # delta_depth = np.max(depth) / (n_tbins-1) #Set max depth to the max of the image
    # print(delta_depth)

    (min_depth_error_val, max_depth_error_val ) = (0, 110)
    #convert depth map to transient img
    [transients,nz_ind] = tof_utils.depthmap2tirfLumi(depth,n_tbins,delta_depth,lumi)
    # transients = tof_utils.depthmap2tirf(depth,n_tbins,delta_depth)


    #create the sim objects
    transient_obj = tirf.TemporalIRF(pulses_list[0].apply(transients), t_domain=t_domain)
    # scene_obj = tirf_scene.ToFScene(transient_obj,rgb)
    no_ambiant = np.ones_like(rgb)*1e-6
    scene_obj = tirf_scene.ToFScene(transient_obj,no_ambiant)



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
                


                # Estimate depths
                decoded_depths = eval_coding_utils.decode_peak(coding_obj, c_vals, coding_id, rec_algo, pw_factor)*tbin_depth_res

                ## Calc error metrics
                abs_depth_errors = np.abs(decoded_depths.squeeze() - depth)*1000
                error_metrics = np_utils.calc_error_metrics(abs_depth_errors, delta_eps = tbin_depth_res*1000)
                # np_utils.print_error_metrics(error_metrics)

                dataDict = {"decoded_depths": decoded_depths,
                                'abs_depth_errors': abs_depth_errors,
                                'error_metrics': error_metrics,
                                'c_vals': c_vals,
                                'rep_freq': rep_freq,
                                'rep_tau': rep_tau,
                                'tbin_res': tbin_res,
                                't_domain': t_domain,
                                'max_depth': max_depth,
                                'tbin_depth_res': tbin_depth_res,
                                'Cmat': coding_obj.C,
                                'Cmat_decoding': coding_obj.decoding_C,
                                'rgb_imgs' : rgb,
                                'transients' : transients,
                                'depth_imgs' : depth,
                                'nonzero_ind' : nz_ind,
                                'coding_obj' : coding_obj,
                                'coding_id' : coding_id,
                                'curr_mean_nphotons' : curr_mean_nphotons,
                                'curr_mean_sbr' : curr_mean_sbr,
                                'coding_params_str' : coding_params_str,
                                'min_depth_error_val' : min_depth_error_val,
                                'max_depth_error_val' : max_depth_error_val,
                                'min_depth_val' : min_depth_val,
                                'max_depth_val' : max_depth_val,
                                'pw_factor' : pw_factor,
                                'rec' : rec_algo
                }

                # out_data_base_dirpath = 'F:/Research/compressive-spad-lidar-cvpr22/data/nyu_results'

    return dataDict

def plotResults(dataDict):
        plt.rcParams['image.cmap']='plasma'
        plt.close('all')
        rgb = dataDict['rgb_imgs']
        depth = dataDict['depth_imgs']
        abs_depth_errors = dataDict['abs_depth_errors']
        decoded_depths = dataDict['decoded_depths']
        c_vals = dataDict['c_vals']
        min_depth_error_val = dataDict['min_depth_error_val']
        max_depth_error_val = dataDict['max_depth_error_val']
        min_depth_val = dataDict['min_depth_val']
        max_depth_val = dataDict['max_depth_val']
       ## Plot depths and depth errors
        plt.clf()
        plt.subplot(2,2,1)
        # img=plt.imshow(improc_ops.gamma_tonemap(rgb, gamma=1/4))
        img = plt.imshow(rgb)
        plot_utils.remove_ticks()
        plt.title("RGB Image")
        plt.subplot(2,2,2)
        img=plt.imshow(depth*1000, vmin=min_depth_val, vmax=max_depth_val)
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


def saveResults(dataDict,saveData,savePlots,out_data_base_dirpath,exper_type,file_name):
        plt.rcParams['image.cmap']='plasma'

        rgb = dataDict['rgb_imgs']
        depth = dataDict['depth_imgs']
        abs_depth_errors = dataDict['abs_depth_errors']
        decoded_depths = dataDict['decoded_depths']
        c_vals = dataDict['c_vals']
        min_depth_error_val = dataDict['min_depth_error_val']
        max_depth_error_val = dataDict['max_depth_error_val']
        min_depth_val = dataDict['min_depth_val']
        max_depth_val = dataDict['max_depth_val']
        curr_mean_nphotons = dataDict['curr_mean_nphotons']
        curr_mean_sbr = dataDict['curr_mean_sbr']
        coding_id = dataDict['coding_id']
        rec_algo = dataDict['rec']
        pw_factor = dataDict['pw_factor']
        coding_obj = dataDict['coding_obj']




        out_data_dirpath = os.path.join(out_data_base_dirpath, '{}'.format(exper_type))
        # out_data_dirpath = out_data_base_dirpath

        os.makedirs(out_data_dirpath, exist_ok=True)
        if(saveData):
            print("Saving result Data, this may take a while......")
            out_fname_base = dataDict['coding_params_str']
            # np.savez(os.path.join(out_data_dirpath, out_fname_base+'.npz')
            #             , decoded_depths= dataDict['decoded_depths']
            #             , abs_depth_errors= dataDict['abs_depth_errors']
            #             , error_metrics= dataDict['error_metrics']
            #             , c_vals = dataDict['c_vals']
            #             , rep_freq = dataDict['rep_freq']
            #             , rep_tau = dataDict['rep_tau']
            #             , tbin_res = dataDict['tbin_res']
            #             , t_domain = dataDict['t_domain']
            #             , max_depth = dataDict['max_depth']
            #             , tbin_depth_res = dataDict['tbin_depth_res']
            #             , Cmat = coding_obj.C
            #             , Cmat_decoding = coding_obj.decoding_C
            #             , rgb_imgs = dataDict['rgb_imgs']
            #             , transients = dataDict['transients']
            #             , depth_imgs = dataDict['depth_imgs']
            # )
            np.save(os.path.join(out_data_dirpath, file_name+'.npy'), dataDict)




        out_dirpath = out_data_dirpath

        x_labels = np.linspace(0,round(max_depth_val-1000,-3),4)
        if savePlots == 1:
            print("Saving Results...")
            # sim_params_str = 'np-{:.2f}_sbr-{:.2f}'.format(curr_mean_nphotons, curr_mean_sbr)
            # out_dirpath = os.path.join(out_data_dirpath, sim_params_str)

            # coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo, pw_factor)

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
            # plt.xticks(x_labels,x_labels)
            plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_fullsim', file_ext='pdf')
            #plt.show()
            #Errors
            # plt.figure()
            # plot_utils.update_fig_size(height=5, width=6)
            # img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
            # plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            # plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_fullsimErrors', file_ext='pdf')
            # #plt.show()

            plt.figure()
            img = plt.imshow(np.sum(np.squeeze(c_vals),2),cmap='gray')
            plot_utils.update_fig_size(height=5, width=6)
            plot_utils.remove_ticks()
            # plt.title("Grayscale from Histogram")
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_histGS', file_ext='pdf')

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(improc_ops.gamma_tonemap(rgb, gamma=1))
            plot_utils.remove_ticks()
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'nyu'+'_rgb', file_ext='pdf')
            #plt.show()

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(depth*1000, vmin=min_depth_val, vmax=max_depth_val)
            # plt.xticks(x_labels,x_labels)
            plot_utils.remove_ticks()
            plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'nyu'+'_depths', file_ext='pdf')
            #plt.show()

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(dataDict["mono_raw"]*1000, vmin=min_depth_val, vmax=np.max(dataDict["mono_raw"])*1000)
            # plt.xticks(np.linspace(0,round((np.max(dataDict["mono_raw"])*1000)-1000,-3),len(x_labels)), np.linspace(0,round((np.max(dataDict["mono_raw"])*1000)-1000,-3),len(x_labels)))
            plot_utils.remove_ticks()

            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'monocular_depth', file_ext='pdf')
            # #plt.show()
        elif savePlots == 2:
            print("Saving Results...")
            # sim_params_str = 'np-{:.2f}_sbr-{:.2f}'.format(curr_mean_nphotons, curr_mean_sbr)
            # out_dirpath = os.path.join(out_data_dirpath, sim_params_str)
            # coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo, pw_factor)

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
            plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_memDepths', file_ext='pdf')
            #plt.show()

            # plt.figure()
            # plot_utils.update_fig_size(height=5, width=6)
            # img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
            # plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            # plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_memErrors', file_ext='pdf')
            # #plt.show()

            # plt.figure()
            # img = plt.imshow(np.sum(np.squeeze(c_vals),2),cmap='gray')
            # plot_utils.remove_ticks()
            # plt.title("Grayscale from Histogram")
            # plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_memGS', file_ext='pdf')
            if 'sparse_depths' in dataDict:
                from scipy.ndimage import grey_dilation
                import matplotlib as mpl
                import copy
                fig, ax = plt.subplots()
                img = dataDict['sparse_depths']#np.sum(np.squeeze(c_vals),2)
                dia_img = grey_dilation(img,size=(7, 7))
                dia_img = np.ma.masked_where(dia_img==0,dia_img)
                cmap = copy.copy(mpl.cm.get_cmap('plasma'))
                cmap.set_bad(color='black')
                ax.set_facecolor('black')
                ax.imshow(dia_img,cmap=cmap)

                plot_utils.remove_ticks()
                # plt.title("Grayscale from Histogram")
                plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_sparse', file_ext='pdf')

            if 'quant_mono' in dataDict:
                cmap = plt.get_cmap('plasma')
                img = plt.imshow(dataDict['quant_mono']*1000,cmap=cmap)
                plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
                plot_utils.remove_ticks()
                # plt.title("Grayscale from Histogram")
                plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_quant', file_ext='pdf')

        elif savePlots == 3:
            print("Saving Results...")
            # sim_params_str = 'np-{:.2f}_sbr-{:.2f}'.format(curr_mean_nphotons, curr_mean_sbr)
            # out_dirpath = os.path.join(out_data_dirpath, sim_params_str)
            # coding_params_str = eval_coding_utils.compose_coding_params_str(coding_id, coding_obj.n_codes, rec_algo, pw_factor)

            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(decoded_depths.squeeze()*1000, vmin=min_depth_val, vmax=max_depth_val)
            # plt.xticks(x_labels,x_labels)
            plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_DSDepths', file_ext='pdf')
            #plt.show()

            # plt.figure()
            # plot_utils.update_fig_size(height=5, width=6)
            # img=plt.imshow(abs_depth_errors, vmin=min_depth_error_val, vmax=max_depth_error_val)
            # plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            # plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_DSDepthsErrors', file_ext='pdf')
            # #plt.show()

            plt.figure()
            img = plt.imshow(np.sum(np.squeeze(dataDict['c_vals_ds']),2),cmap='gray')
            plot_utils.remove_ticks()
            # plt.title("Grayscale from Histogram")
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_DSGS', file_ext='pdf')


            plt.figure()
            plot_utils.update_fig_size(height=5, width=6)
            img=plt.imshow(dataDict["decoded_depths_ds"]*1000, vmin=min_depth_val, vmax=max_depth_val)
            # plt.xticks(x_labels,x_labels)
            plot_utils.remove_ticks()
            # plot_utils.set_cbar_ticks(img, cbar_orientation='horizontal',Ticks=x_labels)
            plot_utils.save_currfig(dirpath=out_dirpath, filename=file_name+'_DS', file_ext='pdf')
            #plt.show()
        else:
             print("Not Saving Plots")


