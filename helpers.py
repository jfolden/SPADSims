import numpy as np
from scipy.stats import norm
import h5py
import torch
import numpy as np
import sys
from PIL import Image
from sklearn.cluster import KMeans
from scipy.optimize import minimize

sys.path.append('../ZoeDepth')
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize, save_raw_16bit,pil_to_batched_tensor
import torch.nn as nn

def get_p_i(r_i):
    B = len(r_i)
    q_i = 1-np.exp(-r_i)
    p_i = np.zeros(len(q_i)+1)
    p_i[0] = q_i[0]
    prev_one_minus_prod = 1.0
    for i in range(1,B):
        prev_one_minus_prod = prev_one_minus_prod * (1-q_i[i-1])
        p_i[i] = q_i[i] * prev_one_minus_prod #np.prod(1-q_i[0:i])
        #p_i[i] = q_i[i] * np.prod(1-q_i[0:i])

    #p_i = np.append(p_i, 1.0-np.sum(p_i))
    p_i[-1] = 1.0-np.sum(p_i)
    return p_i

def get_deterministic_shift_sequence(L,B):
    if L<B:
        s_l = np.array(np.floor(np.arange(L,dtype=float)*B/L),dtype=np.int)
    else:
        num_full_cycles = np.int(L)//np.int(B)
        num_extra_cycles = L - B*num_full_cycles
        s_l = np.zeros(L,dtype=np.int)
        for i in range(num_full_cycles):
            s_l[i*B:(i+1)*B] = range(B)
        s_l[B*num_full_cycles:] = np.array(np.floor(np.arange(num_extra_cycles,dtype=float)*B/num_extra_cycles),dtype=np.int)

    return(s_l)

def get_upsilon_opt(B, Phi_bkg):
    # the optimal filtering fraction reduces the flux to ~1 photon/laser cycle
    opt_att = 1./B/Phi_bkg
    return min(1.0,opt_att)

def att_frac_to_od(frac):
    return np.log10(1/frac)

def od_to_att_frac(od):
    return 10.0**(-od)

def generate_true_waveform(Phi_sig, Phi_bkg, B=1024, true_depth_bin=500, pulse_fwhm=100e-12, binsize=100e-12):
    ''' We assume bin size of 100ps
    Note that FWHM = 2.355 sigma for a Gaussian shaped pulse
    '''

    # generate Gaussian pulse shape
    pulse_sigma = pulse_fwhm/2.355
    nbins = np.int(np.ceil(3*pulse_sigma/binsize))
    r_pulse = np.zeros(nbins) # this stores one half side of the pulse, pulse is 2*nbins-1 bins wide
    for i in range(nbins):
        if i==0:
            bin_area = norm.cdf(binsize/2, loc=0, scale=pulse_sigma)-norm.cdf(-binsize/2, loc=0, scale=pulse_sigma)
        else:
            bin_area = norm.cdf(i*binsize+binsize/2, loc=0, scale=pulse_sigma) - norm.cdf((i-1)*binsize+binsize/2, loc=0, scale=pulse_sigma)

        r_pulse[i] = Phi_sig * bin_area
    
    r_i = np.ones(B) * Phi_bkg
    r_i[true_depth_bin : true_depth_bin+nbins] = r_i[true_depth_bin : true_depth_bin+nbins] + r_pulse
    r_i[true_depth_bin-nbins+1:true_depth_bin] = r_i[true_depth_bin-nbins+1:true_depth_bin] + r_pulse[-1:0:-1]

    return r_i


def depth_to_waveform(depth_image,SPAD_resolution=1024,decay=100):
    from sklearn.preprocessing import KBinsDiscretizer

    '''
    Transform each pixel into a simulator input
    '''
    h,w = np.shape(depth_image)
    B = SPAD_resolution

    depth_vec = depth_image.flatten()
    # min_vec = min(depth_vec)
    # max_vec = max(depth_vec)
    min_nyu = 0
    max_nyu = 10
    signal_vec = []
    signal_bins = []
    bins = np.arange(B)
    bin_vals = np.linspace(min_nyu,max_nyu,num=B)
    
    # Find bins based on depth
    diff = np.subtract.outer(depth_vec,bin_vals)
    bin_indx = np.abs(diff).argmin(axis=1)
    depth_to_bin = bins[bin_indx]
    print(np.shape(depth_to_bin))
    # depth_to_bin = depth_to_bin.astype('int')
    for i in range(len(depth_vec)):
        # depth = depth_vec[i]
        # depth_to_bin = B*(depth-min_vec)/(max_vec-min_vec)
        # depth_to_bin = B*(depth-min_nyu)/(max_nyu-min_nyu)

        # depth_to_bin = np.round(depth_to_bin).astype('int16')
        # depth_to_bin = signal_bins[i].astype('int16')
        signal = 0.5*np.roll(np.exp(-(np.arange(B))/decay),depth_to_bin[i]) # Coeff (1/(1+depth_to_bin))
        signal_vec.append(signal)
        signal_bins.append(depth_to_bin[i])
    return signal_vec,signal_bins

def load_nyu(path):
    # Load NYU from file, restructure rgb from (3,N,M) to (N,M,3)
    dset = h5py.File(path,'r')
    list(dset.keys())
    rgb = dset[('rgb')][()]
    rgb = rgb.transpose((-2,-1,-3))
    depth = dset[('depth')][()]
    [h,w] = np.shape(depth)
    return(rgb,depth,h,w)

def loadModel():
    #load in model
    conf = get_config("zoedepth_nk", "infer",dataset='nyu')
    # conf = get_config("zoedepth_nk", "infer", dataset='kitti')
    model_zoe_nk = build_model(conf)

    #set device to cuda
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)
    return(zoe)

def close_model(model):
    del model
    import gc         # garbage collect library
    gc.collect()
    torch.cuda.empty_cache() 

def monoDepth(model,rgb):
    '''
    Using Zoedepth estimate depth
    '''
    rgb = Image.fromarray(rgb)
    raw_depth = model.infer_pil(rgb)  # as numpy
    colorized_depth = colorize(raw_depth)

    return(raw_depth,colorized_depth)

def create_gs(histogram):
    image = np.sum(np.squeeze(histogram),axis=-1).astype(np.uint8())
    return(image)

"""
ERROR METRICS

"""

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg
    
class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = torch.from_numpy(pred)
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif len(pred.shape) == 3:
            pred = pred.unsqueeze(0)
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        pred = pred.squeeze().cpu().numpy()

    pred = pred.squeeze()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])




class optSelect:
    def __init__(self, image):
        self.image = image

    def objective_function(self, coords):
        coords = coords.reshape((-1, 2)).astype(int)
        pixel_values = self.image[coords[:, 0], coords[:, 1]]

        spatial_variation = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        pixel_variation = np.sum(np.abs(np.diff(pixel_values)))

        return -(spatial_variation + pixel_variation)

    def optimize_pixel_locations(self, N):
        height, width = self.image.shape[:2]
        kmeans = KMeans(n_clusters=N, random_state=666)

        coords_flat = np.column_stack(np.unravel_index(np.arange(height * width), (height, width)))
        kmeans.fit(coords_flat)
        initial_coords = kmeans.cluster_centers_
        initial_coords_flat = initial_coords.flatten()

        result = minimize(self.objective_function, initial_coords_flat, method='L-BFGS-B')

        optimized_coords = result.x.reshape((-1, 2)).astype(int)

        return optimized_coords

def local_scale(gt,pred,n_pts):
    opt = optSelect(gt)
    coords = opt.optimize_pixel_locations(n_pts)
    gt_values = gt[coords[:,0],coords[:,1]]
    pred_values = pred[coords[:,0],coords[:,1]]

    poly_coef = np.polyfit(pred_values,gt_values,2)
    curve = np.poly1d(poly_coef)

    return(curve)
