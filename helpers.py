import numpy as np
from scipy.stats import norm
import h5py

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
    #
    dset = h5py.File(path,'r')
    list(dset.keys())
    rgb = dset[('rgb')][()]
    depth = dset[('depth')][()]
    [h,w] = np.shape(depth)
    return(rgb,depth,h,w)

    
