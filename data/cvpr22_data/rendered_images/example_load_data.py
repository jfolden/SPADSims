'''
Example script that loads the data in this folder.
You need to install numpy and matplotlib to run this script
'''
## Standard Library Imports
import glob

## Library Imports
import numpy as np
import matplotlib.pyplot as plt

## Local Imports

def get_unique_scene_ids( transient_dirpath, render_params_str ):
	unique_scene_filepaths = glob.glob('{}/*{}_view-0*'.format(transient_dirpath, render_params_str))
	scene_ids = []
	for i in range(len(unique_scene_filepaths)):
		scene_filename = unique_scene_filepaths[i].split('/')[-1]
		scene_ids.append(scene_filename.split('_nr')[0])
	return scene_ids

def clip_rgb(img, max_dynamic_range=1e4):
	epsilon = 1e-7
	min_img_val = np.min(img)
	new_max_img_val = max_dynamic_range *  (min_img_val + epsilon)
	# Clip all pixels with intensities larger than the max dynamic range
	img[img > new_max_img_val] = new_max_img_val
	return img
	
def gamma_compress(img, gamma_factor=1./2.2): return np.power(img, gamma_factor)

def simple_tonemap(rgb_img):
	rgb_img = clip_rgb(rgb_img)
	rgb_img = gamma_compress(rgb_img)
	return rgb_img

data_dirpath = '.'
transient_data_dirpath = '{}/transient_images'.format(data_dirpath)
rgb_data_dirpath = '{}/rgb_images'.format(data_dirpath)
gt_depths_data_dirpath = '{}/ground_truth_depthmaps'.format(data_dirpath)

# Render parameters
n_rows = 240
n_cols = 320
n_tbins = 2000
n_samples = 2048

render_params_str = 'nr-{}_nc-{}_nt-{}_samples-{}'.format(n_rows, n_cols, n_tbins, n_samples)

# Get unique scene ids
unique_scene_ids = get_unique_scene_ids(transient_data_dirpath, render_params_str)
print('N Unique Scene IDs: {}'.format(len(unique_scene_ids)))
print('Scene IDs Available: {}'.format(unique_scene_ids))

scene_id = 'bathroom-cycles-2'


for scene_id in unique_scene_ids:
	scene_filename = '{}_{}_view-0'.format(scene_id, render_params_str)
	print("Loading: {}".format(scene_filename))
	gt_depths_img = np.load('{}/{}.npy'.format(gt_depths_data_dirpath, scene_filename))
	rgb_img = np.load('{}/{}.npy'.format(rgb_data_dirpath, scene_filename))
	transient_img = np.load('{}/{}.npz'.format(transient_data_dirpath, scene_filename))['arr_0']

	print("GT Depths Shape {}".format(gt_depths_img.shape))
	print("RGB Image Shape {}".format(rgb_img.shape))
	print("Transient Image Shape {}".format(transient_img.shape))


	rgb_img = rgb_img / np.max(rgb_img)
	rgb_img = simple_tonemap(rgb_img)
	plt.clf()
	plt.imshow(rgb_img)
	plt.show()
	plt.pause(1.0)