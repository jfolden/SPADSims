import numpy as np
import numpy.polynomial.polynomial as poly
import helpers
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


fnames = ["C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\dining_room_0019\\00216.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\office_kitchen_0003\\00591.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0019\\00451.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\furniture_store_0001e\\00076.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\kitchen_0028a\\00406.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\living_room_0078\\00926.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\study_room_0005b\\00661.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\dining_room_0037\\01011.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\bedroom_0124\\00151.h5",
          "C:\\Users\\Justin\\tensorflow_datasets\\downloads\\extracted\\NYUv2\\nyudepthv2\\train\\basement_0001a\\00021.h5"]
rgbs = []
depths = []
Pred_depths = []


monomodel = helpers.loadModel()





for i in range(len(fnames)):
    [rgb,depth,h,w] = helpers.load_nyu(fnames[i])
    rgbs.append(rgb)
    depths.append(depth)
    [pred_depth,pred_depth_col] = helpers.monoDepth(monomodel,rgb)
    Pred_depths.append(pred_depth)
    


# plt.title("Monocular")
# plt.subplot(1,2,1)
# plt.imshow(depth)

# plt.subplot(1,2,2)
# plt.imshow(pred_depth)
# plt.show()
# input()


sorted_depths = []
sorted_norm_depths = []
depth_values = []
Pred_depth_values = []
norm_values = []

samples = np.linspace(0,(h*w)-1,num=200,dtype=int)
Pred_depths = np.clip((Pred_depths-np.min(Pred_depths)),a_min=0,a_max=10)

# samples = np.append(samples,(np.arange(samples[-1]-30,samples[-1]-1,step=1)))
for i in range(len(fnames)):
    d = Pred_depths[i]

    de = depths[i].flatten()

    norm = (d-np.min(Pred_depths))/(np.max(Pred_depths)-np.min(Pred_depths))
    # norm = d
    # norm = (d - np.min(Pred_depths))*((np.max(depths)-np.min(depths))/(np.max(Pred_depths)-np.min(Pred_depths)))+np.min(depths)

    d = d.flatten()
    norm = norm.flatten()
    idx = np.argsort(de)
    # sorted_depth = d[idx]
    # sorted_norm = norm[idx]
    # sorted_de = de[idx]
    sorted_depth = d
    sorted_norm = norm
    sorted_de = de
    
    Pred_depth_values = np.append(depth_values,sorted_depth)#[samples]
    depth_values = np.append(depth_values,sorted_de)#[samples]

    norm_values = np.append(norm_values,sorted_norm)#[samples]
    sorted_depths.append(sorted_depth)
    sorted_norm_depths.append(sorted_norm)

mask = np.nonzero(depth_values > 0)
depth_values = depth_values[mask]
norm_values = norm_values[mask]



poly_coef = np.polyfit(norm_values,depth_values,3)
print(np.max(norm_values))
plt.figure()
plt.scatter(norm_values,depth_values)
plt.show()
curve = np.poly1d(poly_coef)
# curve = LinearRegression().fit(norm_values.reshape(-1,1),depth_values.reshape(-1,1))
# curve.score(norm_values,depth_values)
xp = np.linspace(0,1,num=200)
# plt.plot(xp,curve.predict(np.expand_dims(xp,1)))
plt.plot(xp,curve(xp))
plt.show()
"""

"""
np.savez("poly3_gen_scale_all.npz",curve=curve,max=np.max(Pred_depths),min=np.min(Pred_depths))
# import time
# data_fovea = np.load('F:\\Research\\compressive-spad-lidar-cvpr22\\data\\nyu_results\\12_13_2023\\dining_room_0031\\00216\\np-2000.00_sbr-1.00_foveated\\Identity_ncodes-2000_rec-linear-irf_pw-1.0.npz')
# data_norm = np.load('F:\\Research\\compressive-spad-lidar-cvpr22\\data\\nyu_results\\12_13_2023\\dining_room_0031\\00216\\np-2000.00_sbr-1.00\\Identity_ncodes-2000_rec-linear-irf_pw-1.0.npz')

# c_fovea = data_fovea['c_vals']
# c_norm = data_norm['c_vals'].squeeze()

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#ax3 = fig.add_subplot(1,3,3)
ax1.set_ylim([0,1200])
ax2.set_ylim([0,1200])
line1, = ax1.plot(np.squeeze(c_norm[1,1,:]))
line2, = ax2.plot((c_fovea[1,1,:]))
for i in range(240):
    for j in range(320):
        line1.set_ydata(np.squeeze(c_norm[i,j,:]))
        line2.set_ydata(np.squeeze(c_fovea[i,j,:]))
        fig.canvas.draw()
    

        # to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(0.01)
        
# mask_fovea = 0
# fovea_sum = np.sum(c_fovea,-1)
# for i in range(fovea_sum.shape[0]):
#     for j in range(fovea_sum.shape[1]):
#         x = fovea_sum[i,j]
#         if x < 250:
#             mask_fovea += 1
