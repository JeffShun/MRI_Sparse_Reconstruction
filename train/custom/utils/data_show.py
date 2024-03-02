import numpy as np
import matplotlib.pylab as plt
import h5py

sample = r"E:\ShenFile\Code\MRI_Sparse_Reconstruction\MRI_Sparse_Reconstruction_SenseMap\example\data\input\test\1000267_22.h5"

with h5py.File(sample, 'r') as f:
    full_sampling_img = f['full_sampling_img'][:]                # 320*320 -complex64
    full_sampling_kspace = f['full_sampling_kspace'][:]          # 15*320*320 -complex64
    random_sample_img = f['random_sample_img'][:]                # 320*320 -complex64
    random_sample_mask = f['random_sample_mask'][:]              # 320*320 -int
    sensemap = f['sensemap'][:]                                  # 15*320*320 -complex64
plt.figure(1)
plt.subplot(1,2,1)  
plt.imshow(np.abs(full_sampling_img),cmap="gray") 
plt.subplot(1,2,2)
plt.imshow(np.abs(random_sample_img),cmap="gray") 

for i in range(4,8):
    plt.figure(i+2)
    full_sampling_kspace_i = full_sampling_kspace[i]
    plt.subplot(1,2,1)
    plt.imshow(np.abs(full_sampling_kspace_i)>0.1)

    sensemap_i = sensemap[i]
    plt.subplot(1,2,2)
    plt.imshow(np.abs(sensemap_i))   
    
plt.show()