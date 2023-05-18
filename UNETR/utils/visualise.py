import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import monai 
from itkwidgets import view

ground_truth = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\BTCV\labelsTr\label0038.nii.gz"
image = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\BTCV\imagesTr\img0038.nii.gz"
prediction = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\pred_img0035.nii.gz"
 
img = nib.load(image).get_fdata()
ground = np.expand_dims(nib.load(ground_truth).get_fdata(), 0)
pred = nib.load(prediction).get_fdata()


view(img)

# fig, ax = plt.subplots(1,2)

 
# ax[0].imshow(ground[:,:,70])
# ax[1].imshow(pred[:,:,70])
# plt.show()


monai.visualize.utils.blend_images(img, ground, alpha=0.5, cmap='hsv', rescale_arrays=True, transparent_background=True) 
