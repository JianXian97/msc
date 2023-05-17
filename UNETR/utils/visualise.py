import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ground_truth = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\BTCV\labelsTr\label0038.nii.gz"
prediction = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\pred_img0038.nii.gz"

ground = nib.load(ground_truth).get_fdata()
pred = nib.load(prediction).get_fdata()


fig, ax = plt.subplots(1,2)

 
ax[0].imshow(ground[:,:,70])
ax[1].imshow(pred[:,:,70])
plt.show()