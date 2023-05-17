import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\datasets\BTCV\labelsTr\label0038.nii.gz"

test_load = nib.load(path).get_fdata()
test = test_load[:,:,59]
plt.imshow(test)
plt.show()