import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colours
import matplotlib
from matplotlib import colors
import monai 
from itkwidgets import view


def padding(array, hh, ww):
    """
    :param array: numpy array
    :param hh: desired height
    :param ww: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (hh - h) // 2
    aa = hh - a - h

    b = (ww - w) // 2
    bb = ww - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


labels = ['background', 'spleen', 'rkid', 'lkid', 'gall', 'eso', 'liver', 'sto', 'aorta', 'IVC', 'veins', 'pancreas', 'rad', 'lad',""]

fig, ax = plt.subplots(2,5)


cmap = plt.cm.jet   
cmaplist = [cmap(i) for i in range(cmap.N)][::-1]
cmaplist[0] = (0, 0, 0, 1)
cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
 
bounds = np.array(range(15))
norm = colors.BoundaryNorm(bounds, cmap.N)

max_height = 0
max_width = 0
for i,j in enumerate([35,36,37,38,39]):
    ground_truth = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\ground_img00" + str(j) + ".nii.gz"
    prediction = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\pred_img00" + str(j) + ".nii.gz"
     
    ground = nib.load(ground_truth).get_fdata()
    pred = nib.load(prediction).get_fdata()
    
    if pred.shape[0] > max_height:
        max_height = pred.shape[0]
    if pred.shape[1] > max_width:
        max_width = pred.shape[1]
 
for i,j in enumerate([35,36,37,38,39]):
    ground_truth = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\ground_img00" + str(j) + ".nii.gz"
    prediction = r"C:\Users\sooji\OneDrive - Imperial College London\Imperial\MSc Project\predictions\pred_img00" + str(j) + ".nii.gz"
     
    ground = nib.load(ground_truth).get_fdata()
    pred = nib.load(prediction).get_fdata()
    
    ground = padding(ground[:,:,200], max_height, max_width)
    pred = padding(pred[:,:,200], max_height, max_width)
    
    ax[0,i].imshow(ground, cmap=cmap, norm=norm, interpolation='nearest')
    ax[1,i].imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
    
    if i == 0:        
        ax[0,i].set_ylabel('Ground Truths', fontsize=18)
        ax[1,i].set_ylabel('Predictions', fontsize=18)
    
    ax[0,i].set_title('Test Image ' + str(i + 1), fontsize=18)
    ax[1,i].set_title('Test Image ' + str(i + 1), fontsize=18)
    
    ax[0,i].xaxis.set_tick_params(labelbottom=False)
    ax[0,i].yaxis.set_tick_params(labelleft=False)
    ax[1,i].xaxis.set_tick_params(labelbottom=False)
    ax[1,i].yaxis.set_tick_params(labelleft=False)
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])
    
    box = ax[0,i].get_position()
    box.x0 -= 0.1
    ax[0,i].set_position(box)
    box = ax[1,i].get_position()
    box.x0 -= 0.1
    ax[1,i].set_position(box)

ax2 = fig.add_axes([0.87, 0.1, 0.03, 0.8])

cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
ticks = bounds.astype(float)
ticks[:-1] += 0.5
cb.set_ticks(ticks)
 

cb.ax.set_yticklabels(labels, fontsize=18)

plt.show()


# monai.visualize.utils.blend_images(img, ground, alpha=0.5, cmap='hsv', rescale_arrays=True, transparent_background=True) 
