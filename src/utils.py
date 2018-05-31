import cv2
import pdb
import numpy as np 
# from matplotlib import pyplot as plt
from scipy.stats import entropy
def gaussian(**kwargs):

	return np.exp(-4*np.log(2) * ((kwargs['x'] - kwargs['mu_x'])**2 + \
				 (kwargs['y'] - kwargs['mu_y'])**2) / kwargs['sigma']**2)

def get_gaussian_mask(image, mx, my, sigma):
	"""
	The given fuction takes cordinates mx and my 
	and returns a matrix the size of image .
	The matrix is calculated using the gaussian equation

	"""

	rows, cols, channels = image.shape
	mask = np.zeros(shape=(image.shape[0], image.shape[1]))
	for i in range(rows):
		for j in range(cols):
			mask[i, j] = gaussian(x = i, y = j,
								 mu_x = mx, mu_y = my,
								 sigma = sigma)
		
	return mask

def apply_mask(image,**kwargs):
	"""takes the gaussian mask and applies it on the image"""
	mask = kwargs['mask']
	if kwargs['mask'] is None:
		mask = get_gaussian_mask(image, kwargs['mx'], kwargs['my'], kwargs['sigma'])
	for ch in range(image.shape[2]):
		image[:, :, ch] = np.multiply(image[:, :, ch], mask)
	return image, mask

def KL_divergence(probs_true, probs):
	return entropy(probs_true, probs)
# path = 'cat.jpg'
# if __name__ == '__main__':
# 	image = cv2.imread(path)
# 	plt.axis("off")
# 	# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# 	mx, my = image.shape[0]//2, image.shape[1]//2
# 	masked_image = apply_mask(image, mx, my, 300)
# 	plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
# 	plt.show() 	