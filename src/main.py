import os
import torch
import numpy as np
import pdb
from PIL import Image
import cv2
import models.VGG_FACE

from src.utils import apply_mask, get_gaussian_mask, KL_divergence
# import torchvision.models as models

model = models.VGG_FACE.VGG_FACE
model.load_state_dict(torch.load('models/VGG_FACE.pth'))

path = 'data/'

# print(model)
def toNumpy(tensor):
	return tensor.data.cpu().numpy()[0]

def toTensor(image):
	return torch.Tensor(image)

def predict(probs):
	_, predicted = torch.max(probs.data, 1)
	return toNumpy(predicted)

def evaluate(image, layer=None):
	k = 0
	model.eval()
	if layer is not None:
		k = layer
	probs = model[k:](image)
	return probs
	
def get_activation(image, layer=None):

	def take_transpose(activations):
		return torch.squeeze(activations).permute(1 ,2, 0)
	
	if layer is not None:
		return take_transpose(model[:layer](image))
	return take_transpose(model(image))


def get_class_names(lookup_file):
	with open(lookup_file, 'r') as in_file:
		lines = in_file.readlines()
	classes = [line.strip() for line in lines]	
	return classes

def get_idx(classes):
	name_to_idx = {}
	for i, class_ in enumerate(classes):
		name_to_idx[class_] = len(name_to_idx)
	return name_to_idx

def image_prep(image):
	im = np.array(image).astype(np.float32)
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	im = toTensor(im)
	return im

if __name__ == '__main__':
	filepath = os.path.join(path, 'files')
	lookup_file = 'names.txt'
	classes = get_class_names(lookup_file)
	name_to_idx = get_idx(classes)
	idx_to_name = dict(zip(name_to_idx.values(), name_to_idx.keys()))
	image_file = 'data/faces/Nina_Arianda/8.jpg'
	im = cv2.imread(image_file)
	dim = (224, 224)
	im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
	mx, my = im.shape[0]//2, im.shape[1]//2
	im_temp = im
	im = image_prep(im)
	probs_true = evaluate(im)
	activation = get_activation(im, 17)
	activation = activation.data.cpu().numpy()
	# print(activation.shape)
	masked_image, mask = apply_mask(activation, mx=mx, my=my, sigma=30, mask=None)
	masked_image = image_prep(masked_image)
	probs = evaluate(masked_image, layer=17)
	p_scores = {
				'probs_true': probs_true,
				'probs': probs
				}
	for key in p_scores:
		p_scores[key] = toNumpy(p_scores[key])
	# pdb.set_trace()
	print(KL_divergence(p_scores['probs_true'], p_scores['probs']))
	mask_reshaped = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
	masked_image, mask = apply_mask(im_temp, mask=mask_reshaped)
	cv2.imwrite('masked.jpg', masked_image)
	pdb.set_trace()
	# print(idx_to_name[int(new_predict)])
	# print(idx_to_name[int(predicted)])
	



