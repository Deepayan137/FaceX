import os
import torch
import numpy as np
import pdb
from PIL import Image
import cv2
import models.VGG_FACE
# import torchvision.models as models

model = models.VGG_FACE.VGG_FACE
model.load_state_dict(torch.load('models/VGG_FACE.pth'))

path = 'data/'
def evaluate(image):
	model.eval()
	out = model(image)
	_, predicted = torch.max(out.data, 1)
	return predicted.data.cpu().numpy()[0]
def toTensor(image):
	return torch.Tensor(image)

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


if __name__ == '__main__':
	filepath = os.path.join(path, 'files')
	lookup_file = 'names.txt'
	classes = get_class_names(lookup_file)
	name_to_idx = get_idx(classes)
	idx_to_name = dict(zip(name_to_idx.values(), name_to_idx.keys()))
	image_file = 'data/faces/Adam_Hicks/9.jpg'
	im = cv2.imread(image_file)
	dim = (224, 224)
	im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
	im = np.array(im).astype(np.float32)
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	im = toTensor(im)
	predicted = evaluate(im)
	print(idx_to_name[int(predicted)])
	



