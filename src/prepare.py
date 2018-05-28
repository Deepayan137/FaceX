import os
import requests
from PIL import Image
import pdb
from tqdm import *

data_path = '/home/saurabh/Facex/data/files'

class NotFound(BaseException):
	pass
def gmkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def clean(base_name):
		return base_name.split('.')[0]


def read_file(path):
	def get_(line):
		id_, url, x1, y1, x2, y2, _, _ , _ = line.split(' ')
		return id_, url, [x1, y1, x2, y2]
	with open(path, 'r') as in_file:
		lines = in_file.readlines()
	
	return [get_(x) for x in lines]

def get_image(key, items):
	dir_path = os.path.join(os.path.dirname(data_path), 'faces') + '/' + key
	gmkdir(dir_path)
	im_cropped = None
	def crop_image(item):
		
		id_, url, bbox = item
		x1, y1, x2, y2 = list(map(float, bbox))

		try:
			response = requests.get(url, stream=True)
			response.raw.decode_content = True
			im = Image.open(response.raw)
			im_cropped = im.crop((x1, y1, x2, y2))
			im_name = '%s'%(id_)
			return im_cropped, im_name
		except NotFound:
			print('404')
		except Exception as e:
			print('os err')
			print(e)
		
	for i, item in enumerate(items):
		if len(os.listdir(dir_path)) >= 10:
			print('Finished Downloading')
			break
		# pdb.set_trace()
		image, image_name = crop_image(item)
		if image:
			try:
				image.save(os.path.join(dir_path,'%s.jpg'%image_name))
			except Exception as e:
				print(e)

				

def parse_data():
	content = {}
	dirs = lambda f: data_path +'/'+ f
	files = os.listdir(data_path)
	paths = map(dirs, files)
	for path in paths:
		content[clean(os.path.basename(path))] = read_file(path)
	
	keys = [key for key in content.keys()]
	pbar = tqdm(keys[:10])
	for key in pbar:
		pbar.set_description("Processing %s" % key)
		get_image(key, content[key])
	
parse_data()