import torch

def toNumpy(tensor):
	return tensor.data.cpu().numpy()[0]

def toTensor(array):
	return torch.Tensor(array)

def define_grid(image):
	dims = [toNumpy(item) for item in image.size()]
	rows, cols, _ = dims
	xx, yy = np.meshgrid(np.arange(rows), np.range(cols))
	return xx, yy

def gaussian(pts, **kwargs):
	mu, sigma = kwargs['mu'], kwargs['sigma']
	k = -4*np.log(2)
	z = lambda x: k*((x - mu)/float(sigma))**2
	return np.array(list(map(z, kwargs['pts'])))

def gaussian_2d(**kwargs):
	mu_x, mu_y, sigma = kwargs['mu_x'], kwargs['mu_y'], kwargs['sigma']
	gaussian_x = gaussian(kwargs['xx'], mu=mu_x, sigma=sigma)
	gaussian_y = gaussian(kwargs['yy'], mu=mu_y, sigma=sigma)
	return gaussian_x, gaussian_y

def get_gaussian_mask(image, **kwargs):
	xx, yy = define_grid(image)
	for key in kwargs:
		kwargs[key] = toNumpy(kwargs[key])
	gs_x, gs_y = gaussian_2d(xx=xx, yy=yy, kwargs)
	mask = np.exp(gs_x + gs_y)
	mask = toTensor(mask)
	return mask



mask = get_gaussian_mask(image, mu_x=mu_x, mu_y=mu_y, sigma=sigma)


