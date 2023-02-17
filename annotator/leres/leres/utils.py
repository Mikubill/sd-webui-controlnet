def scale_torch(img):
	"""
	Scale the image and output it in torch.tensor.
	:param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
	:param scale: the scale factor. float
	:return: img. [C, H, W]
	"""
	if len(img.shape) == 2:
		img = img[np.newaxis, :, :]
	if img.shape[2] == 3:
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
		img = transform(img.astype(np.float32))
	else:
		img = img.astype(np.float32)
		img = torch.from_numpy(img)
	return img

def estimateleres(img, model, w, h):
	# leres transform input
	rgb_c = img[:, :, ::-1].copy()
	A_resize = cv2.resize(rgb_c, (w, h))
	img_torch = scale_torch(A_resize)[None, :, :, :] 
	
	# compute
	with torch.no_grad():
		img_torch = img_torch.cuda()
		prediction = model.depth_model(img_torch)

	prediction = prediction.squeeze().cpu().numpy()
	prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

	return prediction