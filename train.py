import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import argparse
import utils
import torchvision
from torchvision import datasets, models, transforms
from glob import glob
import apex.amp as amp

data_dir = "data_299/"

def train(checkpoint_path="checkpoints/", checkpoint_name=None):
	freeze_layers = False
	input_shape = 299
	# batch_size = 256 # For resnet101 on 2070
	# batch_size = 16 # For resnet101 on 2070
	batch_size = 200
	mean = [0.5, 0.5, 0.5]
	std = [0.5, 0.5, 0.5]
	scale = 299
	use_parallel = False
	use_gpu = True
	epochs = 100

	epoch = 0
	global_step = 0
	# model_conv = torchvision.models.resnet50(pretrained="imagenet").cuda()
	# model_conv = torchvision.models.resnet101(pretrained="imagenet").cuda()
	model_conv = torchvision.models.resnet18(pretrained="imagenet").cuda()


		# Stage-1 Freezing all the layers 
	# if freeze_layers:
	# 	for i, param in model_conv.named_parameters():
	# 		param.requires_grad = False

	n_class = len(glob(data_dir + "train/*"))

	# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
	num_ftrs = model_conv.fc.in_features
	model_conv.fc = nn.Linear(num_ftrs, n_class).cuda()

	# model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
#                                # nn.ReLU(),
#                                # nn.Dropout(0.2),
#                                # nn.Linear(512, n_class),
#                                nn.LogSoftmax(dim=1)).cuda()

	# model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, n_class),
	# 						 # nn.ReLU(),
	# 						 # nn.Dropout(0.2),
	# 						 # nn.Linear(512, n_class),
	# 						 nn.LogSoftmax(dim=1)).cuda()


	if checkpoint_name != None:
		checkpoint = torch.load(checkpoint_path + checkpoint_name)
		model_conv.load_state_dict(checkpoint["model_state_dict"])
		epoch = int(checkpoint["epoch"] + 1)
		global_step = int(checkpoint["global_step"])

	data_transforms = {
			'train': transforms.Compose([
			transforms.CenterCrop(input_shape),
			transforms.Resize(scale),
			transforms.RandomResizedCrop(input_shape),
			transforms.RandomHorizontalFlip(),
	#         transforms.RandomVerticalFlip(),
			transforms.ColorJitter(hue=.05, saturation=.05, brightness=.15, contrast=.05),
			transforms.RandomRotation(degrees=90),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)]),
			'val': transforms.Compose([
			transforms.CenterCrop(input_shape),
			transforms.Resize(scale),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)]),
	}

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
									  data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
											 shuffle=True, num_workers=8) for x in ['train', 'val']}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	if use_parallel:
		print("[Using all the available GPUs]")
		model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])

	print("[Using CrossEntropyLoss...]")
	criterion = nn.CrossEntropyLoss()

	print("[Using small learning rate with momentum...]")
	optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)
	if checkpoint_name != None:
		optimizer_conv.load_state_dict(checkpoint["optimizer_state_dict"])


	model_conv, optimizer_conv = amp.initialize(model_conv, optimizer_conv, opt_level="O1")
	print("[Creating Learning rate scheduler...]")
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

	print("[Training the model begun ....]")
	model_ft = utils.train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu,
					 num_epochs=epochs, checkpoint_path=checkpoint_path, epoch_start=int(epoch), global_step=int(global_step))


if __name__ == "__main__":
	train(checkpoint_path="checkpoints_299_18_no_freeze/", checkpoint_name=None)
