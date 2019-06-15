import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn

import numpy as np
import time
import argparse
import utils
import torchvision
from torchvision import datasets, models, transforms
from glob import glob
import apex.amp as amp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import apex.amp as amp

import io
from PIL import Image
import tensorflow as tf

data_dir = "data/"
default_prediction = "8815 -100"
def predict(model, dataloaders, dataset_sizes, use_gpu, num_epochs=25, mixup = False, alpha = 0.1):
	print("MIXUP".format(mixup))
	since = time.time()

	best_acc = 0.0

	for phase in ['test']:
		model.train(False)  # Set model to evaluate mode

		running_loss = 0.0
		running_corrects = 0

		epoch_preds = []
		epoch_confs = []
		epoch_labels = []
		epoch_filenames = []

		# Iterate over data.
		for i, data in enumerate(tqdm(dataloaders[phase])):

			inputs, labels, filenames = data
			epoch_filenames.extend(filenames)
			# wrap them in Variable
			if use_gpu:
				inputs = Variable(inputs.cuda())
				labels = Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)

			# forward
			outputs = model(inputs)
			if type(outputs) == tuple:
				outputs, _ = outputs
			confs, preds = torch.max(outputs.data, 1)

			epoch_preds.extend(preds)
			epoch_confs.extend(confs)
			epoch_labels.extend(labels)

	
			running_corrects += torch.sum(preds == labels.data)


		epoch_loss = running_loss / dataset_sizes[phase]
		epoch_acc = running_corrects / dataset_sizes[phase]

		# epoch_gap = GAP_vector(epoch_preds, epoch_confs, epoch_labels)

		print('{} Loss: {:.4f} Acc: {:.4f}'.format(
			phase, epoch_loss, epoch_acc))


	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	return epoch_confs, epoch_preds, epoch_filenames

class ImageFolderWithPaths(datasets.ImageFolder):
	"""Custom dataset that includes image file paths. Extends
	torchvision.datasets.ImageFolder
	"""

	# override the __getitem__ method. this is the method dataloader calls
	def __getitem__(self, index):
		# this is what ImageFolder normally returns 
		original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
		# the image file path
		path = self.imgs[index][0]
		# make a new tuple that includes original and the path
		tuple_with_path = (original_tuple + (path,))
		return tuple_with_path

def make_predictions(save_file, checkpoint_name):
	input_shape = 299

	batch_size = 16
	mean = [0.5, 0.5, 0.5]
	std = [0.5, 0.5, 0.5]
	scale = 299
	use_parallel = False
	use_gpu = True
	epochs = 100

	epoch = 0

	model_conv = torchvision.models.resnet101(pretrained="imagenet").cuda()
	n_class = len(glob(data_dir + "train/*"))

	# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
	num_ftrs = model_conv.fc.in_features
	# model_conv.fc = nn.Linear(num_ftrs, n_class).cuda()

	model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, n_class),
							 # nn.ReLU(),
							 # nn.Dropout(0.2),
							 # nn.Linear(512, n_class),
							 nn.LogSoftmax(dim=1)).cuda()

	checkpoint = torch.load(checkpoint_name)
	model_conv.load_state_dict(checkpoint["model_state_dict"])


	data_transforms = {
			'test': transforms.Compose([
			transforms.CenterCrop(input_shape),
			transforms.Resize(scale),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)]),
	}

	image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
									  data_transforms[x]) for x in ['test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
											 shuffle=True, num_workers=8) for x in ['test']}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

	if use_parallel:
		print("[Using all the available GPUs]")
		model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])


	model_conv = amp.initialize(model_conv, opt_level="O1")


	print("[Training the model begun ....]")
	# model_ft = utils.train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, use_gpu,
	# 				 num_epochs=epochs, checkpoint_path=checkpoint_path, epoch_start=int(epoch), global_step=int(global_step))

	epoch_confs, epoch_preds, filenames = predict(model_conv, dataloaders, dataset_sizes, use_gpu)
	prediction_ids = [f.split("/")[-1].split(".")[0] for f in filenames]
	print(len(epoch_confs), len(epoch_preds), len(filenames))
	print(epoch_confs[:10])
	print(epoch_preds[:10])
	print(prediction_ids[:10])

	test_csv = pd.read_csv("/hdd/kaggle/landmarks/csv/recognition_sample_submission.csv")
	test_ids = test_csv["id"].tolist()

	num_no_images = 0

	with open(save_file, "w") as w:
		print("Writing predictions")
		w.write("id,landmarks\n")
		for test_id in tqdm(test_ids):
			if test_id in prediction_ids:
				idx = prediction_ids.index(test_id)
				pred = epoch_preds[idx]
				conf = epoch_confs[idx]

				w.write("{},{} {}\n".format(test_id, pred, conf))
			else:
				num_no_images += 1
				w.write("{},{}\n".format(test_id, default_prediction))


	print("{} images not found of {}. Percent: {:.3f}".format(num_no_images, len(test_ids), num_no_images / len(test_ids)))



if __name__ == "__main__":
	make_predictions(save_file="predictions/initial_resnet_101_no_freeze_checkpoint_3.csv", checkpoint_name="checkpoints_299_50_no_freeze/checkpoints_3")
