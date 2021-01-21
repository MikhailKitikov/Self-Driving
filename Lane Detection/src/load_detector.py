import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from glob import glob
import sys
from PIL import Image, ImageDraw
import numpy as np
from PIL import Image
import cv2
import six.moves.urllib as urllib
import tarfile


MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
# MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
model_path = "./"
PATH_TO_CKPT = model_path + MODEL_NAME + '/frozen_inference_graph.pb'


def download_model():
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
		file_name = os.path.basename(file.name)
		if 'frozen_inference_graph.pb' in file_name:
			tar_file.extract(file, os.getcwd())


def load_graph():
	if not os.path.exists(PATH_TO_CKPT):
		print('loading model...')
		download_model()

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	return detection_graph
