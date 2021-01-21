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
from load_detector import *
from variables import *


def select_boxes(boxes, classes, scores, masks, threshold=0.3):
	boxes = np.squeeze(boxes)
	classes = np.squeeze(classes)
	scores = np.squeeze(scores)
	masks = np.squeeze(masks)
	ind = scores > threshold
	return boxes[ind], classes[ind], scores[ind], masks[ind]


class Detector:

	def __init__(self):
		self.detection_graph = load_graph()
		self.extract_graph_components()
		self.sess = tf.compat.v1.Session(graph=self.detection_graph)
		dummy_image = np.zeros((100, 100, 3))
		self.detect_multi_object(dummy_image, 0.1)
		self.traffic_light_box = None
		self.classified_index = 0

	def extract_graph_components(self):
		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		self.detection_masks = self.detection_graph.get_tensor_by_name('detection_masks:0')
		self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
	
	def detect_multi_object(self, image_np, threshold):
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(boxes, scores, classes, num, masks) = self.sess.run(
			[self.detection_boxes, self.detection_scores, self.detection_classes, 
				self.num_detections, self.detection_masks],
			feed_dict={self.image_tensor: image_np_expanded})
		boxes, classes, scores, masks = select_boxes(boxes, classes, scores, masks, threshold)
		return boxes, classes, scores, masks
		# (boxes, scores, classes, num) = self.sess.run(
		# 	[self.detection_boxes, self.detection_scores, self.detection_classes, 
		# 		self.num_detections],
		# 	feed_dict={self.image_tensor: image_np_expanded})
		# masks = boxes
		# boxes, classes, scores, masks = select_boxes(boxes, classes, scores, masks, threshold)
		# return boxes, classes, scores, masks


def crop_roi_image(image_np, sel_box):
	im_height, im_width, _ = image_np.shape
	(left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
								  sel_box[0] * im_height, sel_box[2] * im_height)
	cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
	return cropped_image


def visualize_detections(image, image_np, boxes, classes, scores, masks):

	for box, label, score, mask in zip(boxes, classes, scores, masks):

		x1, y1, x2, y2 = box
		im_height, im_width, _ = image_np.shape
		x1, x2 = int(x1 * im_height), int(x2 * im_height)
		y1, y2 = int(y1 * im_width), int(y2 * im_width)
		w, h = x2 - x1, y2 - y1

		start_point = (y1, x1) 
		end_point = (y2, x2) 
		label = int(label)
		if label not in DETECTION_COLORS:
			continue
		color = DETECTION_COLORS[label] 
		thickness = 2

		mask_img = np.zeros_like(image)
		mask = cv2.resize(mask, (h, w))
		mask = mask > 0.5
		mask = np.stack([mask * color[0], mask * color[1], mask * color[2]], 2)
		mask_img[x1: x2, y1: y2] = mask

		image = cv2.rectangle(image, start_point, end_point, color, thickness) 
		image = cv2.addWeighted(image, 1, mask_img, 0.6, 0)

	return image
