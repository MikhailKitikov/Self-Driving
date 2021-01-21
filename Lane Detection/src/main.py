import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
import numpy as np
import matplotlib.pyplot as plt
import os
# import tensorflow as tf
from glob import glob
import sys
from lane_utils import *
from detection_utils import *
from variables import *

import numpy as np
import cv2
import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# left_fit_list = []
# right_fit_list = []
# center_fit_list = []

# left_fitx_list = []
# right_fitx_list = []
# center_fitx_list = []


def run_alg(frame):
	# perspective transfrom
	# plt.imshow(frame)
	# plt.show()
	birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
	# plt.imshow(birdView)
	# plt.imshow(birdViewR)
	# plt.show()

	# process image
	# img, hls, grayscale, thresh, blur, canny = processImage(birdView)
	# imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
	# imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)

	thresh = processImageSeg(birdView)
	threshL = processImageSeg(birdViewL)
	threshR = processImageSeg(birdViewR)

	# draw histogram
	hist, leftBase, rightBase = plotHistogram(thresh)
	# plt.imshow(thresh)
	# plt.show()

	# find lines
	ploty, left_fit, right_fit, center_fit, left_fitx, right_fitx, center_fitx = slide_window_search(thresh, hist)

	# if len(left_fit_list):
	# 	left_fit = 0.1 * left_fit + 0.9 * left_fit_list[-1]
	# if len(right_fit_list):
	# 	right_fit = 0.1 * right_fit + 0.9 * right_fit_list[-1]
	# if len(center_fit_list):
	# 	center_fit = 0.1 * center_fit + 0.9 * center_fit_list[-1]

	# if len(right_fit_list):
	# 	print(np.linalg.norm(right_fit - right_fit_list[-1]))

	# if len(left_fit_list) and np.linalg.norm(left_fit - left_fit_list[-1]) > 150:
	# 	left_fit = left_fit_list[-1]
	# 	left_fitx = left_fitx_list[-1]
	# left_fit_list.append(left_fit)
	# left_fitx_list.append(left_fitx)
	# if len(right_fit_list) and np.linalg.norm(right_fit - right_fit_list[-1]) > 150:
	# 	print(np.linalg.norm(right_fit - right_fit_list[-1]))
	# 	right_fit = right_fit_list[-1]
	# 	right_fitx = right_fitx_list[-1]
	# right_fit_list.append(right_fit)
	# right_fitx_list.append(right_fitx)
	# if len(center_fit_list) and np.linalg.norm(center_fit - center_fit_list[-1]) > 150:
	# 	center_fit = center_fit_list[-1]
	# 	center_fitx = center_fitx_list[-1]
	# center_fit_list.append(center_fit)
	# center_fitx_list.append(center_fitx)

	# plt.plot(left_fit)
	# plt.show()

	draw_info = general_search(thresh, left_fit, right_fit)
	# plt.show()

	# get curvature
	curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx, center_fit)

	# draw lines
	meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info, 
		ploty, left_fitx, ploty, right_fitx, ploty, center_fitx)

	deviation, directionDev = offCenter(meanPts, frame)
	# add text
	finalImg = addText(result, curveRad, curveDir, deviation, directionDev)
	return finalImg


if __name__ == '__main__':

	# model = load_model('mobilenet_v2_unet_all_classes.hdf5')

	reader = cv2.VideoCapture('videos/video.mp4')
	writer = cv2.VideoWriter('videos/output1.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (1280, 720))

	# # define detector
	detector = Detector()
	frame_count = 0

	while True:

		# read frame
		ret, frame = reader.read()
		if not ret: 
			break

		frame_count += 1
		if frame_count < SKIP_FRAMES:
			continue

		if frame_count > 190:
			break

		finalImg = run_alg(frame)

		# detect cars
		image_np = finalImg
		image = image_np[...]
		boxes, classes, scores, masks = detector.detect_multi_object(frame, threshold=0.3)

		vehicle_boxes_ind = []
		for i in range(len(boxes)):
			if classes[i] in [3, 8]:
				vehicle_boxes_ind.append(i)

		ok_boxes = [i for i in range(len(boxes)) if i not in vehicle_boxes_ind]
		sort_ind = np.argsort(scores[vehicle_boxes_ind])[::-1]
		mark = [True for i in range(len(sort_ind))]
		for i in range(len(sort_ind)):
			ax1, ay1, ax2, ay2 = boxes[vehicle_boxes_ind[sort_ind[i]]]
			acx = (ax1 + ax2) / 2
			acy = (ay1 + ay2) / 2
			if not mark[sort_ind[i]]:
				continue
			for j in range(i + 1, len(sort_ind)):
				bx1, by1, bx2, by2 = boxes[vehicle_boxes_ind[sort_ind[j]]]
				bcx = (bx1 + bx2) / 2
				bcy = (by1 + by2) / 2
				if (acx - bcx)**2 + (acy - bcy)**2 < 100:
					mark[sort_ind[j]] = False
		
		good = [vehicle_boxes_ind[i] for i in range(len(mark)) if mark[i]]
		ok_boxes.extend(good)
		boxes, classes, scores, masks = boxes[ok_boxes], classes[ok_boxes], scores[ok_boxes], masks[ok_boxes]
		image = visualize_detections(image, image_np, boxes, classes, scores, masks)

		cv2.imshow('image', image)


		##### via segmentation ########

		# IMAGE_SIZE = 224
		# img = cv2.resize(frame, (IMAGE_SIZE*2, IMAGE_SIZE*2))
		# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# pred = model.predict(np.expand_dims(preprocess_input(img), axis=0))[0]
		# mask = (((pred.argmax(axis=2)==6) > 0.5)*255).astype('uint8')
		# image = mask #pred.argmax(axis=2)

		# display
		# cv2.imshow("Final", image)
		# plt.imshow(pred.argmax(axis=2))
		# plt.show()

		# write
		image = cv2.flip(image, 0)
		writer.write(image)

		# wait for key
		if cv2.waitKey(10) == ord('q'):
			break

	# clear

	reader.release()
	writer.release()
	cv2.destroyAllWindows()

	# seg = cv2.imread('seg.png')
	# finalImg = run_alg(seg)
	# plt.imshow(finalImg)
	# plt.show()

	# seg = cv2.imread('seg.png')
	# # seg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# segfl = seg.reshape(seg.shape[0] * seg.shape[1], 3)
	# line_mask = np.zeros((seg.shape[0] * seg.shape[1]))
	# # side_mask = np.zeros((seg.shape[0] * seg.shape[1]))
	# for i in range(len(segfl)):
	# 	line_mask[i] = (segfl[i] == np.array([50, 234, 157])).all()
	# 	# line_mask[i] = (segfl[i] == np.array([0, 127, 255])).all()

	# line_mask = line_mask.reshape(seg.shape[:2]) * 255
	# line_mask = np.stack([line_mask, line_mask, line_mask], axis=2).astype('uint8')
	# # plt.imshow(line_mask)
	# # plt.show()
	# finalImg = run_alg(line_mask)
	# plt.imshow(finalImg)
	# plt.show()
