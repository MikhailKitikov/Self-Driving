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
from variables import *


# process image
def processImage(inpImage):

	# Apply HLS color filtering to filter out white lane lines
	hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
	mask = cv2.inRange(inpImage, LOWER_WHITE, UPPER_WHITE)
	hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)

	# Convert image to grayscale, apply threshold, blur & extract edges
	gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
	blur = cv2.GaussianBlur(thresh,(3, 3), 0)
	canny = cv2.Canny(blur, 40, 60)

	# Display the processed images
	# cv2.imshow("Image", inpImage)
	# cv2.imshow("HLS Filtered", hls_result)
	# cv2.imshow("Grayscale", gray)
	# cv2.imshow("Thresholded", thresh)
	# cv2.imshow("Blurred", blur)
	# cv2.imshow("Canny Edges", canny)

	return inpImage, hls_result, gray, thresh, blur, canny


def processImageSeg(inpImage):

	gray = cv2.cvtColor(inpImage, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

	# plt.imshow(thresh)
	# plt.show()

	return thresh


# perspective transform
def perspectiveWarp(inpImage):

	# Get image size
	img_size = (inpImage.shape[1], inpImage.shape[0])

	# Matrix to warp the image for birdseye window
	matrix = cv2.getPerspectiveTransform(SRC, DST)
	# Inverse matrix to unwarp the image for final window
	minv = cv2.getPerspectiveTransform(DST, SRC)
	birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

	# Get the birdseye window dimensions
	height, width = birdseye.shape[:2]

	# Divide the birdseye view into 2 halves to separate left & right lanes
	birdseyeLeft  = birdseye[0:height, 0:width // 2]
	birdseyeRight = birdseye[0:height, width // 2:width]

	# Display birdseye view image
	# cv2.imshow("Birdseye" , birdseye)
	# cv2.imshow("Birdseye Left" , birdseyeLeft)
	# cv2.imshow("Birdseye Right", birdseyeRight)

	return birdseye, birdseyeLeft, birdseyeRight, minv


# histogram
def plotHistogram(inpImage):

	histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis = 0)

	midpoint = np.int(histogram.shape[0] / 2)
	leftxBase = np.argmax(histogram[:midpoint])
	rightxBase = np.argmax(histogram[midpoint:]) + midpoint

	# plt.figure(figsize=(5,3))
	# plt.grid()
	# plt.xlabel("x")
	# plt.ylabel("Принадлежность линии")

	# Return histogram and x-coordinates of left & right lanes to calculate
	# lane width in pixels
	return histogram, leftxBase, rightxBase


# sliding window search
def slide_window_search(binary_warped, histogram):

	# Find the start of left and right lane lines using histogram info
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))# * 255
	midpoint = np.int(histogram.shape[0] / 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# A total of 9 windows will be used
	nwindows = 10
	window_height = np.int(binary_warped.shape[0] / nwindows)
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	margin = 100
	minpix = 50
	left_lane_inds = []
	right_lane_inds = []

	# center_x, center_y = [], []

	#### START - Loop to iterate through windows and search for lane lines #####
	for window in range(nwindows):
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		color_left = (255,255,255) if len(good_left_inds) > 100 else (255,0,0)
		color_right = (255,255,255) if len(good_right_inds) > 100 else (255,0,0)
		# cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color_left, 1)
		# cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), color_right, 1)
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# center_x.append(((win_xleft_low + win_xleft_high) // 2 + (win_xright_low + win_xright_high) // 2) // 2)
		# center_y.append((win_y_high + win_y_low) // 2)

		# test_leftx = np.mean(nonzerox[good_left_inds]) if len(good_left_inds) > minpix else leftx_current
		# test_rightx = np.mean(nonzerox[good_right_inds]) if len(good_right_inds) > minpix else rightx_current

		# center_x.append((test_leftx + test_rightx) / 2)
		# center_y.append((win_y_low + win_y_high) / 2)

		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	#### END - Loop to iterate through windows and search for lane lines #######

	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Apply 2nd degree polynomial fit to fit curves
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	center_fit = (left_fit + right_fit) / 2
	# print(center_fit[0])

	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
	center_fitx = center_fit[0] * ploty**2 + center_fit[1] * ploty + center_fit[2]

	ltx = np.trunc(left_fitx)
	rtx = np.trunc(right_fitx)
	ctx = np.trunc(center_fitx)
	# plt.plot(right_fitx)
	# plt.show()

	# cv2.imshow('image', out_img.astype('uint8'))
	# cv2.waitKey(0)

	# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 255, 255]
	# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 255, 255]

	# plt.imshow(out_img)
	# plt.plot(left_fitx,  ploty, color = 'green', linewidth=3)
	# plt.plot(right_fitx, ploty, color = 'green', linewidth=3)
	# plt.plot(center_fitx, ploty, color = 'yellow', linestyle='dashed', linewidth=2)
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	return ploty, left_fit, right_fit, center_fit, ltx, rtx, ctx

# detect lanes
def general_search(binary_warped, left_fit, right_fit):

	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
	left_fit[1]*nonzeroy + left_fit[2] + margin)))

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
	right_fit[1]*nonzeroy + right_fit[2] + margin)))

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	## VISUALIZATION ###########################################################

	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
								  ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# plt.imshow(result)
	# plt.plot(left_fitx,  ploty, color = 'yellow')
	# plt.plot(right_fitx, ploty, color = 'yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)

	ret = {}
	ret['leftx'] = leftx
	ret['rightx'] = rightx
	ret['left_fitx'] = left_fitx
	ret['right_fitx'] = right_fitx
	ret['ploty'] = ploty

	return ret


# calc curvature
def measure_lane_curvature(ploty, leftx, rightx, center_fit):

	leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
	rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

	# Choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)

	# Fit new polynomials to x, y in world space
	left_fit_cr = np.polyfit(ploty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
	right_fit_cr = np.polyfit(ploty*YM_PER_PIX, rightx*XM_PER_PIX, 2)

	# Calculate the new radii of curvature
	left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	# print(left_curverad, 'm', right_curverad, 'm')

	# Decide if it is a left or a right curve
	# if leftx[0] - leftx[-1] > 20:
	# 	curve_direction = 'Left Curve'
	# elif leftx[-1] - leftx[0] > 20:
	# 	curve_direction = 'Right Curve'
	# else:
	# 	curve_direction = 'Straight'

	if center_fit[0] < -0.0001:
		curve_direction = 'Left Curve'
	elif center_fit[0] > 0.0001:
		curve_direction = 'Right Curve'
	else:
		curve_direction = 'Straight'

	return (left_curverad + right_curverad) / 2.0, curve_direction


# draw lines
def draw_lane_lines(original_image, warped_image, Minv, draw_info, 
	points_y_left, points_x_left, points_y_right, points_x_right, points_y_center, points_x_center):

	leftx = draw_info['leftx']
	rightx = draw_info['rightx']
	left_fitx = draw_info['left_fitx']
	right_fitx = draw_info['right_fitx']
	ploty = draw_info['ploty']

	warp_zero = np.zeros_like(warped_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	mean_x = np.mean((left_fitx, right_fitx), axis=0)
	pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

	cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 255))
	# cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 0, 0))

	points = np.array(list(zip(points_x_left, points_y_left))).astype(int)
	color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 15) 

	points = np.array(list(zip(points_x_right, points_y_right))).astype(int)
	color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 15) 

	points = np.array(list(zip(points_x_center, points_y_center))).astype(int)
	color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 10) 

	# cv2.imshow('image', color_warp)

	newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
	result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

	return pts_mean, result


# dist from center
def offCenter(meanPts, inpFrame):

	# Calculating deviation in meters
	mpts = meanPts[-1][-1][-2].astype(int)
	pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
	deviation = pixelDeviation * XM_PER_PIX
	direction = "left" if deviation < 0 else "right"

	return deviation, direction


# text
def addText(img, radius, direction, deviation, devDirection):

	img1 = np.copy(img)

	if direction == 'Straight':
		start_point = (1280 // 2, 100)
		end_point = (1280 // 2, 100 - 25)
		color = (255, 0, 0) 
		thickness = 7
		img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)
	elif direction == 'Left Curve':
		start_point = (1280 // 2, 100)
		end_point = (1280 // 2 - 25, 100 - 25)
		color = (255, 0, 0) 
		thickness = 7
		img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)
	elif direction == 'Right Curve':
		start_point = (1280 // 2, 100)
		end_point = (1280 // 2 + 25, 100 - 25)
		color = (255, 0, 0) 
		thickness = 7
		img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)

	img = cv2.addWeighted(img, 0.7, img1, 0.3, 0)

	# Add the radius and center position to the image
	# font = cv2.FONT_HERSHEY_TRIPLEX

	# if (direction != 'Straight'):
	#   text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
	#   text1 = 'Curve Direction: ' + (direction)

	# else:
	#   text = 'Radius of Curvature: ' + 'N/A'
	#   text1 = 'Curve Direction: ' + (direction)

	# cv2.putText(img, text , (50,100), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)
	# cv2.putText(img, text1, (50,150), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)

	# # Deviation
	# deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
	# cv2.putText(img, deviation_text, (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 2, cv2.LINE_AA)

	return img
