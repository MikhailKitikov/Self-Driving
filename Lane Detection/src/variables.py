import numpy as np

# pixels to meters
YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 720

# white thresholds	
LOWER_WHITE = np.array([0, 160, 10])
UPPER_WHITE = np.array([255, 255, 255])

# Perspective points to be warped
SRC = np.float32([[690, 440], [790, 440], [560, 680], [1260, 680]])
DST = np.float32([[[200, 0], [1200, 0], [200, 710], [1200, 710]]], dtype=np.int32)

# germany
# SRC = np.float32([[600, 440], [730, 440], [50, 700], [900, 700]])
# DST = np.float32([[[200, 0], [1200, 0], [200, 710], [1200, 710]]], dtype=np.int32)
# SRC = np.float32([[600, 450], [680, 450], [300, 700], [940, 700]])
# DST = np.float32([[200, 0], [1200, 0], [200, 710], [1200, 710]])

# SRC = np.float32([[590, 440],
#                   [690, 440],
#                   [200, 640],
#                   [1000, 640]])

# # Window to be shown
# DST = np.float32([[200, 0],
#                   [1200, 0],
#                   [200, 710],
#                   [1200, 710]])

# SRC = np.float32([[380, 330], [420, 330], [270, 500], [540, 500]])
# DST = np.float32([[[200, 0], [600, 0], [200, 550], [600, 550]]], dtype=np.int32)
# SRC = np.float32([[380, 330], [420, 330], [0, 500], [800, 500]])
# DST = np.float32([[[300, 0], [500, 0], [300, 550], [500, 550]]], dtype=np.int32)

# video
SKIP_FRAMES = 110

# detections
DETECTION_COLORS = {
	1: (0, 0, 255),
	2: (128, 255, 128),
	3: (255, 0, 0),
	6: (0, 255, 0),
	8: (255, 0, 0),
	10: (128, 255, 0),
	12: (128, 255, 0),
	13: (128, 255, 0)
}
