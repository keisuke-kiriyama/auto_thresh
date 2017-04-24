import cv2
import numpy as np


#29,71,68

h = 29
s = 71
v = 68
size = 150, 150, 3
contours = np.array([[0, 0], [0, 150], [150, 150], [150, 0]])
color_img = np.zeros(size, dtype=np.uint8)
cv2.fillPoly(color_img, pts=[contours], color=(h, s, v))
img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR)
cv2.imshow("test", img)
cv2.waitKey()