import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary image
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank mask for the ROI
mask = np.zeros_like(img)

# Draw the contours on the mask
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Create an image where the contours are filled and the other areas are white
result = cv2.bitwise_and(img, img, mask=mask)

# Create an image where the rest of the image outside the contours is filled with white
background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

# Merge the original image's non-contour areas back to the white-filled regions
final_result = cv2.add(result, background)

# Display the result
plt.imshow(final_result, cmap='gray')
plt.title("ROI Extracted and Rest Filled")
plt.axis('off')
plt.show()
