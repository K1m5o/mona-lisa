import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Mona Lisa image
url = "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
image = cv2.imread(cv2.samples.findFile(cv2.utils.getCacheDirectory() + url), cv2.IMREAD_GRAYSCALE)



# Resize for processing
image = cv2.resize(image, (400, 600))

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours from edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas to draw on
canvas = np.ones_like(image) * 255

# Draw contours
cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)

# Display the generated Mona Lisa sketch
plt.figure(figsize=(6, 9))
plt.imshow(canvas, cmap='gray')
plt.axis('off')
plt.title("Simplified Mona Lisa Sketch")
plt.show()
# mona-lisa
