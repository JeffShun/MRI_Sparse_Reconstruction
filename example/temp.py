
import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np

# Read the PNG image
image_path = r'C:\Users\Administrator\Desktop\a.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform Haar wavelet decomposition
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# Create a figure and display original and decomposed images
# Original Image
plt.figure(1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.figure(2)
# Approximation Coefficient (cA)
plt.subplot(2, 2, 1)
plt.imshow(cA, cmap='gray')
plt.title('Approximation Coefficient (cA)')

# Horizontal Detail Coefficient (cH)
plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title('Horizontal Detail Coefficient (cH)')

# Vertical Detail Coefficient (cV)
plt.subplot(2, 2, 3)
plt.imshow(cV, cmap='gray')
plt.title('Vertical Detail Coefficient (cV)')

# Diagonal Detail Coefficient (cD)
plt.subplot(2, 2, 4)
plt.imshow(cD, cmap='gray')
plt.title('Diagonal Detail Coefficient (cD)')

plt.tight_layout()
plt.show()

