import numpy as np
import matplotlib.pyplot as plt

# Create grayscale image
image = np.zeros((300, 300), dtype=np.uint8)

# Add objects
y, x = np.ogrid[:300, :300]

# Circle
circle = (x - 100) ** 2 + (y - 150) ** 2 <= 60 ** 2
image[circle] = 200

# Rectangle
image[80:220, 170:260] = 220

# Thresholding
threshold = np.zeros_like(image)
threshold[image > 100] = 255

# Morphological operations using NumPy
def erosion(img):
    result = np.zeros_like(img)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            region = img[i-1:i+2, j-1:j+2]
            if np.all(region == 255):
                result[i, j] = 255

    return result


def dilation(img):
    result = np.zeros_like(img)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            region = img[i-1:i+2, j-1:j+2]
            if np.any(region == 255):
                result[i, j] = 255

    return result


# Opening = Erosion followed by Dilation
eroded = erosion(threshold)
opening = dilation(eroded)

# Closing = Dilation followed by Erosion
dilated = dilation(threshold)
closing = erosion(dilated)

# Display images
plt.figure(figsize=(10, 7))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(threshold, cmap="gray")
plt.title("Thresholding")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(eroded, cmap="gray")
plt.title("Erosion")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap="gray")
plt.title("Opening")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(dilated, cmap="gray")
plt.title("Dilation")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(closing, cmap="gray")
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()
