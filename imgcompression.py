import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image and convert to grayscale
image_gray = mpimg.imread("data/cameraman.png")  # replace with your image file

# Perform SVD
U, S, VT = np.linalg.svd(image_gray)

# Function to reconstruct the image using k singular values
def svd_approx(U, S, VT, k):
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Ranks to visualize
ranks = [5, 20, 50, 100]

# Plot original and approximations
plt.figure(figsize=(14, 6))
plt.subplot(1, len(ranks) + 1, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Original")
plt.axis("off")

for i, k in enumerate(ranks):
    approx_img = svd_approx(U, S, VT, k)
    plt.subplot(1, len(ranks) + 1, i + 2)
    plt.imshow(approx_img, cmap='gray')
    plt.title(f"Rank {k}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("imgcompression.png")
plt.show()
