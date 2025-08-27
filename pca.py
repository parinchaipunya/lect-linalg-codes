import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate and center data
np.random.seed(1)
n_points = 300
mean_3d = np.array([0, 0, 0])
cov_3d = np.array([[3, 2.5, 2],
                   [2.5, 3, 2.5],
                   [2, 2.5, 3]])
data_3d = np.random.multivariate_normal(mean_3d, cov_3d, n_points)
centered_data = data_3d - np.mean(data_3d, axis=0)

# Step 2: PCA
cov_matrix = np.cov(centered_data, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_matrix)
sorted_idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_idx]
eigvecs = eigvecs[:, sorted_idx]

# Normalize PC1 and PC2
pcs = eigvecs[:, :2]
pcs_normalized = pcs / np.linalg.norm(pcs, axis=0)
projected_data = centered_data @ pcs_normalized

# Step 3: Plotting
fig = plt.figure(figsize=(15, 6))
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax2d = fig.add_subplot(1, 2, 2)

# Original features and colors
features = ['X', 'Y', 'Z']
colors = ['red', 'green', 'blue']

# Plot centered 3D data
ax3d.scatter(centered_data[:, 0], centered_data[:, 1], centered_data[:, 2], alpha=0.3, color="orange")

# Plot original coordinate axes (X, Y, Z)
for i in range(3):
    ax3d.quiver(0, 0, 0, *np.eye(3)[i], color=colors[i], length=3, linewidth=2)
    ax3d.text(*(np.eye(3)[i] * 3.3), features[i], color=colors[i], fontsize=12)

# Plot principal components (PC1, PC2, PC3)
for i in range(3):
    vec = eigvecs[:, i] * np.sqrt(eigvals[i])
    ax3d.quiver(0, 0, 0, *vec, color='black' if i == 0 else 'gray', linewidth=2)
    ax3d.text(*(vec * 1.1), f'PC{i+1}', color='black' if i == 0 else 'gray', fontsize=12)

ax3d.set_title("Original 3D Data with Axes and Principal Components")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

# Plot 2D projection of data
ax2d.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.4, color='skyblue', label='Projected data')

# Plot projections of X, Y, Z axes onto PC1-PC2 plane
for i in range(3):
    loading = pcs_normalized[i] * 3
    ax2d.arrow(0, 0, loading[0], loading[1], color=colors[i],
               width=0.02, head_width=0.1, length_includes_head=True)
    ax2d.text(loading[0]*1.1, loading[1]*1.1, features[i], color=colors[i], fontsize=12)

ax2d.set_xlabel("PC1")
ax2d.set_ylabel("PC2")
ax2d.set_title("2D PCA Projection with Projected Axes")
ax2d.grid(True)
ax2d.set_aspect('equal')
ax2d.legend()

# Add space between the plots
plt.subplots_adjust(wspace=0.6)
plt.savefig("pca.png")
plt.show()

