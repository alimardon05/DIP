import os
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Paths
base_dir = Path(r"c:\Users\alima\OneDrive\Desktop\DIP\DIP\Week 10\LAB_02")
train_dir = base_dir / "images" / "train"
mask_dir = base_dir / "images" / "train_masks"
edge_dir = base_dir / "images" / "train_edges"

mask_dir.mkdir(parents=True, exist_ok=True)
edge_dir.mkdir(parents=True, exist_ok=True)

# global thresholds for the whole dataset
binary_threshold = 90  # adjust by dataset brightness
edge_threshold = 80

file_list = sorted([p for p in train_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
print('Found', len(file_list), 'images in', train_dir)

# function to run pipeline for a single image

def process_image(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('FAILED to read', p)
        return

    # 2. create mask using thresholding
    _, mask = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)

    # 3. detect edges using gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = np.uint8(np.clip((grad_mag / grad_mag.max()) * 255, 0, 255))

    # 4. create edge image by thresholding gradient magnitude
    _, edge_mask = cv2.threshold(grad_mag, edge_threshold, 255, cv2.THRESH_BINARY)

    # save
    out_mask_path = mask_dir / (p.stem + '_mask.png')
    out_edge_path = edge_dir / (p.stem + '_edge.png')
    cv2.imwrite(str(out_mask_path), mask)
    cv2.imwrite(str(out_edge_path), edge_mask)

    return img, mask, grad_mag, edge_mask

# 6-9. apply whole dataset
for i, p in enumerate(file_list, start=1):
    process_image(p)
    if i % 50 == 0:
        print(f'Processed {i}/{len(file_list)} images')

print('Done processing all images.')

# 10. show one original image and its mask with edges
if file_list:
    sample = file_list[0]
    img, mask, grad_mag, edge_mask = process_image(sample)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img, cmap='gray'); axs[0].set_title('Original')
    axs[1].imshow(mask, cmap='gray'); axs[1].set_title('Binary mask')
    axs[2].imshow(grad_mag, cmap='gray'); axs[2].set_title('Gradient magnitude')
    axs[3].imshow(edge_mask, cmap='gray'); axs[3].set_title('Edge mask')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
