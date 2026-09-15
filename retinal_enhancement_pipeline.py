# ==========================================
# WEEK 5 - FINAL HYBRID PIPELINE
# SURP Project
# ==========================================

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ==========================================
# LOAD APTOS IMAGE
# ==========================================

image = io.imread(
    "/Users/olivialusignan/Desktop/APTOS/train_images/0104b032c141.png"
)

# ==========================================
# GREEN CHANNEL (No FOV mask for APTOS)
# ==========================================

green = image[:, :, 1]

green_f = green.astype(float) / 255.0

# ==========================================
# CLAHE
# ==========================================

green_8bit = (green_f * 255).astype(np.uint8)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(green_8bit)
clahe_f = clahe_img.astype(float) / 255.0

# ==========================================
# GAUSSIAN SMOOTHING
# ==========================================

gaussian_img = gaussian_filter(clahe_f, sigma=1)

# ==========================================
# QFT-INSPIRED FILTER
# ==========================================

f = fftshift(fft2(gaussian_img))

rows, cols = gaussian_img.shape
crow, ccol = rows // 2, cols // 2

y, x = np.ogrid[:rows, :cols]
dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

weights = np.ones_like(dist)

weights[dist <= 15] = 0.4
weights[(dist > 15) & (dist < 80)] = 2.2
weights[dist >= 80] = 0.9

filtered_f = f * weights

pipeline_C = np.abs(ifft2(ifftshift(filtered_f)))
pipeline_C = np.clip(pipeline_C, 0, 1)

# ==========================================
# FINAL HYBRID (BLEND WITH ORIGINAL)
# ==========================================

blend_weight = 0.6

final_enhanced = (
    blend_weight * pipeline_C
    + (1 - blend_weight) * green_f
)

final_enhanced = np.clip(final_enhanced, 0, 1)

# ==========================================
# DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(green_f, cmap="gray")
plt.title("Original Green")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pipeline_C, cmap="gray")
plt.title("Pipeline C")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_enhanced, cmap="gray")
plt.title(f"Final Hybrid (Blend={blend_weight})")
plt.axis("off")

plt.tight_layout()
plt.show()