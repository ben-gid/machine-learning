from PIL import Image
from PIL import ImagePalette
from pathlib import Path
import sys
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from clusters import run_kMeans, find_closest_centroids

JPEG_IMG = Path("imgs/image.jpeg")
# desired amount of colors in compressed image
MAX_COLORS = 16
# iterations to run kmeans
ITERS = 10
# path to save img to
OUT_PATH = Path("imgs/compressed_image.jpeg")

def main():
    img_2d, original_image_shape = shape_img_2d(JPEG_IMG)
    compressed_img = compress(img_2d, MAX_COLORS, original_image_shape)
    plt.imsave(OUT_PATH, compressed_img)
    
    # get sizes in bytes
    original_size = JPEG_IMG.stat().st_size
    compressed_size = OUT_PATH.stat().st_size
    
    print("Main results")
    print(f"compressed {JPEG_IMG} to {MAX_COLORS} colors and saved to {OUT_PATH}")
    print(f"Original Size: {original_size / 1024:.2f} KB")
    print(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    print(f"Reduction: {100 * (1 - compressed_size / original_size):.1f}%")

def shape_img_2d(img:Path) -> tuple[np.ndarray, Sequence[int]]:
    f_type = img.suffix
    # load image
    original_img = plt.imread(img) 
    
    # get original image shape (height, width, channels)
    original_img_shape = original_img.shape
    
    # jpg devide all pixels by 255 so values are between 0 and 1
    if f_type in [".jpeg", ".jpg"]:
        original_img = original_img / 255.0
    
    # reshape img; if image has color shape will be (number of pixels, 3)
    # if image is black and white shape will be (number of pixels, 1)
    img_flattened = np.reshape(original_img, (-1, original_img_shape[-1]))
    return img_flattened, original_img_shape
    

def compress(img_as_2darray: np.ndarray, max_colors: int, shape: Sequence[int]):
    # get the centroids
    centroids, _ = run_kMeans(img_as_2darray, max_colors, ITERS)
    # get each pixels centroid
    indices = find_closest_centroids(img_as_2darray, centroids)
    # replace each pixel with the color of the closest centroid
    compressed = centroids[indices, :]
    # reshape
    compressed_img = np.reshape(compressed, shape)
    
    return compressed_img

def optimised_main():
    """
    Orchestrates the image compression pipeline using JPEG-specific optimization.

    Authored by Gemini.

    Problem with previous versions:
        - plt.imsave/PNG: Saved the 16-color image as 'lossless' data, which lacked 
        the patterns needed for PNG to be efficient, leading to a larger file.
        - Standard JPEG: Struggles with the sharp color borders created by K-Means,
        adding overhead to preserve those edges.

    Solution:
        This version saves back to JPEG but uses PIL's 'optimize' flag to strip 
        unnecessary metadata and headers. It also applies a custom 'quality' 
        setting, allowing the JPEG algorithm to compress the 16-color clusters 
        much more aggressively than the high-quality original.
    """
    img_2d, original_shape = shape_img_2d(JPEG_IMG)
    
    # 1. Compress
    compressed_img_floats = compress(img_2d, MAX_COLORS, original_shape)
    
    # 2. Convert to uint8
    img_uint8 = (np.clip(compressed_img_floats, 0, 1) * 255).astype(np.uint8)
    
    # 3. Create PIL Image
    image_to_save = Image.fromarray(img_uint8)
    
    # 4. Save as JPEG with 'optimize' and 'quality' control
    # Note: We must ensure it's in RGB mode because JPEG doesn't support 'P' mode directly
    if image_to_save.mode != 'RGB':
        image_to_save = image_to_save.convert('RGB')
        
    image_to_save.save(
        OUT_PATH, 
        "JPEG", 
        optimize=True,  # This strips extra metadata/headers
        quality=50      # Lower quality slightly (standard is 75)
    )
    
    # get sizes in bytes
    original_size = JPEG_IMG.stat().st_size
    compressed_size = OUT_PATH.stat().st_size

    print("Optimized main results")
    print(f"compressed {JPEG_IMG} to {MAX_COLORS} colors and saved to {OUT_PATH}")
    print(f"Original Size: {original_size / 1024:.2f} KB")
    print(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    print(f"Reduction: {100 * (1 - compressed_size / original_size):.1f}%")

if __name__ == "__main__":
    optimised_main()
    main()