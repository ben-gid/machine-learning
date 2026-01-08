from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from clusters import run_kMeans

def main():
    l_argv = len(sys.argv)
    if l_argv != 3:
        raise ValueError(f"Usage: python {__file__} [in_img_path] [out_img_path]")
    
    try:
        in_img = plt.imread(sys.argv[1]) 
    except FileNotFoundError as e:
        print(e)
    out_img = Path(sys.argv[2])
    
    compressed_img = compress(in_img, 16)
    compressed_img
        

def shape_img_2d(img:Path, to:str):
    pass

def compress(img_as_array: np.ndarray, max_colors: int):
    centroids, cost = run_kMeans(img_as_array, max_colors, 10)

if __name__ == "__main__":
    main()