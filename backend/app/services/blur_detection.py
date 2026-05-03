import cv2
import numpy as np

# Frames whose Laplacian variance is below this are discarded.
# Tune this value if you find too many/few frames being rejected.

DEFAULT_BLUR_THRESHOLD: float = 100.0


def laplacian_variance(frame: np.ndarray) -> float: #Laplacian: Mathematical operation that runs through each pixel and measures how is it different 
                                                    # from the last pixel. 
                                                    # (Sharp Image: high varience cause the edges are sharp completely differnet case for the blur ones.)

    if frame is None or frame.size == 0:
        return 0.0

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame  
    #Used to detect the brightness of the image. not the color cause color doesnot give the shrapness but rather the brightness.
    #(0->black, 255->White, in betweee all grey)
    # 3 colors Red Green Blue --> (grey = 0.299 × R + 0.587 × G + 0.114 × B) --> cvtColor handles this weighting automatically

    laplacian = cv2.Laplacian(grey, cv2.CV_64F)  
    # images are stored in 8 bit unsigned binary. Laplacian operation produces -ve values which can result in the loss of the information(regarding the decrease in the brightness of the edge)
    # CV_64F -> tells OpenCV to store the Laplacian results as 64-bit floating point numbers, which can hold negatives, decimals, and very large values with high precision.
    # To sumarize if 8-bit binary was used the variance will be low and hence the point of using the threshold will be use less.
    return float(laplacian.var())  


def is_blurry(frame: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> bool:
    return laplacian_variance(frame) < threshold
