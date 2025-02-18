# VR_Assignment1_AnukritiSingh_MS202450

Prerequisites
Ensure you have Python installed along with the necessary dependencies. You can install them using:

```bash
pip install -r requirements.txt
```

Methods used:
1. Coin Detection and Segmentation
Hough Circles Transform: This technique is chosen due to its effectiveness in detecting circular objects even in cluttered backgrounds.

Why Not Canny Edge Detector?
Canny edge detection alone does not differentiate between circles and other edges.
Hough Circles method is specialized for circular object detection, making it more accurate for coin segmentation.


