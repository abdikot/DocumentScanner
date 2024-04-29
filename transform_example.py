from transform import four_points_transform
import numpy as np
import argparse
import cv2

# membuat argument parse dan parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="patch to image file")
ap.add_argument("-c", "--coords",help="comma seperated list of source points")
args = vars(ap.parse_args())

# memuat gambar dan mengambil 'source coordinate' (list dari x, dan y)
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

# aplikasikan 'bird eye view' pada gambar
wraped = four_points_transform(image,pts)

cv2.imshow("Original", image)
cv2.imshow("Wraped", wraped)
cv2.waitKey(0)