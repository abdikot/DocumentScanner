from transform import four_points_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# load gambar dan menghitung ratio dari tinggi lama ke  tinggi baru, 
# menduplikatnya dan meresizenya
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# mengubah gambar menjadi hitam-putih, 
# memblur gambar dan mencari sudut-sudut gambar
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

# menampilkan gambar orginal dan sudut-sudut gambar yang terdeteksi
print("STEP 1: Mendeteksi Sudut")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# mencari contours dari gambar yang sudah di 'edged'
# menyimpan hannya yang paling besar, dan 
# menginisialisasi contours gambar
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

# melakukan loop di contours
for c in cnts:
    # perkirakan contournya
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # melakukan if
    # jika perkiraan contour memiliki 4 titik, maka
    # kita berasumsi bahwa kita sudah menemukan 'screen'
    if len(approx) == 4:
        screenCnt = approx
        break

print("STEP 2: Mencari Kertasnya")
cv2.drawContours(image, [screenCnt], -1,(0,255,0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 


# mengaplikasikan fungsi four_point_transform
# untuk mendapatkan gambar dari atas 
warped = four_points_transform(orig, screenCnt.reshape(4, 2) * ratio)

# mengubah gambar yang sudah di warped menjadi hitam-putih, lalu melakukan 'threshold'
# untuk mendapat tampilan hitam-putih
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# menampilkan gambar original dan yang sudah di scan
print("STEP 3: Mengapply pengubahan perpektif")
cv2.imshow("ORIGINAL", imutils.resize(orig, height = 650))
cv2.imshow("SCANNED", imutils.resize(warped, height = 650))
cv2.waitKey(0)