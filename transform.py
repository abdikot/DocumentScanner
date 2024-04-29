import numpy as np
import cv2


def order_points(pts):
    # menginsisialisasi sebuah list koordinat yang akan di urutkan
    # sehingga list dimulai dari kiri atas, yang berada di kanan atas,
    # yang ketiga berada di kanan bawah dan yang terakhir ada di kiri bawah

    rect = np.zeros((4,2), dtype="float32")

    # kiri atas akan memiliki sum yang paling kecil, 
    # dan kanan bawah akan  akan memiliki sum paling besar
    s =  pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # menghitung perbedaan titik, 
    # kanan atas memiliki perbedaan paling kecil
    # kiri bawah memiliki perbedaan paling besar
    diff = np.diff(pts, axis = 1) 
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # mengembalikan nilai koordinat
    return rect

def four_points_transform(image, pts):
    # mendapatkan urutan  koordinat yang konsisten dan
    # membukanya secara terpisah
    rect = order_points(pts)

    # tl = top-left     (kiri atas)
    # tr = top-right    (kanan atas)
    # br = bottom-right (kanan bawah)
    # bl = bottom-left  (kiri bawah)
    (tl, tr, br, bl) = rect

    # menghitung luas dari gambar, yang akan menjadi
    # jarak maksimal antara kanan bawah dan kiri bawah
    # x-coordinate dari kanan atas dan x-coordinate dari kiri atas
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # menghitung tinggi dari gambar, yang akan menjadi 
    # jarak maksimal antara kanan atas dan kanan bawah
    # y-coordinate dari kiri atas dan y-coordinate dari kiri bawah
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # membuat kumpulan poin destinasi untuk mendapatkan "bird eye view", 
    # menspesifikasi kordinat kiri atas, kanan atas, kanan bawah, dan  kiri bawah
    # secara berurutan
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype= "float32")

    # menghhitung maktriks sudut pandang dan mengaplikasikanya
    M = cv2.getPerspectiveTransform(rect,  dst)
    wraped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # mengembalikan nilai wraped
    return wraped
