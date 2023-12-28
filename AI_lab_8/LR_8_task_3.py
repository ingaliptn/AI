import cv2
img = cv2.imread("tkachuk_full.jpg")
print(img.shape)
imgResize = cv2.resize(img, (1000, 450))
print(imgResize.shape)
imgCropped = img[20:350, 250:460]
cv2.imshow("Image", img)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)

