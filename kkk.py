import cv2

frame = cv2.imread("123123.jpg")
frame = cv2.resize(frame, (512, 512))
cv2.imwrite("123123.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

