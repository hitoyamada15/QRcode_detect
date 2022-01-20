import cv2

img = cv2.imread("4K_Jet_1642499491.jpg")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
