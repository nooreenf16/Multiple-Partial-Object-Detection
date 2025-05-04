import cv2
import segment
import symmetry 
import detection
import numpy as np

# img = cv2.imread("images/vase.webp")
# img = cv2.imread("images/keyboard.webp")
img = cv2.imread("images/mul.jpg")
# img = cv2.resize(img, ())

cv2.imshow("Original", img)
cv2.waitKey(0)

labels = [] # [[100,200,'here', 200,400, 30]]
seg_contours = segment.segment(img)
print(len(seg_contours))
for s in seg_contours:
    labels.append( symmetry.ellipse(img,[s]))
    cv2.destroyAllWindows()


# draw labels on original image
mask_total = np.zeros(img.shape[:2], dtype=np.uint8)

(h, w) = img.shape[:2]
center = (w // 2, h // 2)

for l in labels:  # x, y, label, w, h, angle
    temp = np.zeros_like(mask_total)

    # Draw single box (filled or bordered)
    cv2.rectangle(temp, (l[0], l[1]), (l[0] + l[3], l[1] + l[4]), 255, 3)
    cv2.putText(temp, l[2], (l[0], l[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)  
    
    # Rotate that box around image center
    if l[5] > 190: angle = l[5]+90 
    elif 175 <= l[5] <= 190: angle = 0
    else: angle = l[5]-90
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(temp, M, (w, h), borderValue=0)

    mask_total = cv2.bitwise_or(mask_total, rotated)

img[mask_total == 255] = 255

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

