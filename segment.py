import cv2
import numpy as np

# Load image

def segment(img):
    # Apply mean shift filtering
    shifted = cv2.pyrMeanShiftFiltering(img, sp=20, sr=60)
    
    cv2.imshow('s', shifted)
    cv2.waitKey(0)
    
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200) 
    kernel = np.ones((3, 3), np.uint8)
    cv2.imshow('s', edges)
    cv2.waitKey(0)
    edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('s', edges)
    cv2.waitKey(0)

    # Find all contours separately
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# filter contours
    selected_conts = []
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        if area > max_area:
            max_area = area
            max_contour = c
        selected_conts.append(c)

# drop the contour with the max area (overall img)
    selected_conts = [c for c in selected_conts if c is not max_contour]
            
    for s in selected_conts:
        filled = np.zeros_like(edges)
        
        print(cv2.contourArea(s))
    
        cv2.drawContours(filled, [s],-1,255,cv2.FILLED)
        cv2.imshow('fill', filled)
        cv2.waitKey(0)

    return selected_conts