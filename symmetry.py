import cv2
import numpy as np
import math
import detection


def determine_side(left, right, w):  
    temp = np.zeros(left.shape, dtype=np.uint8)
    for j in range(w):
        for i in range(left.shape[0]):
            if left[i][j] < 245 and 245 <= right[i][j] <= 255:
                temp[i][j] = 255

    # cv2.imshow('t',temp)
    # cv2.waitKey(0)
    return cv2.mean(temp)



def symmetry(img, pt1, pt2):
    # Average the x-coordinates to get a clean vertical axis (in case pt1 and pt2 are not perfectly aligned)
    x_axis = (pt1[0] + pt2[0]) // 2

    x_axis = max(0, min(x_axis, img.shape[1] - 1))

    left_half = img[:, :x_axis]
    
    
    right_half = img[:, x_axis:]
    
    if left_half.shape[1] != right_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]
        

    mirrored_left = np.flip(left_half, axis=1)
    mirrored_right = np.flip(right_half, axis=1)

    paste_width_l = min(mirrored_left.shape[1], img.shape[1] - x_axis)
    mirror_left_mean = determine_side(cv2.cvtColor(mirrored_left, cv2.COLOR_BGR2GRAY),cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY), paste_width_l)
    
    paste_width_r = min(mirrored_right.shape[1], img.shape[1] - x_axis)
    mirror_right_mean = determine_side(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY), cv2.cvtColor(mirrored_left, cv2.COLOR_BGR2GRAY), paste_width_r)
    
    print("Mirrored Means: ", mirror_left_mean[0], mirror_right_mean[0])
    
    if abs(mirror_left_mean[0] - mirror_right_mean[0]) < 20:
        mirrored = mirrored_left
    else:   mirrored = mirrored_left if mirror_left_mean > mirror_right_mean else mirrored_right
    
    paste_width = min(mirrored.shape[1], img.shape[1] - x_axis)



    output = img.copy()

    output[:, x_axis:x_axis + paste_width] = mirrored[:, :paste_width]
    # cv2.imshow('o', output)
    # cv2.waitKey(0)
    return output


def angle_from_bottom(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt1[1] - pt2[1]  # invert y because image origin is top-left

    angle_rad = np.arctan2(dy, dx)  # angle from x-axis
    angle_deg = np.degrees(angle_rad)

    angle_deg = (angle_deg + 360) % 360
    return angle_deg

def rotate_symm(img, pt1, pt2):
    # Rotation parameters
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)  
    orig_angle = angle_from_bottom(pt1, pt2)                 # Rotation angle in degrees
    scale = 1.0        
    print("Angle is: ", orig_angle)
    
    isRotated = True
    if (0 <= orig_angle <= 20) or (75 <= orig_angle <= 100) or (170 <= orig_angle <= 190) or (340 <= orig_angle <= 360):
        isRotated = False
        print("No rotation needed")
        rotated = img
    
    else:
        # angle = 90 - orig_angle if orig_angle < 180 else 270 - orig_angle
        upright_angles = [0, 90, 180, 270]
    
    # Find the nearest upright angle (using absolute difference)
        closest_angle = min(upright_angles, key=lambda x: abs(x - orig_angle))
        
        # Calculate the shortest rotation (clockwise or counterclockwise)
        diff = (closest_angle - orig_angle) % 360
        
        # If the angle difference is greater than 180 degrees, rotate the other way (shortest direction)
        if diff > 180:
            rotation_needed = diff - 360
        else:
            rotation_needed = diff
            
        M = cv2.getRotationMatrix2D(center, rotation_needed, scale)

        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        # cv2.imshow("rotate", rotated)
        # cv2.waitKey(0)
        
        
    symm = symmetry(rotated, pt1, pt2)
    # cv2.imshow("symmetry", symm)
    # cv2.waitKey(0)
    
    
    detected,x,y,label,w_det,h_det = detection.detection(symm)
    # cv2.imshow("detected", detected)
    # cv2.waitKey(0)

    # New Rotation parameters
    if isRotated:
        inv_angle = 360 - rotation_needed
        (h, w) = symm.shape[:2]
        center = (w // 2, h // 2) 
        M_new = cv2.getRotationMatrix2D(center, inv_angle, scale)
        new_rotated = cv2.warpAffine(symm, M_new, (w, h), borderValue=(255, 255, 255))
        # cv2.imshow("unrotate", new_rotated)
        # cv2.waitKey(0)
        
    return x,y,label,w_det,h_det, orig_angle
    
   
def ellipse(img, contours):

    largest_ellipse = None
    max_area = 0
    extracted = None 

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (_, _), (MA, ma), _ = ellipse

            area = MA * ma
            if MA > 0 and ma > 0 and area > max_area and MA < max(img.shape[:2]):
                max_area = area
                largest_ellipse = ellipse
                contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                extracted = cv2.bitwise_and(img, img, mask=contour_mask)
                extracted[(extracted == [0, 0, 0]).all(axis=2)] = [255, 255, 255]

    # Draw and compute ellipse axes
    if largest_ellipse:
        (cx, cy), (MA, ma), angle = largest_ellipse

        # Convert angle to radians
        theta = math.radians(angle)

        # Major axis (half-length in each direction)
        # dx_major = (MA / 2) * math.cos(theta)
        # dy_major = (MA / 2) * math.sin(theta)

        # Minor axis (half-length in each direction)
        dx_minor = (ma / 2) * math.cos(theta + math.pi / 2)
        dy_minor = (ma / 2) * math.sin(theta + math.pi / 2)

        # Endpoints of major axis
        # pt1_major = [int(cx - dx_major), int(cy - dy_major)]
        # pt2_major = [int(cx + dx_major), int(cy + dy_major)]
        # cv2.line(img, pt1_major, pt2_major, (0, 0, 255), 2)

        # Endpoints of minor axis
        pt1_minor = [int(cx - dx_minor), int(cy - dy_minor)]
        pt2_minor = (int(cx + dx_minor), int(cy + dy_minor))


        
    # symm = img[ pt1_major[1]: , pt1_major[0]: pt2_major[0]]
    temp = extracted.copy()
    cv2.ellipse(temp, largest_ellipse, (0, 255, 0), 2)
    cv2.line(temp, pt1_minor, pt2_minor, (0, 0, 255), 2)
    # cv2.imshow("ellipse and axis", temp)
    # cv2.waitKey(0)
    
    x,y, label,w,h, angle = rotate_symm(extracted, pt1_minor, pt2_minor)
    return [x,y, label,w,h, angle]
