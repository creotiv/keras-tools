import numpy as np
import cv2
import math


def deskew(orig_image, ench_image, M, res_path="agon2.jpg"):
    im_out = cv2.warpPerspective(orig_image, np.linalg.inv(M), (ench_image.shape[1], ench_image.shape[0]))
    return im_out
    
def unwarp(orig, ench):
    '''
        Converts original image to shape and crop like enchanced. 
        Needed if you original images in dataset differs from resulted by crop and shape
    '''
    ench_image = cv2.imread(ench, 0)
    orig_image = cv2.imread(orig, 0)
    orig_image_rgb = cv2.imread(orig)

    surf = cv2.ORB_create()
    kp1, des1 = surf.detectAndCompute(ench_image, None)
    kp2, des2 = surf.detectAndCompute(orig_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
                              
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))

        return deskew(orig_image_rgb, ench_image, M)
        
    else:
        print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None
