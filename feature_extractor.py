# feature_extractor.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

LBP_POINTS = 24
LBP_RADIUS = 3

def extract_features(img):
    img = cv2.resize(img, (128,128))

    # ORB descriptors
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.zeros((1,32))
    orb_feature = des.flatten()
    orb_feature = orb_feature[:500] if orb_feature.size > 500 else np.pad(orb_feature, (0,500 - orb_feature.size))

    # HSV histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).flatten()

    # RGB color histogram
    chans = cv2.split(img)
    color_hist = []
    for c in chans:
        hist = cv2.calcHist([c],[0],None,[32],[0,256]).flatten()
        color_hist.extend(hist)
    color_hist = np.array(color_hist)

    # LBP features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, LBP_POINTS+3), density=True)

    # Combine all features
    final_feature = np.hstack([orb_feature, hsv_hist, color_hist, lbp_hist])
    return final_feature
