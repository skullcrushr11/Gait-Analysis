import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame=cap.read()
    cv2.imshow('opencv feed',frame)

    if cv2.waitKey(10) & 0xFF ** ord('q'):
        break
cap.release()
cv2.destroyAllWindows()