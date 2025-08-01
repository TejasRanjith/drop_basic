import cv2
import time
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

@app.route("/")
def detect_drops():
    cap = cv2.VideoCapture('unfocused.mp4')
    drop_count = 0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return "Failed to read video"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    last_drop_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            rect_area = w * h
            extent = float(area) / rect_area if rect_area != 0 else 0

            if current_time - last_drop_time >= 0.3:
                if 0.3 < aspect_ratio < 1.2 and 0.6 < solidity < 0.95 and 0.4 < extent < 0.9:
                    drop_count += 1
                    last_drop_time = current_time
                    break

        prev_gray = gray.copy()

    cap.release()
    return f"Total drops detected: {drop_count}"
