from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from util import get_car, read_license_plate, write_csv
from sort.sort import Sort  # Tracking only for videos

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Define vehicle classes (from COCO dataset)
vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Input file (change this to your file path)
input_path = './sample.mp4'  # Change to an image or video file

# Check if input is an image or a video
is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))
results = {}

if is_video:
    print("Processing video...")
    cap = cv2.VideoCapture(input_path)
    mot_tracker = Sort()  # Initialize tracker only for videos
    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Could not read frame {frame_nmr}. Skipping...")
            break

        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        if len(detections_) == 0:
            print(f"Warning: No vehicles detected in frame {frame_nmr}. Skipping tracking.")
            continue  # Skip this frame if no vehicles detected

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Ensure get_car() returns valid values
            car_info = get_car(license_plate, track_ids)
            if car_info is None or (isinstance(car_info, (list, tuple)) and len(car_info) != 5):
                print(f"Warning: No matching car found for license plate in frame {frame_nmr}. Skipping...")
                continue

            xcar1, ycar1, xcar2, ycar2, car_id = car_info
            car_id = int(car_id)

            # Extract and process license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            if license_plate_crop is None or license_plate_crop.shape[0] == 0 or license_plate_crop.shape[1] == 0:
                print(f"Warning: Invalid license plate crop in frame {frame_nmr}. Skipping OCR.")
                continue

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if car_id not in results[frame_nmr]:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text if license_plate_text else "Unknown",
                        'bbox_score': score,
                        'text_score': license_plate_text_score if license_plate_text_score else "Unknown"
                    }
                }

    cap.release()

else:
    print("Processing image...")
    frame = cv2.imread(input_path)
    results[0] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Ensure get_car() returns valid values
        car_info = get_car(license_plate, detections_)
        if car_info is None or (isinstance(car_info, (list, tuple)) and len(car_info) != 5):
            print("Warning: No matching car found for license plate. Skipping...")
            continue

        xcar1, ycar1, xcar2, ycar2, car_id = car_info
        car_id = int(car_id)

        # Extract and process license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        if license_plate_crop is None or license_plate_crop.shape[0] == 0 or license_plate_crop.shape[1] == 0:
            print("Warning: Invalid license plate crop. Skipping OCR.")
            continue

        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

        if car_id not in results[0]:
            results[0][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text if license_plate_text else "Unknown",
                    'bbox_score': score,
                    'text_score': license_plate_text_score if license_plate_text_score else "Unknown"
                }
            }

# Debugging: Print results before saving
print(json.dumps(results, indent=4))

# Write results to CSV
output_csv = './results.csv'
write_csv(results, output_csv)
print(f"Results saved to {output_csv}")
