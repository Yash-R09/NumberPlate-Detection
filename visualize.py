import ast
import cv2
import numpy as np
import pandas as pd
import os

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """Draws fancy borders around the detected objects."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Load results from CSV
results = pd.read_csv('./test_interpolated.csv')

# Input file
input_path = './sample.mp4'  
is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
license_plate = {}

if is_video:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
else:
    frame = cv2.imread(input_path)

for car_id in np.unique(results['car_id']):
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    best_result = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)].iloc[0]
    license_plate[car_id] = {'license_crop': None, 'license_plate_number': best_result['license_number']}
    
    if is_video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_result['frame_nmr'])
        ret, frame = cap.read()
    else:
        frame = cv2.imread(input_path)

    x1, y1, x2, y2 = ast.literal_eval(best_result['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    license_plate[car_id]['license_crop'] = license_crop

if not is_video:
    df_ = results
    for _, row in df_.iterrows():
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, 200, 200)
        
        x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        if license_crop.size > 0:
            license_crop_resized = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            y_start = max(0, int(car_y1) - 500)
            y_end = min(frame.shape[0], int(car_y1) - 100)
            x_start = max(0, int((car_x2 + car_x1 - license_crop_resized.shape[1]) / 2))
            x_end = min(frame.shape[1], int((car_x2 + car_x1 + license_crop_resized.shape[1]) / 2))
            H_new = y_end - y_start
            W_new = x_end - x_start

            if H_new > 0 and W_new > 0:
                license_crop_resized = cv2.resize(license_crop, (W_new, H_new))
                frame[y_start:y_end, x_start:x_end, :] = license_crop_resized

cv2.imwrite('./out_image.jpg', frame)
print("Processed image saved as out_image.jpg")
