import csv
import numpy as np
from scipy.interpolate import interp1d
import os

def interpolate_bounding_boxes(data, is_image=False):
    if not data:
        print("No data to process. Skipping interpolation.")
        return []

    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        if is_image:
            for i in range(len(car_bboxes[car_mask])):
                row = {
                    'frame_nmr': '0',
                    'car_id': str(car_id),
                    'car_bbox': ' '.join(map(str, car_bboxes[car_mask][i])),
                    'license_plate_bbox': ' '.join(map(str, license_plate_bboxes[car_mask][i])),
                    'license_plate_bbox_score': data[i].get('license_plate_bbox_score', '0'),
                    'license_number': data[i].get('license_number', '0'),
                    'license_number_score': data[i].get('license_number_score', '0')
                }
                interpolated_data.append(row)
        else:
            first_frame_number = car_frame_numbers[0]
            last_frame_number = car_frame_numbers[-1]

            for i in range(len(car_bboxes[car_mask])):
                frame_number = car_frame_numbers[i]
                car_bbox = car_bboxes[car_mask][i]
                license_plate_bbox = license_plate_bboxes[car_mask][i]

                if i > 0:
                    prev_frame_number = car_frame_numbers[i-1]
                    prev_car_bbox = car_bboxes_interpolated[-1]
                    prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                    if frame_number - prev_frame_number > 1:
                        frames_gap = frame_number - prev_frame_number
                        x = np.array([prev_frame_number, frame_number])
                        x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                        interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                        interpolated_car_bboxes = interp_func(x_new)
                        interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                        interpolated_license_plate_bboxes = interp_func(x_new)

                        car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                        license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

                car_bboxes_interpolated.append(car_bbox)
                license_plate_bboxes_interpolated.append(license_plate_bbox)

            for i in range(len(car_bboxes_interpolated)):
                frame_number = first_frame_number + i
                row = {
                    'frame_nmr': str(frame_number),
                    'car_id': str(car_id),
                    'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                    'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i])),
                }

                if str(frame_number) not in frame_numbers_:
                    row['license_plate_bbox_score'] = '0'
                    row['license_number'] = '0'
                    row['license_number_score'] = '0'
                else:
                    original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                    row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                    row['license_number'] = original_row.get('license_number', '0')
                    row['license_number_score'] = original_row.get('license_number_score', '0')

                interpolated_data.append(row)

    return interpolated_data


# Determine if input is an image or video based on filename
input_filename = 'results.csv'
is_image = False

if os.path.exists(input_filename):
    with open(input_filename, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

        # Check if data is empty
        if not data:
            print("Error: CSV file is empty. Exiting program.")
            exit(1)

        # Check if all frame numbers are '0' (likely an image)
        if 'frame_nmr' in data[0] and all(row.get('frame_nmr', '0') == '0' for row in data):
            is_image = True

else:
    print("Error: File 'results.csv' not found. Exiting program.")
    exit(1)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data, is_image)

# Write updated data to a new CSV file
output_filename = 'test_interpolated.csv'
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']

with open(output_filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

print(f"Processed data saved as {output_filename}")
