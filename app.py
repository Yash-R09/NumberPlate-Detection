import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import asyncio
import os
import logging
from ultralytics import YOLO
from sort.sort import Sort  # Tracking for videos
from util import read_license_plate  # Import the function from utils.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("license_plate_app")

# Prevent Torch JIT issues
os.environ["PYTORCH_JIT"] = "0"

# Fix Asyncio Runtime Error for Windows
if os.name == "nt":  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure Asyncio does not conflict with Streamlit
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

get_or_create_eventloop()

# Force model to use CPU if no GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load YOLO models
@st.cache_resource
def load_models():
    logger.info("Loading models...")
    coco_model = YOLO("yolov8n.pt").to(device)
    license_plate_detector = YOLO("license_plate_detector.pt").to(device)
    return coco_model, license_plate_detector

coco_model, license_plate_detector = load_models()

# Vehicle class IDs from COCO dataset
vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Streamlit UI
st.title("🚗 Automatic License Plate Recognition using YOLOv8")
st.sidebar.header("Upload Image or Video")

# Upload input
uploaded_file = st.sidebar.file_uploader("Choose an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Processing function for images
def process_image(image):
    frame = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    
    # Detect vehicles
    detections = coco_model(frame)[0]
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    extracted_texts = []
    
    # Debug log
    st.sidebar.write(f"Found {len(license_plates.boxes.data)} license plates")
    
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 < frame.shape[1] and y2 < frame.shape[0]:
            # Draw rectangle around license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Crop the license plate region
            license_plate_crop = frame[y1:y2, x1:x2].copy()
            
            if license_plate_crop.size > 0 and license_plate_crop.shape[0] > 0 and license_plate_crop.shape[1] > 0:
                # Convert to grayscale
                gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Try multiple preprocessing approaches
                # Approach 1: Binary thresholding
                _, thresh_plate1 = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
                
                # Approach 2: Adaptive thresholding
                thresh_plate2 = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)
                
                # Try both approaches for OCR
                plate_text1, confidence1 = read_license_plate(thresh_plate1)
                plate_text2, confidence2 = read_license_plate(thresh_plate2)
                
                # Use the result with higher confidence
                if confidence1 > confidence2 and confidence1 > 0.3:
                    plate_text, confidence = plate_text1, confidence1
                elif confidence2 > 0.3:
                    plate_text, confidence = plate_text2, confidence2
                else:
                    plate_text, confidence = "", 0.0
                
                # Display the cropped plate for debugging
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(license_plate_crop, caption="Cropped Plate", width=150)
                with col2:
                    st.image(thresh_plate1, caption="Binary Threshold", width=150)
                with col3:
                    st.image(thresh_plate2, caption="Adaptive Threshold", width=150)
                
                if plate_text:
                    # Add text to image with high visibility
                    text_bg = (x1, y1-25)
                    text_position = (x1+5, y1-5)
                    
                    # Add background rectangle for text visibility
                    cv2.rectangle(frame, text_bg, (x2, y1), (255, 0, 0), -1)
                    cv2.putText(frame, plate_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (255, 255, 255), 2)
                    
                    # Save to results list
                    extracted_texts.append((plate_text, confidence, license_plate_crop))
                    
                    # Log for debugging
                    st.sidebar.success(f"✅ Plate detected: {plate_text} (conf: {confidence:.2f})")
                else:
                    st.sidebar.warning("⚠️ Plate detected but text could not be read")
            else:
                st.sidebar.warning("⚠️ Empty plate crop")
    
    return frame, extracted_texts

# Processing function for videos
def process_video(video_path):
    # Create a status placeholder for progress updates
    status_placeholder = st.empty()
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video dimensions: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    status_placeholder.text(f"Processing video: {width}x{height} at {fps} FPS")

    if fps == 0 or width == 0 or height == 0:
        st.error("Error: Invalid video file.")
        return None

    # Ensure FPS is reasonable
    fps = max(1, min(30, fps))  # Cap FPS between 1 and 30
    
    # Create a named temporary file with .mp4 extension
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_out.name
    temp_out.close()  # Close it to ensure it's written properly
    
    logger.info(f"Output path: {output_path}")
    
    # Use H.264 codec which is web-compatible
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Fallback to more compatible codecs if avc1 fails
        if not out.isOpened():
            logger.warning("avc1 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            logger.warning("mp4v codec failed, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        logger.error(f"Error initializing VideoWriter: {e}")
        st.error(f"Error initializing video writer: {e}")
        return None

    mot_tracker = Sort()
    frame_count = 0
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Create a container for detected plates
    plate_container = st.container()
    with plate_container:
        st.subheader("Detected License Plates")
        plate_display = st.empty()
    
    # Dictionary to store detected plates (vehicle_id -> plate_text)
    detected_plates = {}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when no more frames
            
            # Update progress
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                if frame_count % 10 == 0:
                    status_placeholder.text(f"Processing frame {frame_count}/{total_frames} ({int(progress*100)}%)")
            
            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            vehicle_boxes = []
            
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and score > 0.4:  # Confidence threshold
                    detections_.append([x1, y1, x2, y2, score])
                    vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2), int(class_id)))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Add vehicle class label
                    vehicle_type = ["", "", "Car", "Motorcycle", "", "Bus", "", "Truck"][int(class_id)]
                    cv2.putText(frame, f"{vehicle_type} ({score:.2f})", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Track vehicles
            track_bbs_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else []
            
            # Map tracked vehicle IDs to their bounding boxes
            vehicle_tracks = {}
            for track in track_bbs_ids:
                x1, y1, x2, y2, track_id = track
                vehicle_tracks[int(track_id)] = (int(x1), int(y1), int(x2), int(y2))
                # Add tracking ID to vehicle
                cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y2) + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            new_detections = []
            
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                
                # Convert to integer coordinates and validate
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                if (x2-x1 > 10 and y2-y1 > 10 and  # Minimum size check
                    x1 >= 0 and y1 >= 0 and 
                    x2 < frame.shape[1] and y2 < frame.shape[0] and
                    score > 0.5):  # Confidence threshold
                    
                    # Draw rectangle around license plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Crop the license plate region
                    license_plate_crop = frame[y1:y2, x1:x2].copy()
                    
                    if license_plate_crop.size > 0:
                        # Convert to grayscale
                        gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        
                        # Apply multiple preprocessing techniques
                        # 1. Basic binary threshold
                        _, thresh_plate1 = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
                        
                        # 2. Adaptive threshold
                        thresh_plate2 = cv2.adaptiveThreshold(gray_plate, 255, 
                                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                             cv2.THRESH_BINARY, 11, 2)
                        
                        # Try both preprocessing methods
                        plate_text1, confidence1 = read_license_plate(thresh_plate1)
                        plate_text2, confidence2 = read_license_plate(thresh_plate2)
                        
                        # Use the result with higher confidence
                        if confidence1 > confidence2 and confidence1 > 0.4:
                            plate_text, confidence = plate_text1, confidence1
                        elif confidence2 > 0.4:
                            plate_text, confidence = plate_text2, confidence2
                        else:
                            plate_text, confidence = "", 0.0
                        
                        # Clean up plate text (remove non-alphanumeric)
                        if plate_text:
                            import re
                            plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)
                        
                        if plate_text and len(plate_text) >= 3:  # Minimum length check
                            # Find which vehicle this plate belongs to by checking overlap
                            vehicle_id = None
                            for track_id, vehicle_box in vehicle_tracks.items():
                                vx1, vy1, vx2, vy2 = vehicle_box
                                # Check if plate is inside vehicle bounding box
                                if (x1 >= vx1 and y1 >= vy1 and 
                                    x2 <= vx2 and y2 <= vy2):
                                    vehicle_id = track_id
                                    break
                            
                            # Add background rectangle for text visibility
                            cv2.rectangle(frame, (x1, y1-25), (x2, y1), (255, 0, 0), -1)
                            cv2.putText(frame, f"{plate_text} ({confidence:.2f})", 
                                        (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (255, 255, 255), 2)
                            
                            # Store the detection
                            if vehicle_id is not None:
                                detected_plates[vehicle_id] = (plate_text, confidence, frame_count)
                                new_detections.append((vehicle_id, plate_text, confidence))
                            
                            # Log periodically
                            if frame_count % 30 == 0:
                                logger.info(f"Detected plate: {plate_text} (conf: {confidence:.2f})")
            
            # Update detected plates display periodically
            if frame_count % 15 == 0 or new_detections:
                # Display table of detected plates
                plate_info = []
                for vehicle_id, (text, conf, frame_num) in detected_plates.items():
                    plate_info.append(f"Vehicle #{vehicle_id}: {text} (Conf: {conf:.2f}, Frame: {frame_num})")
                
                if plate_info:
                    plate_display.markdown("  \n".join(plate_info))
                else:
                    plate_display.text("No license plates detected yet.")
            
            # Write the frame
            out.write(frame)
            frame_count += 1
            
            # Log progress periodically
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count} frames")
                
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
        return None
    finally:
        # Ensure resources are released properly
        cap.release()
        out.release()
        progress_bar.empty()
        status_placeholder.empty()
    
    # Verify output file
    if frame_count == 0:
        st.error("Error: No frames were processed.")
        return None
    
    if not os.path.exists(output_path):
        st.error("Error: Output file was not created.")
        return None
        
    file_size = os.path.getsize(output_path)
    logger.info(f"Output file size: {file_size} bytes, frames processed: {frame_count}")
    
    if file_size < 1000:  # Less than 1KB
        st.error(f"Error: Output file is too small ({file_size} bytes). Video processing likely failed.")
        return None
    
    # Ensure file has been properly written to disk
    import time
    time.sleep(0.5)  # Small delay to ensure file is available
    
    # Final summary
    st.sidebar.success(f"Video processing complete! Processed {frame_count} frames.")
    st.sidebar.info(f"Output file: {os.path.basename(output_path)} ({file_size/1024:.1f} KB)")
    
    # Display summary of all detected plates
    if detected_plates:
        with st.expander("License Plate Detection Summary", expanded=True):
            st.write(f"Total unique vehicles tracked: {len(vehicle_tracks)}")
            st.write(f"Total license plates detected: {len(detected_plates)}")
            
            # Create a simple table
            data = []
            for vehicle_id, (text, conf, frame_num) in detected_plates.items():
                data.append({
                    "Vehicle ID": vehicle_id,
                    "License Plate": text,
                    "Confidence": f"{conf:.2f}",
                    "Detected at Frame": frame_num
                })
            
            if data:
                import pandas as pd
                st.table(pd.DataFrame(data))
    
    return output_path

# Handle Upload
if uploaded_file is not None:
    file_type = uploaded_file.type
    
    try:
        if "image" in file_type:
            st.sidebar.info("Processing Image... ⏳")
            processed_image, license_texts = process_image(uploaded_file)
            st.image(processed_image, channels="BGR", use_container_width=True)
            
            if license_texts:
                st.subheader("📌 Detected License Plates:")
                for text, conf in license_texts:
                    st.write(f"**{text}** (Confidence: {conf:.2f})")
            else:
                st.write("No license plate detected.")
                
        elif "video" in file_type:
            st.sidebar.info("Processing Video... ⏳")
            
            # Create a named temporary file to save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name
            
            # Process the video
            st.sidebar.text("Video processing started...")
            processed_video_path = process_video(video_path)
            
            # Display the processed video
            if processed_video_path and os.path.exists(processed_video_path):
                st.success("Video processing complete!")
                
                # Display video file info
                file_size = os.path.getsize(processed_video_path) / 1024  # KB
                st.sidebar.text(f"Processed file size: {file_size:.1f} KB")
                
                # Display the video
                st.video(processed_video_path)
            else:
                st.error("Error: Processed video not found or invalid.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Exception: {str(e)}", exc_info=True)

else:
    st.sidebar.info("Please upload an image or video.")