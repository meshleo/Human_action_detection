import os
from django.shortcuts import render
from .forms import VideoUploadForm
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load your trained model
MODEL_PATH = 'ConvolutionalLongShortTermMemory_model.h5'  # Replace with your model's path
model = load_model(MODEL_PATH)

def adjust_frames(video_frames, required_frames=20):
    """
    Adjusts the number of frames in the video to match the required frame count.

    Parameters:
    - video_frames (numpy.ndarray): Input video frames, shape (num_frames, height, width, channels).
    - required_frames (int): The number of frames required by the model.

    Returns:
    - numpy.ndarray: Adjusted video frames with shape (required_frames, height, width, channels).
    """
    current_frames = video_frames.shape[0]
    height, width, channels = video_frames.shape[1:]

    if current_frames < required_frames:
        # Pad with zeros if there are fewer frames than required
        padding = np.zeros((required_frames - current_frames, height, width, channels))
        adjusted_frames = np.concatenate((video_frames, padding), axis=0)
    elif current_frames > required_frames:
        # Trim excess frames
        adjusted_frames = video_frames[:required_frames]
    else:
        # No adjustment needed
        adjusted_frames = video_frames

    return adjusted_frames

def extract_frames(video_path, frame_count=20):
    """
    Extracts frames from the video for prediction and adjusts them to match the required frame count.

    Parameters:
    - video_path (str): Path to the video file.
    - frame_count (int): The number of frames required by the model.

    Returns:
    - numpy.ndarray: Processed and adjusted frames with shape (1, frame_count, 96, 96, 3).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // frame_count)
    count = 0

    while len(frames) < frame_count and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (96, 96))  # Resize to the model's expected input size
            frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames) / 255.0  # Normalize
    adjusted_frames = adjust_frames(frames, required_frames=frame_count)
    return np.expand_dims(adjusted_frames, axis=0)

def predict_activity(video_path):
    """Predict human activity using the uploaded video."""
    frames = extract_frames(video_path)
    predictions = model.predict(frames)
    predicted_class = np.argmax(predictions)
    classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding']  # Replace with your model's class names
    return classes[predicted_class]

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            video_path = video.video_file.path
            predicted_activity = predict_activity(video_path)
            return render(request, 'video_upload/result.html', {'activity': predicted_activity})
    else:
        form = VideoUploadForm()
    return render(request, 'video_upload/upload.html', {'form': form})
