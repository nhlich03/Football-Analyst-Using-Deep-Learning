import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(fps,ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


def get_unique_filename(filepath):
    base, extension = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{extension}"
        counter += 1

    return new_filepath 