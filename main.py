import os
import cv2
from ultralytics import YOLO

from detection import Detection
from team_classifier import TeamClassifier
from process_video import ProcessVideo
from utils.video_utils import get_unique_filename

def main():
    # Init
    video_path =   "video/SNGS-118.mp4"
    video_path2 = "video/video_MU_30s.avi"
    video_path2 = "video/video_MU_2s.avi"
    ball_model_path = "model/ball_model.pt"
    player_model_path = "model/best_yolo8x_model.pt"
    output_video_path = "processed_video.avi"
    yolo_results_dir = os.path.join(os.getcwd(), "yolo_results", "detection_results")

    # Get unique directory and output file name
    yolo_results_dir = get_unique_filename(yolo_results_dir)
    output_video_path = os.path.join(yolo_results_dir, output_video_path)

    # Detection
    detection = Detection()
    detection.player_model = YOLO(player_model_path)
    detection.ball_model = YOLO(ball_model_path)
    detection.take_crop = 1  # Save player crop

    detection.detect_video(video_path=video_path,
                           output_results_path=yolo_results_dir)
    print(f"All results saved to: {yolo_results_dir}")

    # Team classification
    team_classifier = TeamClassifier()
    all_player_crops = detection.all_player_crop
    team_classifier.fit_kmeans(all_player_crops)

    # Process video with ball possession analysis
    processor_video = ProcessVideo()
    processor_video.team_classifier = team_classifier
    processor_video.threshold = 70

    processor_video.process_video(video_path, output_video_path, yolo_results_dir)

    print(len(processor_video.output_frames))
    print(f"Processed video saved at: {output_video_path}")


if __name__ == "__main__":
    main()