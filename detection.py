import os
import numpy as np

from utils.video_utils import read_video
from utils.bbox_utils import convert_xywh_to_xyxy


class Detection:
    def __init__(self):
        self.player_model = None
        self.ball_model = None
        self.all_player_crop = []
        self.take_crop = 0

    def detect_video(self, video_path, output_results_path=os.path.join(os.getcwd(), "detect_results")):
        video_frames = read_video(video_path)
        for frame_num, frame in enumerate(video_frames):
            frame_dir = os.path.join(output_results_path, f"frame_{frame_num}")
            os.makedirs(frame_dir, exist_ok=True)
            self.detect_frame(frame, frame_dir)

    def detect_frame(self, frame, output):
        self.player_model.predict(
            frame,
            conf=0.8,
            save_txt=True,
            save_conf=True,
            project=output,
            name="player",
            classes=[1, 2, 3],
            verbose=False
        )
        self.ball_model.predict(
            frame,
            conf=0.1,
            save_txt=True,
            save_conf=True,
            project=output,
            name="ball",
            verbose=False
        )

        # Keep only the highest confidence ball detection
        ball_file_path = os.path.join(output, "ball", "labels", "image0.txt")
        player_file_path = os.path.join(output, "player", "labels", "image0.txt")
        self.keep_highest_ball_conf(ball_file_path, player_file_path)

        if self.take_crop:
            labels_crop_path = os.path.join(output, "player", "labels", "image0.txt")
            if os.path.exists(labels_crop_path):
                self.take_all_player_crop(frame, labels_crop_path)

    def take_all_player_crop(self, frame, player_crop_dir):
        with open(player_crop_dir, 'r') as file:
            for line in file:
                parts = line.strip().split()
                cls, x, y, w, h, conf = map(float, parts)
                bounding_box = [x, y, w, h]

                if cls == 2:
                    x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bounding_box, frame.shape)
                    player_crop = frame[y_min:y_max, x_min:x_max]
                    player_img_array = np.array(player_crop)
                    self.all_player_crop.append(player_img_array)

    def keep_highest_ball_conf(self, ball_file_path, player_file_path):
        """
        Check if a file has two or more lines, then keep only the line with the highest confidence score.

        Args:
            ball_file_path (str): Path to the file containing ball detection results.
            player_file_path (str): Path to the file where to append the highest confidence ball detection.

        Returns:
            None
        """
        if not os.path.exists(ball_file_path):
            return

        highest_conf_line = ""
        highest_conf = -1

        with open(ball_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 5:  # Ensure there's a confidence score
                    conf = float(parts[5])
                    if conf > highest_conf:
                        highest_conf = conf
                        highest_conf_line = line.strip()

        # Write the line with the highest confidence to the player file
        with open(player_file_path, 'a') as output_file:
            if highest_conf_line:  # Ensure there's a line to write
                output_file.write(highest_conf_line + '\n')



'''# Init
video_path = "video/video.mp4"
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

for crop in detection.all_player_crop:
    print(crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)'''

# team_classifier = TeamClassifier()
# all_player_crops = detection.all_player_crop
# team_classifier.fit_kmeans(all_player_crops)
