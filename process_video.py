import cv2
import os
import numpy as np

from team_classifier import TeamClassifier
from utils.bbox_utils import convert_xywh_to_xyxy, get_distance
from utils.video_utils import read_video


class ProcessVideo:
    def __init__(self):
        self.colors = {
            0: (0, 0, 255),  # Ball - đỏ
            2: (0, 255, 0),  # Player - xanh lá
            3: (255, 0, 0),  # Referee - xanh dương
            4: (255, 255, 0),  # Team1 Vàng
            5: (255, 0, 255)  # Team2 Hồng
        }
        self.team_keeping_ball = None
        self.pass_count = {'Team1': 0, 'Team2': 0}
        self.possession_count = {'Team1': 0, 'Team2': 0}
        self.team_classifier = None
        self.threshold = 100 
        self.output_frames = []

    def process_video(self, video_path, output_video_path, detect_results_dir):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        video_frames = read_video(video_path)
        for frame_num, frame in enumerate(video_frames):
            frame_dir = os.path.join(detect_results_dir, f"frame_{frame_num}")
            labels_file_path = os.path.join(frame_dir, "player", "labels", "image0.txt")

            if not os.path.exists(labels_file_path):
                continue

            processed_frame = self.process_frame(frame, labels_file_path)
            self.output_frames.append(processed_frame)
            out.write(processed_frame)

    def process_frame(self, frame, labels_file_path):
        ball_box = None
        team1_players = []
        team2_players = []

        with open(labels_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                cls, x, y, w, h, conf = map(float, parts)
                bounding_box = [x, y, w, h]
                object_type = cls

                # If player, then split to team1 and team2 
                if cls == 2:
                    x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bounding_box, frame.shape)
                    player_crop = frame[y_min:y_max, x_min:x_max]
                    player_img_array = np.array(player_crop)
                    team_id = self.team_classifier.get_player_team(player_img_array)

                    if team_id == 1:
                        team1_players.append(bounding_box)
                        object_type = 4
                    elif team_id == 2:
                        team2_players.append(bounding_box)
                        object_type = 5

                elif cls == 0:
                    ball_box = bounding_box

                # Get color for bounding box
                color = self.get_color(object_type)

                # Draw bounding box to frame
                #self.draw_bbox(frame, bounding_box, conf, color)
                self.draw_ellipse_player(frame, bounding_box, color)

                # If ball exist, assign ball to player and draw ball connection
                if ball_box is not None:
                    self.assign_ball_player(frame, ball_box, team1_players, team2_players, self.threshold)

                # Draw possession
                self.draw_possession_info(frame, self.possession_count)

        return frame

    def draw_bbox(self, frame, bounding_box, conf, color):
        x1, x2, y1, y2 = convert_xywh_to_xyxy(bounding_box, frame.shape)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)

        # Show confidence index
        conf_text = f'{conf:.2f}'
        cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)

    def draw_possession_info(self, frame, possession_count):
        """
        Draw possession information on the frame.

        :param frame: Current frame of the video.
        :param possession_count: Dictionary containing the number of ball possessions for each team.
        """
        total = sum(possession_count.values())
        team1_percent = possession_count['Team1'] / total * 100 if total > 0 else 0
        team2_percent = possession_count['Team2'] / total * 100 if total > 0 else 0
        
        team_keeping_ball_number = "None"
        if self.team_keeping_ball == 4:
            team_keeping_ball_number = "1"
        elif self.team_keeping_ball == 5:
            team_keeping_ball_number = "2"

        # Get colors for Team1 and Team2
        team1_color = self.get_color(4)
        team2_color = self.get_color(5)
        team_keeping_ball_color = self.get_color(self.team_keeping_ball)

        # Coordinates
        x = frame.shape[1] - 500
        
        # Vẽ thông tin 2 team 
        cv2.putText(frame, f"Team1 possession: {team1_percent:.1f}%",
                    (x, 50),  # Adjust y-coordinate for two lines
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team1_color, 2)
        cv2.putText(frame, f"Team2 possession: {team2_percent:.1f}%",
                    (x, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team2_color, 2)
        cv2.putText(frame, f"Team keeping ball: {team_keeping_ball_number}",
                    (x - 700, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, team_keeping_ball_color, 2)
    
    def assign_ball_player(self, frame, ball_box, team1_players, team2_players, threshold):
        min_distance = float('inf')
        team_id = -1
        assigned_player_box = None

        # Get min distance for all player vs ball
        for player_box in team1_players:
            distance = get_distance(player_box, ball_box, frame.shape)
            if distance < min_distance:
                min_distance = distance
                team_id = 4
                assigned_player_box = player_box

        for player_box in team2_players:
            distance = get_distance(player_box, ball_box, frame.shape)
            if distance < min_distance:
                min_distance = distance
                team_id = 5
                assigned_player_box = player_box

        # If min distance from player to ball less or equal threshold then player is keeping ball 
        if min_distance <= threshold:
            # Draw traingle for player who is keeping ball
            color = self.get_color(team_id)
            self.draw_traingle_player(frame, assigned_player_box, color)

            # Assign team of that player is keeping ball
            self.team_keeping_ball = team_id

        if self.team_keeping_ball == 4:
            self.possession_count['Team1'] += 1
        elif self.team_keeping_ball == 5:
            self.possession_count['Team2'] += 1 

    def draw_traingle_player(self, frame, player_box, color):
        x_min, x_max, y_min, _ = convert_xywh_to_xyxy(player_box, frame.shape)

        x = (x_min + x_max) / 2
        y = y_min

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ], dtype=np.float32).astype(np.int32)

        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 3)

        return frame

    def draw_ellipse_player(self,frame,bbox,color):
            x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bbox, frame.shape)
            
            width = int((y_max - y_min) * 0.6)
            x_center = int((x_min + x_max) / 2)

            cv2.ellipse(
                frame,
                center=(x_center,y_max),
                axes=(int(width), int(0.35*width)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color = color,
                thickness=2,
                lineType=cv2.LINE_4
            )

    def get_color(self, object_type):
        color = self.colors.get(object_type, (255, 255, 255))
        return color
    