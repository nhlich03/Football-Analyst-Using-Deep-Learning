import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self):
        """
        Initialize with paths for player crops, YOLO results, and video.

        Args:
            all_player_crops_dir (str): Directory path for all player cropped images.
            labels_dir (str): Directory path where YOLO results are stored.
            video_path (str): Path to the video file.
        """ 
        self.all_player_color = []
        self.team_colors = {}
        self.kmeans = None 
    
    def fit_kmeans(self, list_player_crops):
        self.get_all_player_colors(list_player_crops)
        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.all_player_color)
        
    def get_all_player_colors(self, list_player_crops):
        """
        Collect colors from all cropped player images.

        Returns:
            list: List of dominant colors for each player crop.
        """
        for player in list_player_crops:
            color = self.get_player_color(player)
            self.all_player_color.append(color)

    def get_player_color(self, image):
        """
        Extract the dominant color from an image, assuming the player's jersey is in the center.

        Args:
            image (numpy.ndarray): Image to analyze.

        Returns:
            numpy.ndarray: Dominant color in LAB color space.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h, w, _ = lab.shape
        #center = lab[h//4:3*h//4, w//4:3*w//4]
        center = lab[2*h//10:8*h//10, 1*w//10:9*w//10]
        return np.mean(center, axis=(0, 1))
    
    def get_player_team(self, player):
        player_color = self.get_player_color(player)
        team_id = self.kmeans.predict([player_color])[0]+1
        return team_id