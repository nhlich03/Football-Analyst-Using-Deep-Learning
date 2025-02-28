import cv2

video_path = "D:\Seminar Github\VN_LuotDi.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(fps)