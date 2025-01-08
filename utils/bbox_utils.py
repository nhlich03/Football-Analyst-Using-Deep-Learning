def convert_xywh_to_xyxy(bounding_box, frame_shape):
    x, y, w, h = bounding_box
    frame_height, frame_width, _ = frame_shape
    x_min = int((x - w / 2) * frame_width)
    x_max = int((x + w / 2) * frame_width)
    y_min = int((y - h / 2) * frame_height)
    y_max = int((y + h / 2) * frame_height)
    return x_min, x_max, y_min, y_max

def get_distance(player_box, ball_box, frame_shape):
    px_min, px_max, py_min, py_max = convert_xywh_to_xyxy(player_box, frame_shape)
    bx_min, bx_max, by_min, by_max = convert_xywh_to_xyxy(ball_box, frame_shape)

    # Calculate center of the ball
    ball_center_x = (bx_min + bx_max) / 2
    ball_center_y = (by_min + by_max) / 2

    # Calculate left_foot and right_foot of the player
    left_foot = px_max 
    right_foot = px_min 

    left_distance = euclidean_distance(ball_center_x, ball_center_y, left_foot, py_max)
    right_distance = euclidean_distance(ball_center_x, ball_center_y, right_foot, py_max)

    distance = min(left_distance,right_distance)
    
    return distance


def euclidean_distance(x1, y1, x2, y2):
    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    return distance