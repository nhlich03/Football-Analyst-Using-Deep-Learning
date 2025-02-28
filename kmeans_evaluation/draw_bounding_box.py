import cv2
import os

def draw_labels_on_image(image_path, label_path, output_path):
    """
    Gộp nhãn vào ảnh bằng cách vẽ bounding box và thông tin nhãn.

    Parameters:
        image_path (str): Đường dẫn tới file ảnh.
        label_path (str): Đường dẫn tới file nhãn.
        output_path (str): Đường dẫn lưu ảnh đã gộp nhãn.
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    # Đọc file nhãn
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file nhãn: {label_path}")
        return

    height, width, _ = image.shape  # Kích thước ảnh
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls, x, y, w, h, *rest = parts
            x, y, w, h = map(float, [x, y, w, h])

            # Chuyển đổi từ tọa độ YOLO (x_center, y_center, width, height) sang (x_min, y_min, x_max, y_max)
            x_min = int((x - w / 2) * width)
            y_min = int((y - h / 2) * height)
            x_max = int((x + w / 2) * width)
            y_max = int((y + h / 2) * height)

            # Lấy thông tin đội (nếu có)
            team = rest[-1] if rest else "None"
            color = (0, 255, 0) if team == "left" else (0, 0, 255) if team == "right" else (255, 255, 0)

            # Vẽ bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # Ghi thông tin nhãn
            label = f"Class {cls} | Team: {team}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Lưu ảnh đã gộp nhãn
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Đã lưu ảnh với nhãn tại: {output_path}")

# Ví dụ sử dụng
image_path = r"E:\Seminar\Data_sampling_Dataset_3\train\images\\1102000001.jpg"
label_path = r"E:\Seminar\Data_sampling_Dataset_3\train\labels\\1102000001.txt"
output_path = r"E:\Seminar\Data_sampling_Dataset_3\train\output_img.jpg"

draw_labels_on_image(image_path, label_path, output_path)
