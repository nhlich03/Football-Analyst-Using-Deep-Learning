import os
import numpy as np
import cv2


from team_classifier import TeamClassifier
from utils.bbox_utils import convert_xywh_to_xyxy

'''
0: player 
1: goalkeeper
2: referee
3: ball
'''

def save_images_to_folder(image_list, folder_path):
    """
    Lưu danh sách ảnh vào thư mục cụ thể.

    Parameters:
        image_list (list): Danh sách các ảnh (numpy.ndarray).
        folder_path (str): Đường dẫn thư mục để lưu ảnh.
    """
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Lưu từng ảnh trong danh sách
    for idx, image in enumerate(image_list):
        file_path = os.path.join(folder_path, f"{idx+1}.jpg")  # Tạo tên file
        success = cv2.imwrite(file_path, image)

def take_all_crop(data_folder, folder_max):
    img_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")

    all_total_crop = 0
    all_success_predict = 0
    folder_count = 0

    if not os.path.exists(img_folder) or not os.path.exists(labels_folder):
        print(f"Không tìm thấy thư mục {img_folder} hoặc {labels_folder}")
        return

    # Xử lý các img
    for img_file in os.listdir(img_folder):
        # Với mỗi img
        img_players_crop = []
        success_count_1 = 0
        success_count_2 = 0
        total_crop = 0
        shift_team = [1,2]
        team1_crop = []
        team2_crop = []

        if img_file.endswith(".jpg"):  # Kiểm tra định dạng ảnh
            img_name = os.path.splitext(img_file)[0]  # Lấy tên tệp không có đuôi
            label_path = os.path.join(labels_folder, f"{img_name}.txt")  # Tạo đường dẫn tệp nhãn

            if os.path.exists(label_path):
                #print(f"Đã tìm thấy nhãn cho {img_file}: {label_path}")
                
                # Đọc ảnh
                img = cv2.imread(os.path.join(img_folder, img_file))
                if img is None:
                    print(f"Không thể đọc ảnh {img_file}")
                    continue
                
                # Lấy crop từ ảnh để fit
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(' ')
                        cls, x, y, w, h = map(float, parts[:-1]) 
                        team_id = parts[-1]
                        if cls != 0:
                            continue
                        bounding_box = [x, y, w, h]
                        x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bounding_box, img.shape)

                        player_crop = img[y_min:y_max, x_min:x_max]
                        player_img_array = np.array(player_crop)
                        img_players_crop.append(player_img_array)
                if len(img_players_crop) == 0:
                    continue
                kmeans = TeamClassifier()
                kmeans.fit_kmeans(list_player_crops = img_players_crop)
                
                # Predict từng crop để tính số lần predict đúng
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(' ')
                        cls, x, y, w, h = map(float, parts[:-1])  
                        team_id = parts[-1]
                        if cls != 0:
                            continue
                        bounding_box = [x, y, w, h]
                        x_min, x_max, y_min, y_max = convert_xywh_to_xyxy(bounding_box, img.shape)

                        player_crop = img[y_min:y_max, x_min:x_max]
                        player_img_array = np.array(player_crop)
                        team_id_predict = kmeans.get_player_team(player_img_array)

                        if team_id == 'left' and team_id_predict == shift_team[0]:
                            success_count_1 += 1
                        elif team_id == 'right' and team_id_predict == shift_team[1]:
                            success_count_1 += 1

                        if team_id == 'left' and team_id_predict == shift_team[1]:
                            success_count_2 += 1
                        elif team_id == 'right' and team_id_predict == shift_team[0]:
                            success_count_2 += 1
                        
                        if team_id_predict == 1:
                            team1_crop.append(player_img_array)
                        elif team_id_predict == 2:
                            team2_crop.append(player_img_array)

                        total_crop += 1

            else:
                print(f"Không tìm thấy nhãn cho {img_file}")
        
        success_count = max(success_count_1, success_count_2)

        print(img_file, total_crop, success_count)
        print(kmeans.kmeans.cluster_centers_[0])
        print(kmeans.kmeans.cluster_centers_[1])
        all_total_crop += total_crop
        all_success_predict += success_count

        out_dir = r'E:\Seminar\Data_sampling_Dataset_3\train\out' 
        folder_team1 = os.path.join(out_dir, img_name, "team1")
        folder_team2 = os.path.join(out_dir, img_name, "team2")

        save_images_to_folder(team1_crop, folder_team1)
        save_images_to_folder(team2_crop, folder_team2)

        folder_count += 1
        if folder_count > folder_max:
            break 
    
    return all_success_predict , all_total_crop

        

data_folder = r'E:\Seminar\Data_sampling_Dataset_3\train'
folder_max = 700
folder_count = 1
all_success_predict , all_total_crop = take_all_crop(data_folder,folder_max)
ans = all_success_predict / all_total_crop
print(f"Tổng số lượng ảnh cắt: {all_total_crop}")
print(f"Tổng số lượng ảnh cắt chia team đúng: {all_success_predict}")
print(f"Tỉ lệ chia team đúng: {ans}")

'''
15 15
15 15
14 14
13 12
14 2
'''

'''
15 15
15 15
14 14
13 1
14 12
'''