import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_x_type_1 = -2.30336538209
min_y_type_1 = -3.40176282264
min_z_type_1 = -0.976451779318
max_x_type_1 = 7.2808711902
max_y_type_1 = 3.9957190444
max_z_type_1 = 4.76762666942

min_x_type_2 = -19.603912
min_y_type_2 = -19.594337
min_z_type_2 = -19.603912
max_x_type_2 = 19.594337
max_y_type_2 = 19.603912
max_z_type_2 = 19.594337

min_values_type1 = torch.tensor([min_x_type_1, min_y_type_1, min_z_type_1], device=device)
max_values_type1 = torch.tensor([max_x_type_1, max_y_type_1, max_z_type_1], device=device)

min_values_type2 = torch.tensor([min_x_type_2, min_y_type_2, min_z_type_2], device=device)
max_values_type2 = torch.tensor([max_x_type_2, max_y_type_2, max_z_type_2], device=device)

files_directory = 'C:/Users/forgedRice/Desktop/Data/unlabeled'

activity_id_mapping = {
    'brushing_teeth': 0,
    'idle': 1,
    'preparing_sandwich': 2,
    'reading_book': 3,
    'stairs_down': 4,
    'stairs_up': 5,
    'typing': 6,
    'using_phone': 7,
    'using_remote_control': 8,
    'walking_freely': 9,
    'walking_holding_a_tray': 10,
    'walking_with_handbag': 11,
    'walking_with_hands_in_pockets': 12,
    'walking_with_object_underarm': 13,
    'washing_face_and_hands': 14,
    'washing_mug': 15,
    'washing_plate': 16,
    'writing': 17}

id_activity_mapping = {
    0: 'brushing_teeth',
    1: 'idle',
    2: 'preparing_sandwich',
    3: 'reading_book',
    4: 'stairs_down',
    5: 'stairs_up',
    6: 'typing',
    7: 'using_phone',
    8: 'using_remote_control',
    9: 'walking_freely',
    10: 'walking_holding_a_tray',
    11: 'walking_with_handbag',
    12: 'walking_with_hands_in_pockets',
    13: 'walking_with_object_underarm',
    14: 'washing_face_and_hands',
    15: 'washing_mug',
    16: 'washing_plate',
    17: 'writing'
}

sensor_mapping = {
    'smartwatch': 0,
    'vicon': 1
}
