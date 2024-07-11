import os
import re
from fastdtw import fastdtw
import numpy as np

# import tkinter as tk
# from tkinter import messagebox

# 读取检测结果所在的文件夹路径
# folder_path = 'C:/Users/PCuser/Documents/Practice/label2/labels'
# folder_path = r'E:\QQ消息记录\644989641\FileRecv\labels'
# 文件存储的路径
# output_file_path = os.path.dirname(os.path.dirname(folder_path)) + "/output_log.txt"
output_file_path = None

def write_to_file(text, title=None):
    # Append the text to the output file
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        if title is not None:
            output_file.write(f"{title}: {text}\n")
        else:
            output_file.write(text + '\n')


def read_detection_results(file_path):
    # 读取包含每帧目标检测信息的文件
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def parse_detection_result(result):
    # 解析每帧中检测到的目标信息
    # 这里假设每一行的格式为 "class_id x_center y_center width height"
    data = result.split()
    if len(data) >= 5:
        class_id = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])
        return class_id, x_center, y_center, width, height
    return None


def getVector(position1, position2):
    pos1 = [position1[0], position1[1]]
    pos2 = [position2[0], position2[1]]
    dir_vector = np.array(pos2) - np.array(pos1)
    return dir_vector


def are_in_same_direction(cur_obj1, cur_obj2, prev_obj1, prev_obj2, threshold_angle=45):
    if prev_obj1 == None or prev_obj2 == None:
        return False
    v1 = getVector(cur_obj1, prev_obj1)
    v2 = getVector(cur_obj2, prev_obj2)
    norm_vector1 = np.linalg.norm(v1)
    norm_vector2 = np.linalg.norm(v2)
    if norm_vector1 != 0 and norm_vector2 != 0:
        angle = angle_with_vectors(v1, v2)
        return angle < threshold_angle
    elif norm_vector1 == 0 and norm_vector2 != 0:
        return False
    elif norm_vector1 != 0 and norm_vector2 == 0:
        return False
    else:
        return True


def angle_with_vectors(vector1, vector2):
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算两个向量的模长
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # 计算余弦值
    cos_theta = np.clip(dot_product / (norm_vector1 * norm_vector2), -1.0, 1.0)
    # if cos_theta > 1 or cos_theta < -1:
    #     print("%f %f %f" % (dot_product, norm_vector1, norm_vector2))

    # 计算角度，注意 np.arccos 返回的是弧度，需要转换为度数
    angle = np.degrees(np.arccos(cos_theta))

    return angle


def has_stopped_moving(prev_position, current_position, threshold=0):
    if prev_position is None:
        return True
    # Check if the target has stopped moving
    distance = ((current_position[0] - prev_position[0]) ** 2 + (current_position[1] - prev_position[1]) ** 2) ** 0.5
    return distance == threshold


def has_moved(prev_position, current_position, threshold=0.0001):
    if prev_position is None:
        return False
    # 简单的目标移动判断逻辑
    distance = ((current_position[0] - prev_position[0]) ** 2 + (current_position[1] - prev_position[1]) ** 2) ** 0.5
    # write_to_file(frame_number,":",distance)
    return distance > threshold


def has_contact_Euclidean(position01, position02, threshold=0.01):
    # 判断两者之间发生接触的逻辑
    distance = ((position01[0] - position02[0]) ** 2 + (position01[1] - position02[1]) ** 2) ** 0.5
    # write_to_file(frame_number,":",distance)
    return distance < threshold


def has_contact(position01, position02, threshold=0.0):
    px, py, pw, ph = position01
    gx, gy, gw, gh = position02

    px1 = px - pw / 2
    px2 = px + pw / 2
    py1 = py - ph / 2
    py2 = py + ph / 2

    gx1 = gx - gw / 2
    gx2 = gx + gw / 2
    gy1 = gy - gh / 2
    gy2 = gy + gh / 2

    parea = (px2 - px1) * (py2 - py1)  # 计算P的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # 计算G的面积

    # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
    x1 = max(px1, gx1)  # 得到左上顶点的横坐标
    y1 = max(py1, gy1)  # 得到左上顶点的纵坐标
    x2 = min(px2, gx2)  # 得到右下顶点的横坐标
    y2 = min(py2, gy2)  # 得到右下顶点的纵坐标

    # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
    # w = max(0, (x2 - x1))  # 相交矩形的长，这里用w来表示
    # h = max(0, (y1 - y2))  # 相交矩形的宽，这里用h来表示
    # print("相交矩形的长是：{}，宽是：{}".format(w, h))
    # 这里也可以考虑引入if判断
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0

    area = w * h
    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)
    return IoU >= threshold


# Initialize global variables
kelly_bar_moving = False
kelly_bar_start_frame = None
kelly_bar_end_frame = None
drill_worker_position = None
prev_kelly_bar_position = None
prev_hand_position = None
drill_machine_moving = False
drill_machine_B_moving = False
drill_machine_I_moving = False
prev_drill_machine_B_position = None
prev_drill_machine_I_position = None
prev_hammer_position = None

kelly_bar_moving_frame = None  # 钻杆移动帧数
drill_machine_moving_frame = None  # 钻机移动帧数
drill_machine_connect_kelly_bar_frame = None  # 钻机连接钻杆帧数
Worker_unload_kelly_bar_frame = None  # 工人卸下钻杆
Worker_place_kelly_bar_frame = None  # 工人放置钻杆
kelly_bar_moving_frame_range = []  # 记录钻杆移动帧数
drill_machine_moving_frame_range = []  # 记录钻机移动帧数
drill_machine_connect_kelly_bar_frame_range = []  # 记录钻机连接钻杆的帧数
worker_unload_kelly_bar_frame_range = []  # 记录工人卸下钻杆帧数
Worker_place_kelly_bar_frame_range = []  # 记录工人放置钻杆帧数
first_hammer = False
now_event = []
Abnormal_events = []
has_abnormal = False

def analyze_events(frame_number, detection_results):
    # 在这里实现事件分析逻辑
    global kelly_bar_moving, kelly_bar_start_frame, kelly_bar_end_frame, prev_kelly_bar_position, \
        prev_drill_worker_position, prev_hand_position, drill_machine_B_moving, drill_machine_I_moving, \
        drill_machine_moving, prev_drill_machine_B_position, prev_drill_machine_I_position, kelly_bar_moving_frame, drill_machine_moving_frame, \
        drill_machine_connect_kelly_bar_frame_range, worker_unload_kelly_bar_frame_range, Worker_place_kelly_bar_frame_range, prev_hammer_position, first_hammer, has_abnormal

    # Initialize positions for the current frame
    current_drill_worker_position = None
    current_kelly_bar_position = None
    current_hand_position = None
    current_hammer_position = None
    current_drill_machine_B_position = None
    current_drill_machine_I_position = None

    for result in detection_results:
        # 解析每帧中检测到的目标信息
        parsed_result = parse_detection_result(result)

        if parsed_result:
            class_id, x_center, y_center, width, height = parsed_result
            current_target_position = (x_center, y_center, width, height)

            # "0worker工人","1kelly_bar钻杆","2hand·手","3hammer锤子","4paper纸","5drilling_machine-B钻机B","6drilling_machine-l钻机l"

            # 判断工人是否移动
            if class_id == 0:  # 0 是工人的类别 ID
                current_drill_worker_position = current_target_position

            # 判断钻杆是否移动
            if class_id == 1:  # 1 是钻杆的类别 ID
                current_kelly_bar_position = current_target_position

            # 判断手是否移动
            elif class_id == 2:  # 2 是手的类别 ID
                current_hand_position = current_target_position

            elif class_id == 3:  # 2 是锤子的类别 ID
                current_hammer_position = current_target_position

            # 判断钻机B是否移动
            elif class_id == 5:  # 假设 5 是钻机B的类别 ID
                current_drill_machine_B_position = current_target_position

            # 判断钻机I是否移动
            elif class_id == 6:  # 假设 6 是钻机I的类别 ID
                current_drill_machine_I_position = current_target_position

    tag = False
    # 事件1：工人操作钻机推进
    # 过程特点：钻杆不存在，钻机I发生移动, 钻机B开始时未出现，出现后和钻机I保持一致运动
    if current_kelly_bar_position is None:
        if (
                current_drill_machine_I_position is not None
                and prev_drill_machine_I_position is not None
                and (current_hand_position is None or (
                current_hand_position is not None and not has_contact(current_drill_machine_I_position,
                                                                      current_hand_position, 0.01)))
        ):
            if (
                    # has_moved(prev_drill_machine_B_position, current_drill_machine_B_position, threshold=0.001)
                    # and
                    has_moved(prev_drill_machine_I_position, current_drill_machine_I_position, threshold=0.00001) and ((
                    current_drill_machine_B_position is not None and are_in_same_direction(
                current_drill_machine_I_position, current_drill_machine_B_position,
                prev_drill_machine_I_position, prev_drill_machine_B_position)
                    or current_drill_machine_B_position is None)
            )
            ):
                # drill_machine_moving_frame = frame_number
                # write_to_file(f"In frame {frame_number}, Worker operates the drilling rig to advance!")
                # drill_machine_moving_frame_range.append(drill_machine_moving_frame)
                now_event.append(1)
                tag = True
                Abnormal_events.append(frame_number)

    #
    # 事件2：钻机连接钻杆
    # 过程特点：第一次出现钻杆，手与钻机I发生接触 或者是钻杆第一次出现
    if (
            prev_drill_machine_I_position is not None
            and current_drill_machine_I_position is not None
            and current_hand_position is not None
    ):
        if are_in_same_direction(current_hand_position, current_drill_machine_I_position, prev_hand_position,
                                 prev_drill_machine_I_position, threshold_angle=90) and has_contact(
            current_drill_machine_I_position, current_hand_position, 0.001):
            # if has_contact_matrix(current_drill_machine_I_position, current_hand_position, 0.01):
            # drill_machine_connect_kelly_bar_frame = frame_number
            # write_to_file(f"In frame {frame_number},Drilling rig connecting drill pipe")
            # drill_machine_connect_kelly_bar_frame_range.append(drill_machine_connect_kelly_bar_frame)
            now_event.append(2)
            tag = True

    # 事件3：钻机带动钻杆向外抽出
    # 过程特点：钻机B与钻杆保持一致运动方向，钻机I被遮盖
    if current_kelly_bar_position is not None and prev_kelly_bar_position is not None and current_drill_machine_I_position is None and has_moved(prev_kelly_bar_position, current_kelly_bar_position, threshold=0.01):
        # 钻机位置发生变化
        if ( (current_drill_worker_position is not None and not has_contact_Euclidean(current_drill_worker_position, current_kelly_bar_position, threshold= 0.15)) or
                current_drill_worker_position is None 
            ):
            # 检测钻机带动钻杆向外抽出过程中：1、是否有工人2、工人与钻杆之间是否存在交互(接触)
            # if (
            #         current_drill_worker_position is None
            #         or not has_contact_Euclidean(current_drill_worker_position, current_kelly_bar_position,
            #                                      threshold=0.1)
            # ):
            #     # 钻机带动钻杆向外抽出
            #     if not kelly_bar_moving:
            #         # kelly_bar_moving_frame = frame_number
            #         write_to_file(f"In frame {frame_number},Drilling rig drives the drill rod to be pulled out outward")
            #         kelly_bar_moving = True
            #         now_event.append(3)
            #         tag = True
            # 新增事件1：钻机带动钻杆向外抽出过程中，判断手是否与钻杆发生接触
            if prev_hand_position is not None and current_drill_worker_position is not None and has_contact_Euclidean(current_drill_worker_position, current_kelly_bar_position, threshold=0.1) :
                if has_contact(prev_hand_position, current_kelly_bar_position, threshold=0.1):
                    write_to_file(
                        f"In frame {frame_number},在钻机驱动下拔出钻杆的过程中，存在人工接触钻杆、工人操作不当的危险!")
                    Abnormal_events.append(frame_number)
                    # has_abnormal = True
            else :
                # write_to_file(f"In frame {frame_number},Drilling rig drives the drill rod to be pulled out outward")
                kelly_bar_moving = True
                now_event.append(3)
                tag = True

    # 钻杆停止移动
    # if kelly_bar_moving and current_kelly_bar_position is not None and has_stopped_moving(prev_kelly_bar_position,
    #                                                                                       current_kelly_bar_position,
    #                                                                                       threshold=0):
    #     # kelly_bar_moving_frame = frame_number
    #     kelly_bar_moving = False
    #     kelly_bar_moving_frame_range.append(kelly_bar_moving_frame)

    # 事件4：工人卸下钻杆
    # 事件特点：1、手与钻杆发生接触 2、若出现锤子目标则与钻杆接触

    if (
            current_kelly_bar_position is not None
            and current_drill_worker_position is not None
            # and current_drill_machine_B_position is not None
            and current_hand_position is not None
    ):
        if (
                current_hammer_position is not None and (
                has_contact(current_hammer_position, current_kelly_bar_position, threshold=0.0001)
                or has_contact_Euclidean(current_hammer_position, current_kelly_bar_position, threshold=0.5))
        ):
            # Worker_unload_kelly_bar_frame = frame_number
            # write_to_file(f"In frame {frame_number},Worker unload drill pipes!")
            # worker_unload_kelly_bar_frame_range.append(Worker_unload_kelly_bar_frame)
            now_event.append(4)
            tag = True
            if current_hammer_position is not None:
                first_hammer = True

    # 事件5：工人放置钻杆
    if (
            first_hammer is True
            and current_drill_worker_position is not None
            and current_hammer_position is None
            and current_kelly_bar_position is not None
            and are_in_same_direction(current_drill_worker_position, current_kelly_bar_position,
                                      prev_drill_worker_position, prev_kelly_bar_position, 45)
            # 判断两者是否向同一方向进行移动，及是否有相同的运动趋势
    ):
        # Worker_place_kelly_bar_frame = frame_number
        # write_to_file(f"In frame {frame_number},Worker place drill pipes！")
        # Worker_place_kelly_bar_frame_range.append(Worker_place_kelly_bar_frame)
        now_event.append(5)
        tag = True

    if tag is False:
        now_event.append(-1)

    prev_drill_machine_B_position = current_drill_machine_B_position
    prev_drill_machine_I_position = current_drill_machine_I_position
    prev_kelly_bar_position = current_kelly_bar_position
    prev_hand_position = current_hand_position
    prev_drill_worker_position = current_drill_worker_position
    prev_hammer_position = current_hammer_position
    return now_event

def get_file(folder_path):
    # 获取文件夹中所有以 '.txt' 结尾的文件
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    res_event = []
    # 按照文件名中的数字排序文件列表
    # sorted_files = sorted(txt_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    sorted_files = sorted(txt_files,
                        key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

    # 遍历排序后的文件列表
    for filename in sorted_files:
        file_path = os.path.join(folder_path, filename)
        # 从文件名中提取帧的顺序信息，这里使用正则表达式
        match = re.search(r'\d+', filename)
        if match:
            frame_number_str = match.group()
            if frame_number_str:
                frame_number = int(frame_number_str)
                # write_to_file(frame_number)

                # 读取包含每帧目标检测信息的文件
                detection_results = read_detection_results(file_path)
                # print(detection_results)
                # 调用 analyze_events 函数进行事件分析
                res_event.extend(analyze_events(frame_number, detection_results))
            else:
                write_to_file(f"The number in the file name {filename} is empty")
        else:
            write_to_file(f"No valid digits found in file name {filename}")
    return res_event



# Helper function to create an event dictionary
def create_event_range(frame_range, description):
    return {"range": frame_range, "description": description}

def sliding_window(now_event):

    window_size = 90
    count = [0, 0, 0, 0, 0, 0]
    dominant_event = []

    # print(now_event)
    # print(dict[1])
    for i in range(0, len(now_event)):
        if now_event[i] != -1:
            count[now_event[i]] += 1
        if i >= window_size - 1:
            dominant_event.append(np.argmax(count))
            if now_event[i - window_size + 1] != -1:
                count[now_event[i - window_size + 1]] -= 1

            if dominant_event is not None:
                if dominant_event[-1] == 1:
                    drill_machine_moving_frame_range.append(i)
                elif dominant_event[-1] == 2:
                    drill_machine_connect_kelly_bar_frame_range.append(i)
                elif dominant_event[-1] == 3:
                    kelly_bar_moving_frame_range.append(i)
                elif dominant_event[-1] == 4:
                    worker_unload_kelly_bar_frame_range.append(i)
                elif dominant_event[-1] == 5:
                    Worker_place_kelly_bar_frame_range.append(i)

# print(Worker_place_kelly_bar_frame_range)
# Check if the lists are not empty before trying to access their elements
def print_per_frame():
    # write_to_file("*************************异常事件记录******************************")
    has_abnormal = False
    events = []
    if drill_machine_moving_frame_range:
        # events.append(create_event_range([drill_machine_moving_frame_range[0], drill_machine_moving_frame_range[-1]],
        #                                 "Worker operate drilling rigs to advance"))
        events.append(create_event_range([drill_machine_moving_frame_range[0], drill_machine_moving_frame_range[-1]],
                                        "工人操作钻杆推进"))

    if drill_machine_connect_kelly_bar_frame_range:
        # events.append(create_event_range(
        #     [drill_machine_connect_kelly_bar_frame_range[0], drill_machine_connect_kelly_bar_frame_range[-1]],
        #     "Drilling rig connecting drill pipe"))
        events.append(create_event_range(
            [drill_machine_connect_kelly_bar_frame_range[0], drill_machine_connect_kelly_bar_frame_range[-1]],
            "钻机连接钻杆"))

    if kelly_bar_moving_frame_range:
        # events.append(create_event_range([kelly_bar_moving_frame_range[0], kelly_bar_moving_frame_range[-1]],
        #                                 "Drilling rig drives the drill rod to be pulled out outward"))
        events.append(create_event_range([kelly_bar_moving_frame_range[0], kelly_bar_moving_frame_range[-1]],
                                        "钻机带动钻杆向外抽出"))

    if worker_unload_kelly_bar_frame_range:
        # events.append(create_event_range([worker_unload_kelly_bar_frame_range[0], worker_unload_kelly_bar_frame_range[-1]],
        #                                 "Worker unload drill pipes"))
        events.append(create_event_range([worker_unload_kelly_bar_frame_range[0], worker_unload_kelly_bar_frame_range[-1]],
                                        "工人卸下钻杆"))

    if Worker_place_kelly_bar_frame_range:
        # events.append(create_event_range([Worker_place_kelly_bar_frame_range[0], Worker_place_kelly_bar_frame_range[-1]],
        #                                 "Worker place drill pipes"))
        events.append(create_event_range([Worker_place_kelly_bar_frame_range[0], Worker_place_kelly_bar_frame_range[-1]],
                                        "工人放置钻杆"))
        if len(Worker_place_kelly_bar_frame_range)>50:
            write_to_file("工人可能已经闲置了很长时间，这可能导致被动工作,请注意!")
    
    return events

# print(events)
# Sort the events based on the starting frame of each event
def sort_event(events):
    sorted_events = sorted(events, key=lambda x: x["range"][0])
    # Sort the events based on Sliding Window
    # print(sorted_events)
    # Create a list to save events without frame ranges
    event_list = []
    for event in sorted_events:
        frame_range = event["range"]
        description = event["description"]
        if frame_range and len(frame_range) >= 2:
            event_list.append(
                f"Event{sorted_events.index(event) + 1}: {description} ({frame_range[0]}frame - {frame_range[-1]}frame)")
        else:
            event_list.append(f"Event{sorted_events.index(event) + 1}: {description}")
    return sorted_events, event_list

# Create a list to save event descriptions
def compliance_check(sorted_events, event_list):
    description_list = [event['description'] for event in sorted_events]

    # Check if the lists are not empty before trying to access their elements
    # for event in event_list:
    #     write_to_file(event)

    write_to_file("************************合规检查******************************")

    # print the list of event descriptions
    write_to_file(description_list, "识别事件链")

    predefined_events = ['工人操作钻杆推进', '钻机连接钻杆',
                        '钻机带动钻杆向外抽出',
                        '工人卸下钻杆', '工人放置钻杆']
    write_to_file(predefined_events, "预定义事件链:")
    return predefined_events, description_list

def event_maching(predefined_events, description_list):
    write_to_file("*************************事件匹配******************************")

    # Create a mapping from strings to numerical values
    element_mapping = {
        "工人操作钻杆推进": 0,
        "钻机连接钻杆": 1,
        "钻机带动钻杆向外抽出": 2,
        "工人卸下钻杆": 3,
        "工人放置钻杆": 4
    }

    # Convert sequences to numerical representation
    predefined_events_sequence = [element_mapping[element] for element in predefined_events]
    description_list_sequence = [element_mapping[element] for element in description_list]
    return predefined_events_sequence, description_list_sequence


# Function to calculate the absolute difference between two elements
def element_difference(x, y):
    return abs(x - y)

def DTW_dis(predefined_events_sequence, description_list_sequence, predefined_events, description_list):

    # Calculate DTW distance with a custom distance function
    distance, path = fastdtw(predefined_events_sequence, description_list_sequence)

    # Output the results
    write_to_file(f"预定义事件链:{predefined_events}")
    write_to_file(f"识别事件链:{description_list}")
    # write_to_file(f"DTW Distance: {distance}")

    # Output the matched elements with events
    matched_elements = [(predefined_events[i], description_list[j]) for i, j in path]
    write_to_file("事件匹配:")
    write_to_file("预定义事件 匹配 识别事件:")
    for element_pair in matched_elements:
        write_to_file(f"{element_pair[0]} <匹配> {element_pair[1]}")

    # Check if sequences have the same length
    same_length = len(predefined_events_sequence) == len(description_list_sequence)

    # Check if sequences have the same order
    same_order = all(predefined_events_sequence[i] == description_list_sequence[j] for i, j in path)

    write_to_file("*******************匹配********************")
    # Output the results
    if same_length:
        write_to_file("匹配结果识别的事件链与预定义的事件链长度相同")
    else:
        write_to_file(f"标识的事件链与预定义的事件链长度不相等\n"
                    f"识别事件链长度为{len(description_list_sequence)}， 而预定义事件链长度为{len(predefined_events_sequence)}")

        if len(description_list_sequence) < len(predefined_events_sequence):
            missing_elements_list1 = [elem for elem in predefined_events if elem not in description_list]
            # Find elements present in list2 but not in list1
            # missing_elements_list2 = [elem for elem in description_list if elem not in predefined_events]
            if missing_elements_list1:
                write_to_file(f"识别事件链中缺少的步骤: {missing_elements_list1}")

        if len(description_list_sequence) > len(predefined_events_sequence):
            write_to_file(
                "钻杆卸载过程增加了一个不必要的步骤，这可能会带来风险，请检查!")
            adding_steps = [description_list[i] for i in
                            range(len(description_list_sequence), len(predefined_events_sequence))]
            write_to_file(f"添加的步骤: {adding_steps}")

    if same_order:
        write_to_file("识别的事件链与预定义的事件链事件序列一致")
    else:
        write_to_file(
            "识别的事件链与预定义的事件链顺序不一致，工人在推钻杆过程中可能存在危险!")

        # Find positions where the order is different and convert to events
        different_order_positions = [(description_list[j], predefined_events[i]) for i, j in path if
                                    predefined_events_sequence[i] != description_list_sequence[j]]

        write_to_file("不一致的序列:")
        for position_pair in different_order_positions:
            write_to_file(f"{position_pair[0]}(识别事件) 与 {position_pair[1]}(预定义事件) 不匹配")

    if same_order and same_length:
        write_to_file("操作合规")

def Invoke_compliance_checks(folder_path):
    global output_file_path
    output_file_path = os.path.dirname(os.path.dirname(folder_path)) + "/output_log.txt"
    os.remove(output_file_path)
    with open(output_file_path, "w") as file:
        pass  # 创建一个新的空文件
    res_event = get_file(folder_path)
    sliding_window(res_event)
    events = print_per_frame()
    sorted_events, event_list = sort_event(events)
    predefined_events, description_list = compliance_check(sorted_events, event_list)
    predefined_events_sequence, description_list_sequence = event_maching(predefined_events, description_list)
    DTW_dis(predefined_events_sequence, description_list_sequence, predefined_events, description_list)
    print_path()
    return Abnormal_events

def print_path():
    print("Print statements have been saved to", output_file_path)
# 读取检测结果所在的文件夹路径
# folder_path = 'C:/Users/PCuser/Documents/Practice/label2/labels'
# folder_path = r'E:\QQ消息记录\644989641\FileRecv\labels'
# 文件存储的路径
# output_file_path = os.path.dirname(os.path.dirname(folder_path)) + "/output_log.txt"

# solve('runs\\detect\\exp28\\labels')