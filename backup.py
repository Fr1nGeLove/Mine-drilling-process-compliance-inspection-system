from io import StringIO
from pathlib import Path

import cv2
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def is_image_file(filename):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(filename)
    return ext.lower() in image_extensions


def is_video_file(filename):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg'}
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions


if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/best.pt', help='model path or trition URL')
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--data", type=str, default= "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.01, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", default=True, action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    print(opt)

    source = ("图片检测", "视频检测")
    # source = "图片检测"
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('开始检测'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        # print(img)
                        # print(str(Path((get_detection_folder()))) + "\\" + img)
                        if is_image_file(img):
                            dir_path = str(Path((get_detection_folder()))) + "\\labels"
                            print(dir_path)
                            for label in os.listdir(dir_path):
                                with open(dir_path + "\\" + label, 'r', encoding='utf-8') as newfile:
                                    for line in newfile:
                                        # print(line)
                                        info = line[::1]
                                        print(info)
                                        nxt_info = [float(num) for num in info]
                                        print(nxt_info)
                                        id, x, y, w, h = nxt_info[0:5]
                                        # print("%d %f %f %f %f" % (id, x, y, w, h))
                                        w /= 2
                                        h /= 2
                                        workimg = cv2.imread(str(Path((get_detection_folder()))) + "\\" + img)
                                        height, width = workimg[:2]
                                        # cv2.rectangle(workimg, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                                        # 框的中心位置转换为左上角坐标
                                        x_center = x * width
                                        y_center = y * height
                                        # 宽度和高度转换为绝对像素值
                                        w_pixels = w * width
                                        h_pixels = h * height
                                        
                                        print(x_center)
                                        print(w_pixels)
                                        # 计算左上角和右下角坐标
                                        top_left_x = int(x_center - w_pixels / 2)
                                        top_left_y = int(y_center - h_pixels / 2)
                                        bottom_right_x = int(x_center + w_pixels / 2)
                                        bottom_right_y = int(y_center + h_pixels / 2)

                                        cv2.rectangle(workimg, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                                        pil_image = Image.fromarray(cv2.cvtColor(workimg, cv2.COLOR_BGR2RGB))
                            st.image(pil_image, use_column_width=True)
                                        # img = workimg
                            # st.image(str(Path((get_detection_folder()))) + "\\" + img)
                                        # cv2.imshow('work Image', workimg)

                    st.balloons()
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if is_video_file(vid):
                            st.video(str(Path(f'{get_detection_folder()}') / vid))

                    st.balloons()
