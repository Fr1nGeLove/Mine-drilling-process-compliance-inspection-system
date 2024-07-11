from io import StringIO
from pathlib import Path

import cv2
import streamlit as st
# import pyttsx3
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
from ComplianceCheck import Invoke_compliance_checks

# engine = pyttsx3.init()

from utils.general import increment_path
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

    st.title('YOLOv5: 基于目标检测技术的矿井打钻过程合规性检查系统')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/best.pt', help='model path or trition URL')
    parser.add_argument("--source", type=str, default="data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--data", type=str, default= "", help="(optional) dataset.yaml path")
    parser.add_argument('--img-size', type=int, default=670, help='inference size (pixels)')
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
    parser.add_argument("--line-thickness", default=1, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=True, action="store_true", help="hide labels")
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

            path = detect(opt)
            if source_index != 0:
                # path = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
                # path = Path('runs\\detect\\exp29\\labels')
                # print(path)
                label_path = os.path.join(path, 'labels')
                Abnormal_events = Invoke_compliance_checks(label_path)
                # print(len(Abnormal_events))
                for vid in os.listdir(get_detection_folder()):
                    if is_video_file(vid):
                        filename, _ = os.path.splitext(vid)
                        img_save_path = os.path.join(path, 'AbnormalImg')
                        os.makedirs(img_save_path, exist_ok=True)
                        if Abnormal_events is None:
                            continue
                        # for frame in Abnormal_events:
                        #     img_path = dir + '\\pic\\' + str(filename) + f'_{frame}' + '.jpg'
                        #     print(img_path)
                        #     img = cv2.imread(img_path, 1)
                        #     if img is None:
                        #         print(f"Failed to read image from {img_path}. Please check the file path and ensure the file exists.")
                        #     else:
                        #         cv2.imwrite(dir + '\\AbnormalImg', img)
                        for frame in Abnormal_events:
                            # 创建路径
                            img_path = os.path.join(path, 'pic', f'{filename}_{frame}.jpg')
                            # print(img_path)
                            img = cv2.imread(img_path, 1)
                            
                            if img is None:
                                print(f"Failed to read image from {img_path}. Please check the file path and ensure the file exists.")
                            else:
                                save_path = os.path.join(img_save_path, f'{filename}_{frame}.jpg')
                                success = cv2.imwrite(save_path, img)
                                
                                # if success:
                                #     print(f"Image saved successfully at {save_path}")
                                # else:
                                #     print(f"Failed to save image at {save_path}")
                # filename, _ = os.path.splitext(vid)
                # dir = str(path)[:-6]
                # for frame in Abnormal_events:
                #     img_path = dir + 'pic/' + f'_{frame}' + '.jpg'
                #     img = cv2.imread(img_path, 1)
                #     img_save_path = dir + 'AbnormalImg'
                #     os.makedirs(img_save_path, exist_ok=True)
                #     cv2.imwrite(vid, img)       
            

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        # print(img)
                        # print(str(Path((get_detection_folder()))) + "\\" + img)
                        if is_image_file(img):
                            # dir_path = str(Path((get_detection_folder()))) + "\\labels"
                            st.image(str(Path(f'{get_detection_folder()}') / img))
                                        # cv2.imshow('work Image', workimg)

                    st.balloons()
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if is_video_file(vid):
                            # st.video(str(Path(f'{get_detection_folder()}') / vid))
                            Abnormal_events_path = os.path.join(path, 'AbnormalImg')
                            Abnormal_Imgs = os.listdir(Abnormal_events_path)
                            Abnormal_Imgs = [f for f in Abnormal_Imgs if f.endswith(('jpg'))]
                            # for ab_imgs in Abnormal_Imgs:
                            #     ab_imgs_path = os.path.join(Abnormal_events_path, ab_imgs)
                            #     st.image(ab_imgs_path, caption=ab_imgs)
                            with st.expander("异常事件图片"):
                                # 使用 columns 创建网格布局
                                cols = st.columns(4)  # 每行显示4张图片
                                for idx, ab_image_file in enumerate(Abnormal_Imgs):
                                    ab_image_path = os.path.join(Abnormal_events_path, ab_image_file)
                                    cols[idx % 4].image(ab_image_path, caption=ab_image_file, use_column_width=True)
                            # 展示文本文件
                            text_path = 'runs/detect/output_log.txt'
                            with open(text_path, 'r', encoding='utf-8') as file:
                                file_content = file.read()
                                st.text_area("文件内容", file_content, height=300)
                    st.balloons()
