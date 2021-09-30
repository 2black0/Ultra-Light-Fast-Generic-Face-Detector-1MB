"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2
import time
from djitellopy import Tello
import logging

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"
sizes = 40

def main():
    parser = argparse.ArgumentParser(description='detect_video')

    parser.add_argument('--net_type', default="RFB", type=str, help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', default=480, type=int, help='define network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.7, type=float, help='score threshold')
    parser.add_argument('--candidate_size', default=1000, type=int, help='nms candidate size')
    parser.add_argument('--fps_count', default=False, type=bool, help='fps counter')
    parser.add_argument('--path', default="imgs", type=str, help='imgs dir')
    parser.add_argument('--test_device', default="cpu", type=str, help='cuda:0 or cpu')
    parser.add_argument('--cam_source', type=int, default=2, help='99 for Tello or 0-3 for webcam')
    parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str, help='path of video')
    args = parser.parse_args()

    input_img_size = args.input_size
    define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

    net_type = args.net_type

    # cap = cv2.VideoCapture(args.video_path)  # capture from video
    cap = cv2.VideoCapture(args.cam_source)
    cap.set(3, 640)
    cap.set(4, 480)

    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    test_device = args.test_device

    candidate_size = args.candidate_size
    threshold = args.threshold

    if net_type == 'slim':
        #model_path = "models/pretrained/version-slim-320.pth"
        model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    elif net_type == 'RFB':
        #model_path = "models/pretrained/version-RFB-320.pth"
        model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)

    timer = Timer()
    sum = 0

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            print("end")
            break

        if args.fps_count is True:
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(orig_image, fps, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        interval = timer.end()
        #print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        
        '''
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #print('box[0]=', int(box[0]))
            label = f" {probs[i]:.2f}"
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
            cv2.putText(orig_image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        '''
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #print('box[0]=', int(box[0]))
            label = f" {probs[i]:.2f}"
            cent_X = int(640/2)
            cent_Y = int(480/2)
            cent_box_X = int((int(box[3])-int(box[1]))/2)+int(box[0])
            cent_box_Y = int((int(box[2])-int(box[0]))/2)+int(box[1])
            #print(cent_X, cent_Y, cent_box_X, cent_box_Y)
            x=int(box[0])
            y=int(box[1])
            h=int(box[2])-int(box[0])
            w=int(box[3])-int(box[1])
            error = calculateError(cent_X, cent_Y, sizes, cent_box_X, cent_box_Y, int(box[2])-int(box[0]))
            
            '''
            x=int(box[0])
            y=int(box[1])
            x+h=int(box[2])
            h=int(box[2])-int(box[0])
            y+h=int(box[3])
            w=int(box[3])-int(box[1])
            '''
            #cv2.rectangle(orig_image, (x,y), (x+h, y+h), (0, 255, 0), 2)
            cv2.circle(orig_image, (cent_X, cent_Y), 2, (0, 0, 255), 2)
            cv2.rectangle(orig_image, (int(cent_X-sizes), int(cent_Y-sizes)), (int(cent_X+sizes), int(cent_Y+sizes)), (0, 255, 0), 2)
            cv2.circle(orig_image, (cent_box_X, cent_box_Y), 2, (0, 255, 0), 2)
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
            cv2.putText(orig_image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.line(orig_image, (320, 240), (int((w/2)+x), int((h/2)+y)), (0, 0, 255), 2)
            #cv2.line(orig_image, (320, 240), (int((int(box[3])-int(box[1]))/2)+int(box[0])), (int((int(box[2])-int(box[0]))/2)+int(box[1]))), (0, 0, 255), 2)

            
        orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
        #sum += boxes.size(0)
        cv2.imshow('Object Detection', orig_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def calculateError(centerX, centerY, centerH, centerBoxX, centerBoxY, boxH):
    error = [round((centerX - centerBoxX) / centerX * 100), #error X
             round((centerY - centerBoxY) / centerY * 100), #error Y
             round((centerH - boxH/2) / centerH * 100) #error Z
            ]
    #limit the max error to 100
    a = 0
    for i in error:
        if i >= 100:
            error[a] = 100
        if i <= -100:
            error[a] = 100
        a += 1
    return error

if __name__=="__main__":
    main()