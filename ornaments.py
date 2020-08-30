import numpy as np
import cv2
from mtcnn import MTCNN
import json
ori_img = cv2.imread('data/gem2.jpg')
img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
detector = MTCNN()
ori_mask = cv2.imread('data/Md4q20mbvj.png', cv2.IMREAD_UNCHANGED)
# y_scale = ori_mask.shape[0] / ori_img.shape[0]
# x_scale = ori_mask.shape[1] / ori_img.shape[1]
#reshape为(x,y)数组
# src_pts = np.array([[626, 538], [1352, 539], [983, 561], [986, 723]])
# dst_pts = np.array([[626/x_scale, 538/y_scale], [1352/x_scale, 539/y_scale], [983/x_scale, 561/y_scale], [986/x_scale, 723/y_scale]])
# dst_pts = np.array([[74, 73], [1352/x_scale, 539/y_scale], [983/x_scale, 561/y_scale], [986/x_scale, 723/y_scale]])
# dst_pts = np.array([[74, 73], [108, 65], [90, 70], [92, 82]])
src_pts = np.float32([[626, 538], [1352, 539], [986, 950]])
videoCapture = cv2.VideoCapture('data/桃园三结义.mp4')
#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# 写
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout_1 = cv2.VideoWriter()
vout_1.open('data/output.mp4',fourcc,fps,size,True)
success, frame = videoCapture.read()
while success :
    result = frame.copy()
    result_json = detector.detect_faces(frame)

    for i in range(len(result_json)):
        
        left_eye = result_json[i]['keypoints']['left_eye']
        right_eye = result_json[i]['keypoints']['right_eye']
        nose = result_json[i]['keypoints']['nose']

        dst_pts = np.float32([[left_eye[0], left_eye[1]], [right_eye[0], right_eye[1]],
        [nose[0], nose[1]]])


        #用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
        # M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        M = cv2.getAffineTransform(src_pts, dst_pts)
        #使用转换矩阵M计算出img1在img2的对应形状
        h,w = frame.shape[:2]
        # M_r = np.linalg.inv(M)
        mask = cv2.warpAffine(ori_mask, M, (w,h))
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gap = mask[:,:,3]==255
        result[gap] = mask[gap][:,:3]
    vout_1.write(result)
    cv2.imshow('result', result)
    cv2.waitKey(1)
    success, frame = videoCapture.read() #获取下一帧

vout_1.release()