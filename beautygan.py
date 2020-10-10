# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import dlib
import time

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def showImg(img,name="img",delay=0):
    cv2.imshow(name, img)
    cv2.waitKey(delay)

def getFaceList(img, gray):
    rects = detector(gray, 1)
    faceList = []
    for rect in rects:
        top = max(0, int(rect.top() - rect.height() * 0.25))
        bottom = min(gray.shape[0], int(rect.bottom() + rect.height() * 0.25))
        left = max(0, int(rect.left() - rect.width() * 0.25))
        right = min(gray.shape[1], int(rect.right() +  rect.width() * 0.25))
        face = img[top:bottom, left:right]
        faceList.append(face)
    return faceList, rects

def getLandmarks(gray, rects):
    landmarkList = []
    for rect in rects:
        landmark = []
        shape = predictor(gray, rect)
        for point in shape.parts():
            landmark.append([point.x, point.y])
        landmarkList.append(landmark)
    landmarkList = np.array(landmarkList)
    return landmarkList

def getMouseMask(img, landmarkList):
    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    mousesPointList = []
    centerList = []
    for i in range(len(landmarkList)):
        landmark = landmarkList[i]
        pointList = np.zeros(shape=(12, 2))
        pointList[:12] = landmark[48:60]
        mousesPointList.append(pointList.astype(np.int64))
        maxX, maxY = pointList.max(0)
        minX, minY = pointList.min(0)
        centerList.append((((minX + maxX) / 2), (minY + maxY) / 2))

    cv2.fillPoly(mask, mousesPointList, 255)
    return mask, centerList

def faceDetect(img, isFull):
    if not isFull:
        scale = max(img.shape[0], img.shape[1]) / 512
        if scale > 1.0:
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            img = cv2.resize(img, (width, height))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faceList, rects = getFaceList(img, gray)
    landmarkList = getLandmarks(img, rects)
    result = {'img':img,'gray':gray,'faceList':faceList,'rects':rects,'landmarkList':landmarkList}
    return result

def makeupCoverImg(resultList, img, rects, landmarkList, faceList, productType):
    img = deprocess(preprocess(img))
    if len(rects) == 0:
        img = cv2.resize(resultList[0][0], (img.shape[1], img.shape[0]))
        img = img * 255
        img = img.astype(np.uint8)
        return img

    faces = np.zeros(shape=img.shape, dtype=img.dtype)
    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(len(rects)):
        rect = rects[i]
        landmark = landmarkList[i]
        top = max(0, int(rect.top() - rect.height() * 0.25))
        bottom = min(img.shape[0], int(rect.bottom() + rect.height() * 0.25))
        left = max(0, int(rect.left() - rect.width() * 0.25))
        right = min(img.shape[1], int(rect.right() +  rect.width() * 0.25))
        face = cv2.resize(resultList[i][0], (right - left, bottom - top))
        # face = cv2.resize(faceList[i], (right - left, bottom - top))
        # face = deprocess(preprocess(face))
        faces[top:bottom, left:right] = face
        mask[top:bottom, left:right] = 255

    faceMask, centerList, center = getFaceMask(img, landmarkList)
    if productType == 'blush':
        mouseMask, _ = getMouseMask(img, landmarkList)
        mask[mouseMask == 255] = 0
    elif productType == 'lipstick':
        mask = faceMask
    # showImg(mask)

    yList, xList = np.where(mask==255)
    center = (int((xList[0] + xList[-1])/2),int((yList[0] + yList[-1])/2))
    # mask[top:bottom, left:right] = 255
    oriImg = img.copy()
    img[(mask==255)] = faces[(mask==255)]
    img = img * 255
    img = img.astype(np.uint8)
    oriImg = oriImg * 255
    oriImg = oriImg.astype(np.uint8)
    img = cv2.seamlessClone(img, oriImg, mask, (int(center[0]), int(center[1])),  cv2.MIXED_CLONE)
    print('img2: ',img.shape)

    # showImg(img)
    return img

img_size = 256
tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

imgDict = {}

# main
img = imread('imgs/no_makeup/gem1.jpg')
isFull = True
result = faceDetect(img, isFull)

img = result['img']
gray = result['gray']
faceList = result['faceList']
rects = result['rects']
landmarkList = result['landmarkList']

X_imgList = np.empty(shape=(0, img_size, img_size, 3))
for face in faceList: 
    face = cv2.resize(face, (img_size, img_size))
    X_imgList = np.append(X_imgList,  [preprocess(face)], axis=0)

if len(faceList) == 0:
    face = cv2.resize(img, (img_size, img_size))
    X_imgList = np.append(X_imgList,  [preprocess(face)], axis=0)

resultList = []
Y_img = cv2.resize(imread('imgs/makeup/XMY-266.png'), (img_size, img_size))
Y_img = np.expand_dims(preprocess(Y_img), 0)

for X_img in X_imgList:
    # Y_img = np.tile(Y_imgDict[str(brand)], (len(faceList), 1, 1, 1))
    Xs_ = sess.run(Xs, feed_dict={X: np.expand_dims(X_img, 0), Y: Y_img})
    Xs_ = deprocess(Xs_)
    resultList.append(Xs_)

img = makeupCoverImg(resultList, img, rects, landmarkList, faceList, productType)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)