import cv2
import dlib
import os
import sys
import random

#ouput_image folder
output_dir = 'output_image'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
 
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

#使用dlib自带的frontal_face_detector作为我们的特征提取器 dlib for face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')
index = 1
people = [people for people in os.listdir("db/")]
for people in people:
    images = [image for image in os.listdir("db/" + people + '/')]
    for image in images:    
        # 转为灰度图片
        img = cv2.imread("db/" + people+ '/' +image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(img, 0)
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0           
            face = img[x1:y1,x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))           
            face = cv2.resize(face, (size,size))           
            cv2.imwrite(output_dir+'/'+ people +'/'+str(index)+'.jpg', face)
        index += 1
        print(index)
    print('Finished')