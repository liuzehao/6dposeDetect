#coding=utf-8
'''
@Descripttion: 
@version: 1.10
@Author: HaoMax
@Date: 2020-06-22 16:17:10
@LastEditors: HaoMax
@LastEditTime: 2020-06-22 16:24:42
'''
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from nets.resnet_v1 import resnetv1
from utils.timer import Timer
import torch
import cv2
import os
import re
import numpy as np
from MeshPly import Meshply
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def get_type(class_path,class_name):
    restr='[0-9a-zA-Z]'+'+\.'+class_name
    findtxt = re.compile(restr)
    s=os.listdir(class_path)
    s=" ".join(s)
    s=findtxt.findall(s)[0]
    return s
def read_file(path):
    fid = open(path, 'r')
    f_s =fid.readlines()
    fid.close()
    return f_s
def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 320
    K[1, 1], K[1, 2] = 573.5704, 240
    K[2, 2] = 1.
    return K
def compute_projection(points3d, R, K):
    projections_2d = np.zeros((2, points3d.shape[1]), float)
    camera_projection = (K.dot(R)).dot(points3d)
    projections_2d[0,:] = camera_projection[0,:]/ camera_projection[2,:]
    projections_2d[1,:] = camera_projection[1,:]/ camera_projection[2,:]
    return projections_2d
def show2d(im_name,proj_2d_pr):
    img = cv2.imread(im_name)
    l=len(proj_2d_pr[0])
    for i in range(l):
        y=proj_2d_pr[1][i]
        x=proj_2d_pr[0][i]    
        x=int(x)
        y=int(y)
        if x>640:
            x=639
        if x<0:
            x=0
        if y>480:
            y=479
        if y<0:
            y=0
        print(x, y)
        img[y,x,0]=255
        img[y,x,1]=255
        img[y,x,2]=255
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()#liu change
def get_all_files(bg_path):
    files = []
    for f in os.listdir(bg_path):
        if os.path.isfile(os.path.join(bg_path, f)):
            files.append(os.path.join(bg_path, f))
        else:
            files.extend(get_all_files(os.path.join(bg_path, f)))
    files.sort(key=lambda x: int(x[-9:-4]))
    return files
def othertowpont(center,point):
    if point[0]<center[0]:
        x=2*(center[0]-point[0])+point[0]
        if point[1]<center[1]:
            y=point[1]-2*(center[1]-point[1])
        else:
            y=point[1]+2*(center[1]-point[1])
    else:
        x=point[0]-2*(point[0]-center[0])
        if point[1]<center[1]:
            y=point[1]+2*(center[1]-point[1])
        else:
            y=point[1]-2*(point[1]-center[1])
    return (x,y)

def otherpoint(point_c,point_z,point_y):
    x1,y1=othertowpont(point_c,point_z)
    x2,y2=othertowpont(point_c,point_y)
    return ((x1,y1),(x2,y2))
def show_result_3points_glue(cls,bbox,frontp1, frontp2, fcenterp, backp1, backp2, bcenterp,img):
    x1 = frontp1[0]
    y1 = frontp1[1]
    dw1 = frontp1[2]
    dh1 = frontp1[3]

    cx1 = fcenterp[0]
    cy1 = fcenterp[1]

    x3 = frontp2[0]
    y3 = frontp2[1]
    dw2 = frontp2[2]
    dh2 = frontp2[3]

    cv2.circle(img, (x1, y1), 5, (0, 0, 255), 2)
    cv2.circle(img, (cx1, cy1), 5, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 5, (0, 0, 255), 2)

    if cx1 > x1:
        x4 = x1 + dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    else:
        x4 = x1 - dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    if cx1 > x3:
        x2 = x3 + dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2
    else:
        x2 = x3 - dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2
    x2=int(x2)
    x4=int(x4)
    y2=int(y2)
    y4=int(y4)
    x5 = backp1[0]
    y5 = backp1[1]
    dw5 = backp1[2]
    dh5 = backp1[3]

    cx2 = bcenterp[0]
    cy2 = bcenterp[1]

    x7 = backp2[0]
    y7 = backp2[1]
    dw6 = backp2[2]
    dh6 = backp2[3]

    cv2.circle(img, (x5, y5), 5, (0, 255, 255), 2)
    cv2.circle(img, (cx2, cy2), 5, (0, 255, 255), 2)
    cv2.circle(img, (x7, y7), 5, (0, 255, 255), 2)
    if cx2 > x5:
        x8 = x5 + dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    else:
        x8 = x5 - dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    if cx2 > x7:
        x6 = x7 + dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    else:
        x6 = x7 - dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    pr_points = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], float)
    return pr_points
def show_result_3points(cls,bbox,frontp1, frontp2, fcenterp, backp1, backp2, bcenterp,img):
    x1 = frontp1[0]
    y1 = frontp1[1]
    dw1 = frontp1[2]
    dh1 = frontp1[3]

    cx1 = fcenterp[0]
    cy1 = fcenterp[1]

    x3 = frontp2[0]
    y3 = frontp2[1]
    dw2 = frontp2[2]
    dh2 = frontp2[3]

    cv2.circle(img, (x1, y1), 5, (0, 0, 255), 2)
    cv2.circle(img, (cx1, cy1), 5, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 5, (0, 0, 255), 2)
    (x4,y4),(x2,y2)=otherpoint((cx1, cy1),(x1, y1),(x3, y3))
    x2=int(x2)
    x4=int(x4)
    y2=int(y2)
    y4=int(y4)
    x5 = backp1[0]
    y5 = backp1[1]
    dw5 = backp1[2]
    dh5 = backp1[3]

    cx2 = bcenterp[0]
    cy2 = bcenterp[1]

    x7 = backp2[0]
    y7 = backp2[1]
    dw6 = backp2[2]
    dh6 = backp2[3]

    cv2.circle(img, (x5, y5), 5, (0, 255, 255), 2)
    cv2.circle(img, (cx2, cy2), 5, (0, 255, 255), 2)
    cv2.circle(img, (x7, y7), 5, (0, 255, 255), 2)
    (x8,y8),(x6,y6)=otherpoint((cx2, cy2),(x5, y5),(x7, y7))
    x8=int(x8)
    x6=int(x6)
    y8=int(y8)
    y6=int(y6)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)


    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    pr_points = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], float)

    return pr_points
def show_result_8points(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
    img = cv2.imread(img)    
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
def show_result_8points_test(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
  
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.circle(img, (x1, y1), 1, (0, 0, 255),4)
    cv2.putText(img,'1',(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'2',(x2, y2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'3',(x3, y3),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'4',(x4, y4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'5',(x5, y5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'6',(x6, y6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'7',(x7, y7),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'8',(x8, y8),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 0, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 0, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 0, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 0, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 0, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.line(img, (x1, y1), (x5, y5), (255, 0, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 0, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 0, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 0, 0), 1)
def show_result_8points(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
        
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)

    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)

   
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    cv2.putText(img, '1', (x1, y1), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '2', (x2, y2), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '3', (x3, y3), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '4', (x4, y4), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '5', (x5, y5), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '6', (x6, y6), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '7', (x7, y7), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '8', (x8, y8), font, 1.2, (255, 255, 255), 2)
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()#liu change
    return img
def show_result_8points_lunwen(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
        
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)

    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 2)
    cv2.line(img, (x2, y2), (x4, y4), (255, 0, 0), 2)
    cv2.line(img, (x3, y3), (x4, y4),(255, 0, 0), 2)
    cv2.line(img, (x5, y5), (x6, y6),(255, 0, 0), 2)
    cv2.line(img, (x5, y5), (x7, y7),(255, 0, 0), 2)
    cv2.line(img, (x6, y6), (x8, y8),(255, 0, 0), 2)
    cv2.line(img, (x7, y7), (x8, y8),(255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.line(img, (x1, y1), (x5, y5), (255, 0, 0), 2)
    cv2.line(img, (x2, y2), (x6, y6), (255, 0, 0), 2)
    cv2.line(img, (x3, y3), (x7, y7), (255, 0, 0), 2)
    cv2.line(img, (x4, y4), (x8, y8),(255, 0, 0), 2)
    return img
def show_result_8points_lunwen_gt(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(img, (x1, y1), (x3, y3), (0, 255, 0), 2)

    cv2.line(img, (x2, y2), (x4, y4), (0, 255, 0), 2)

    cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 2)

    cv2.line(img, (x5, y5), (x6, y6), (0, 255, 0), 2)

    cv2.line(img, (x5, y5), (x7, y7), (0, 255, 0), 2)
    cv2.line(img, (x6, y6), (x8, y8), (0, 255, 0), 2)
    cv2.line(img, (x7, y7), (x8, y8), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.line(img, (x1, y1), (x5, y5), (0, 255, 0), 2)
    cv2.line(img, (x2, y2), (x6, y6), (0, 255, 0), 2)
    cv2.line(img, (x3, y3), (x7, y7), (0, 255, 0), 2)
    cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 2)
    return img
def show_result_8points_nocenter(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
    img = cv2.imread(img)    
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.line(img, (x1, y1), (x3, y3), (0, 255, 0), 1)
    cv2.line(img, (x2, y2), (x4, y4), (0, 255, 0), 1)
    cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 1)
    cv2.line(img, (x5, y5), (x6, y6), (0, 255, 0), 1)
    cv2.line(img, (x5, y5), (x7, y7), (0, 255, 0), 1)
    cv2.line(img, (x6, y6), (x8, y8), (0, 255, 0), 1)
    cv2.line(img, (x7, y7), (x8, y8), (0, 255, 0), 1)

    cv2.line(img, (x1, y1), (x5, y5), (0, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (0, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (0, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 1)
    return img
def demo(net, image_name):
    #print(image_name)
    im = cv2.imread(image_name)
    timer = Timer()
    timer.tic()
    scores, boxes, front_2_1_points, front_2_2_points, front_center, back_2_1_points, back_2_2_points, back_center= im_detect(net, im)
    timer.toc()
    thresh = 0.75  # CONF_THRESH
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    cntr = -1
    
    prs_points=[]#这里存放预测的点
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #cls就是物体的类别
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_front_2_1_points = front_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_2_2_points = front_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_center = front_center[:, 2*cls_ind:2*(cls_ind + 1)]
        cls_back_2_1_points = back_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_2_2_points = back_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_center = back_center[:, 2*cls_ind:2*(cls_ind + 1)]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #这里是极大值抑制
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        front_2_1_points_det = cls_front_2_1_points[keep.numpy(), :]
        front_2_2_points_det = cls_front_2_2_points[keep.numpy(), :]
        front_center_det = cls_front_center[keep.numpy(), :]
        back_2_1_points_det = cls_back_2_1_points[keep.numpy(), :]
        back_2_2_points_det = cls_back_2_2_points[keep.numpy(), :]
        back_center_det = cls_back_center[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        inds = [0]
        if len(inds) == 0:
            continue
        else:
            cntr += 1
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            frontp1 = front_2_1_points_det[i, :]
            frontp2 = front_2_2_points_det[i, :]
            fcenterp = front_center_det[i, :]
            brontp1 = back_2_1_points_det[i, :]
            brontp2 = back_2_2_points_det[i, :]
            bcenterp = back_center_det[i, :]
            img = cv2.imread(image_name)
            pr_points = show_result_3points_glue(cls,bbox,frontp1, frontp2, fcenterp, brontp1, brontp2, bcenterp,img)
            prs_points.append(pr_points)
    return prs_points
def get_3D_corners(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [min_x, min_y, max_z],#2
                        [min_x, max_y, min_z],#3
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, min_z],#5
                        [max_x, min_y, max_z],#6
                        [max_x, max_y, min_z],#7
                        [max_x, max_y, max_z]])#8
    return corners
def get_3D_corners_test(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, max_y, min_z],#3
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, min_z],#5
                        [max_x, min_y, max_z]])#4
    return corners
def get_3D_corners3(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [max_x, min_y, min_z],#5
                        [min_x, min_y, max_z],#2
                        [max_x, min_y, max_z],#6
                        [min_x, max_y, min_z],#3
                        [max_x, max_y, min_z],#7
                        [min_x, max_y, max_z],#4
                        [max_x, max_y, max_z]])#8
    return corners
def get_3D_corners3_test(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, max_z],#2
                        [max_x, min_y, max_z],#6
                        [min_x, max_y, min_z],#3
                        [max_x, max_y, min_z]])#7
    return corners
def get_3D_corners2(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [min_x, max_y, min_z],#3
                        [max_x, min_y, min_z],#5
                        [max_x, max_y, min_z],#7
                        [min_x, min_y, max_z],#2
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, max_z],#6    
                        [max_x, max_y, max_z]])#8
    return corners
def test_RT(t_gt):
    Tx=t_gt[0]
    Ty=t_gt[1]
    Tz=t_gt[2]
    K=get_camera_intrinsic()
    Cx=K[0][0]*(Tx/Tz)+320
    Cy=K[1][1]*(Ty/Tz)+240
    return (Cx,Cy)
def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)
###路径###
xyz_path_base='../data/OcclusionData/models/'#换成xyz模型
CLASSES = ('__background__',
           'ape','can','cat','driller','duck','eggbox','glue','holepuncher')#liu change
color_list=[(0,255,0),(0,0,255),(255,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0),(255,255,255)]
pose_path='../data/OcclusionData/poses/'
label_linemod_path_base='../data/linemodocculution/'
im_path="../data/OcclusionData/RGB-D/rgb_noseg_all"#遮挡数据集路径
ply_path_base='../data/ply/'#模型基本路径
# model_path='../output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_320000.pth'
model_path='../output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_300000.pth'
if __name__ == '__main__':
    ims_path=get_all_files(im_path)#读入所有测试图片
    errs_2d_ape = []
    errs_3d_ape = []
    errs_2d_can = []
    errs_3d_can = []
    errs_2d_cat = []
    errs_3d_cat = []
    errs_2d_driller = []
    errs_3d_driller = []
    errs_2d_duck = []
    errs_3d_duck = []
    errs_2d_eggbox = []
    errs_3d_eggbox = []
    errs_2d_glue = []
    errs_3d_glue = []
    errs_2d_holepuncher = []
    errs_3d_holepuncher = []
    errs_2d=[errs_2d_ape,errs_2d_can,errs_2d_cat,errs_2d_driller,errs_2d_duck,errs_2d_eggbox,errs_2d_glue,errs_2d_holepuncher]
    errs_3d=[errs_3d_ape,errs_3d_can,errs_3d_cat,errs_3d_driller,errs_3d_duck,errs_3d_eggbox,errs_3d_glue,errs_3d_holepuncher]
    meshlist=[]
    #先把模型读到内存中
    for z in range(1,len(CLASSES)):
        ply_path=ply_path_base+CLASSES[z]+'.ply'
        mesh = Meshply(ply_path)
        meshlist.append(mesh)
    net = resnetv1(num_layers=101)
    saved_model = model_path
    net.create_architecture(9, tag='default', anchor_scales=[8, 12, 16])  # class 7 #fang liu 9
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()
    for im_name in ims_path:
        print(im_name)
        flag_preerro=0
        erro_num=0
        img_name=im_name[-9:-4]
        prs_points = demo(net, im_name)
        img = cv2.imread(im_name)
        for flag_u in range(1,len(CLASSES)):
            mesh = meshlist[flag_u-1]
            vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
            img_name_label = '%06d' % int(img_name)
            label_linemod_path=label_linemod_path_base+CLASSES[flag_u]+'/'+img_name_label+'.txt'
            f=open(label_linemod_path,'r')
            line=f.readline()
            line=line.split()
            label_linemod_points=line[3:19]
            label_linemod_points=np.array(map(float,label_linemod_points))
            label_linemod_points[::2]=label_linemod_points[::2]*640
            label_linemod_points[1::2]=label_linemod_points[1::2]*480

            for zz in range(len(label_linemod_points[::2])):
                if label_linemod_points[::2][zz]>640 or label_linemod_points[1::2][zz]>480:
                    erro_num=erro_num+1
                    print("problem picture is:",im_name)
                    print("problem num is：",CLASSES[flag_u])
                    flag_preerro=1
                    continue
            if flag_preerro==1:
                continue#跳过整张图片的代码---->跳过单个物体
            
            gt_points=label_linemod_points.reshape(8,2)
            K=get_camera_intrinsic()
            pr_points = prs_points[flag_u-1].reshape(8,2)
            img_gt=show_result_8points_lunwen_gt(gt_points[0][0],gt_points[0][1],gt_points[1][0],gt_points[1][1],gt_points[2][0],gt_points[2][1],gt_points[3][0],gt_points[3][1],gt_points[4][0],gt_points[4][1],gt_points[5][0],gt_points[5][1],gt_points[6][0],gt_points[6][1],gt_points[7][0],gt_points[7][1],img.copy())
            img_show=show_result_8points_lunwen(pr_points[0][0],pr_points[0][1],pr_points[1][0],pr_points[1][1],pr_points[2][0],pr_points[2][1],pr_points[3][0],pr_points[3][1],pr_points[4][0],pr_points[4][1],pr_points[5][0],pr_points[5][1],pr_points[6][0],pr_points[6][1],pr_points[7][0],pr_points[7][1],img_gt.copy())
            imgname='show/'+CLASSES[flag_u]+"_"+img_name+'.jpg'
            print(imgname)
            cv2.imwrite(imgname,img_show)
            corners3D_pr = get_3D_corners3(vertices)
            corners3D_gt = get_3D_corners3(vertices)
            _, R_pre, t_pre = cv2.solvePnP(corners3D_pr, pr_points, K, None)
            R_mat_pre, _ = cv2.Rodrigues(R_pre)

            _, R_gt, t_gt = cv2.solvePnP(corners3D_gt, gt_points, K, None)
            R_mat_gt, _ = cv2.Rodrigues(R_gt)
            Cx,Cy=test_RT(t_gt)
            Rt_gt = np.concatenate((R_mat_gt, t_gt), axis=1)
            Rt_pr = np.concatenate((R_mat_pre, t_pre), axis=1)
            corners3D = get_3D_corners(vertices)
            center_point=(corners3D[7]+corners3D[0])/2
            center_point=np.insert(center_point,3,[1],axis=0).reshape(-1,1)

            proj_center = compute_projection(center_point,Rt_gt,K)
            Cx,Cy=test_RT(t_gt)
            try:
                proj_2d_gt = compute_projection(vertices,Rt_gt,K)
                proj_2d_pr = compute_projection(vertices,Rt_pr, K)
            except IndexError as e:
                print("pre erro:",img_name)
                flag_preerro=1
                break
            norm = np.linalg.norm(proj_2d_gt - proj_2d_pr, axis=0)
            #print("norm:",norm)
            pixel_dist = np.mean(norm)
            errs_2d[flag_u-1].append(pixel_dist)
            transform_3d_gt = compute_transformation(vertices, Rt_gt)
            transform_3d_gt=np.array(transform_3d_gt)
            #transform_3d_gt[0]=-transform_3d_gt[0]
            transform_3d_pr = compute_transformation(vertices, Rt_pr)
            transform_3d_pr=np.array(transform_3d_pr)
            norm3d          = np.linalg.norm(transform_3d_gt - transform_3d_pr, axis=0)
            vertex_dist     = np.mean(norm3d)
            # print("vertex3D_dist",vertex_dist)#显示3d结果
            errs_3d[flag_u-1].append(vertex_dist)
        if flag_preerro==1:
            continue
    px_threshold = 5.
    eps = 1e-5
    diam = [0.103,0.202,0.155,0.262,0.109,0.176364,0.176,0.162]
    for u in range(len(errs_3d)):
        print("num of every class:",len(errs_2d[u]))
        error_count = len(np.where(np.array(errs_2d[u]) > px_threshold)[0])
        acc = len(np.where(np.array(errs_2d[u-1]) <= px_threshold)[0]) * 100.0 / (len(errs_2d[u])+eps)
        acc3d10     = len(np.where(np.array(errs_3d[u]) <= diam[u] * 0.1)[0]) * 100. / (len(errs_3d[u])+eps)
        print('Test finish! Object is:',CLASSES[u+1])
        print('Acc using {} px 2D projection = {:.2f}%'.format(px_threshold, acc))
        print('Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam[u] * 0.1, acc3d10))
