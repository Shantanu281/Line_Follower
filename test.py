import numpy as np
import time
import cv2
import math
# alpha is angle of rotation through which we should rotate our vehicle
# negative alpha means anti-clockwise rotation and vice-versa
#
def vehicle_orientation(strip_angle,strip_cen_x,strip_cen_y):
    global alpha,i,velocity
    dx=strip_cen_x-frame_cen_x
    dy=strip_cen_y-frame_cen_y
    dist=math.sqrt(dx**2+dy**2)
    if dx>5:
        if mask[frame_cen_x,frame_cen_y] == 255 or mask[frame_cen_x,0] == 255:
            alpha=strip_angle          #just turn vehicle by strip_angle
            i=0
        elif i == 0:
            delta=delta_max*dx/320
            alpha=strip_angle+delta   #rotate by angle alpha initially
            i=1
        else:
            delta=delta_max*dist/320
            alpha=alpha-delta
    elif dx<-5:
        if mask[frame_cen_x,frame_cen_y] == 255 or mask[frame_cen_x,0] == 255:
            alpha=-strip_angle          #just turn vehicle by strip_angle
            i=0
        elif i == 0:
            delta=delta_max*dx/320
            alpha=-(strip_angle+delta)   #rotate by angle alpha initially and then move along it
            i=1
        else:
            delta=delta_max*dist/320
            alpha=alpha+alpha_def    # and then change alpha in ratio of distance
            i=1                      # to configure our vehicle to specific angle
    else:
        alpha=0
    # now defining velocity according to our angle
    if velocity>vel_max:
        velocity=vel_max
    else:
        velocity= ((vel_min-vel_max))/alpha_max*alpha+vel_max
    print(alpha,end='')
        #feed alpha and velocity in microcontroller
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
frame_cen_x=320
frame_cen_y=240
alpha_def=0
vel_min=0
vel_max=100
delta_max=25
velocity = vel_max
alpha_max=25
alpha=0
i=0
lower_white=np.array([0, 0, 200],dtype=np.uint8)
upper_white=np.array([180, 255, 255],dtype=np.uint8)
cap.set(5,15)
time.sleep(0.5)
# processing video
while(True):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #ret,mask=cv2.threshold(image_gray,25,255,cv2.THRESH_BINARY)
    mask=cv2.inRange(hsv,lower_white,upper_white)
    #mask=cv2.bitwise_and(frame,frame,mask=mask)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.erode(mask,kernel,iterations=5)
    mask=cv2.dilate(mask,kernel,iterations=3)
    abc,mask=cv2.threshold(mask,25,255,cv2.THRESH_BINARY)
    image,contours,heirarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[0]
    rect=cv2.minAreaRect(cnt)
    strip_angle=90-abs(rect[2])
    strip_cen_x=rect[0][0]
    strip_cen_y=rect[0][1]
    #dx=320-strip_cen_x
    #dy=240-strip_cen_y
    vehicle_orientation(strip_angle,strip_cen_x,strip_cen_y)
    #print(90-abs(rect[2]))
    #print(image.shape)
    #print(rect[0][1])
    #print(rect[0][0])
    #print(rect[1][0],rect[1][1])
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    #image=cv2.bitwise_not(image)
    image=cv2.drawContours(mask,[box],0,(255,255,255),1)
    cv2.imshow('image',mask)
    k=cv2.waitKey(5)
    if k==27:
        cap.release()
        cv2.destroyAllWindows()
