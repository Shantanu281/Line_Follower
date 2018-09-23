import numpy as np
import time
import cv2
import math
# alpha is angle of rotation through which we should rotate our vehicle
# negative alpha means anti-clockwise rotation and vice-versa
#
#def initialize():

def vehicle_orientation(strip_angle,strip_cen_x,strip_cen_y,mask):
    global alpha,i,velocity,dist,delta,k
    dx = strip_cen_x-frame_cen_x
    dy = strip_cen_y-frame_cen_y
    centre = int(strip_cen_x),int(strip_cen_y)
    dist=math.sqrt(dx**2+dy**2)
    r=math.sqrt(dx**2+strip_cen_y**2)
    k=-np.sign(dx)
    if abs(dx) > dx_threshold or strip_angle > strip_angle_threshold:
        if mask[frame_cen_x,0] == 255:
            alpha=strip_angle*k          #just turn vehicle by strip_angle
        elif mask[frame_cen_x,frame_cen_y] == 255:
            alpha=-strip_angle*k
        elif mask[frame_cen_x,0] != 255 and mask[frame_cen_x,frame_cen_y] != 255 and i!=1:
            delta=delta_max*dist*k/dist_max
            alpha=strip_angle*k+delta   #rotate by angle greater than angle of line to proceed in directi++++on of line initially
            i=1                     #line to proceed in direction of line initially
        else:
            delta2=(delta-delta_max*dist*k/dist_max)*k
            alpha=delta2
    elif strip_angle < strip_angle_threshold and abs(dx) < dx_threshold:
        alpha=0
    # now defining velocity according to our angle
    if velocity>vel_max:
        velocity=vel_max
    else:
        velocity= ((vel_min-vel_max))*alpha/alpha_max+vel_max
    #x2 = int(strip_cen_x + r*math.cos(alpha))
    #y2 = int(r*math.sin(alpha))
    #mask = cv2.line(mask,(int(frame_cen_x),frame_cen_y*2),(int(strip_cen_x),int(strip_cen_y)),(255,255,255),5)
    #mask = cv2.circle(mask ,centre , 5, (0, 0, 0), -1)
    cv2.imshow('image',mask)
    #cv2.imshow('imag',frame)
    k=cv2.waitKey(5)
    if k==27:
        return True


#cap = cv2.VideoCapture('D:\\Videos\\V_20180918_160701_vHDR_On.mp4')
cap=cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
frame_cen_x=int(cap.get(3)/2)
frame_cen_y=int(cap.get(4)/2)
dx_threshold = 5
alpha_def=0
vel_min=0
vel_max=100
delta_max=25
velocity = vel_max
alpha_max=25
alpha=0
i=0
dist=0
delta=0
k=1
strip_angle_threshold=2
dist_max=math.sqrt(frame_cen_x**2+frame_cen_y**2)
lower_white=np.array([0, 0, 200],dtype=np.uint8)
upper_white=np.array([180, 255, 255],dtype=np.uint8)
#lower_white=np.array([100, 150, 0],dtype=np.uint8)
#upper_white=np.array([140, 255, 255],dtype=np.uint8)
cap.set(5,15)
time.sleep(0.5)
# processing video
while(True):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_white,upper_white)
    #mask=cv2.bitwise_and(frame,frame,mask=mask)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.erode(mask,kernel,iterations=3)
    mask=cv2.dilate(mask,kernel,iterations=4)
    abc,mask=cv2.threshold(mask,25,255,cv2.THRESH_BINARY)
    image,contours,heirarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    rect=cv2.minAreaRect(cnt)
    strip_cen_x=rect[0][0]
    strip_cen_y=rect[0][1]
    [vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    strip_angle=(math.degrees(math.atan(vx/vy)))
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    print(strip_angle)
    mask=cv2.drawContours(mask,[box],0,(255,255,255),1)
    if vehicle_orientation(strip_angle,strip_cen_x,strip_cen_y,mask):
        cap.release()
        cv2.destroyAllWindows()
    else:
        continue
