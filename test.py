import numpy as np
import time
import cv2
import math
# alpha is angle of rotation through which we should rotate our vehicle
# negative alpha means anti-clockwise rotation and vice-versa
#
#def initialize():
def vehicle_orientation(strip_angle,mask,m,c,k):
    global alpha,velocity
    #dx = strip_cen_x-frame_cen_x
    #dy = strip_cen_y-frame_cen_y
    centre = int(frame_cen_x),int(strip_cen_y)
    dist=abs((frame_y-m*frame_cen_x-c)/math.sqrt(1+m**2))
    #r=math.sqrt(dx**2+strip_cen_y**2)
    #k=-np.sign(dx)
    #if
    if dist > dist_threshold or strip_angle > strip_angle_threshold:
        alpha=strip_angle+k*delta_max*dist/dist_max
    elif strip_angle < strip_angle_threshold and dist < dist_threshold:
        alpha=0
    # now defining velocity according to our angle
    if velocity>vel_max:
        velocity=vel_max
    else:
        velocity= ((vel_min-vel_max))*alpha/alpha_max+vel_max
    x2 = int(frame_cen_x + dist*math.sin(alpha))
    y2 = int(frame_y-dist*math.cos(alpha))
    #mask = cv2.line(mask,(int(frame_cen_x),int(frame_cen_y*2)),(x2,y2),(255,255,255),5)
    #mask = cv2.circle(mask ,centre , 5, (0, 0, 0), -1)
    #cv2.putText(mask,'al:{0:<3.2f} '.format(alpha),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),1)
    print(m,dist,alpha,strip_angle,k)
    cv2.imshow('image',mask)
    #cv2.imshow('imag',frame)
    k=cv2.waitKey(5)
    if k==27:
        return True
#cap = cv2.VideoCapture('D:\\Videos\\V_20180918_160701_vHDR_On.mp4')
cap=cv2.VideoCapture(0)
frame_x=int(cap.get(3))
frame_y=int(cap.get(4))
frame_cen_x=int(frame_x/2)
frame_cen_y=int(frame_y/2)
dist_max=math.sqrt(frame_cen_x**2+frame_y**2)
dist_threshold = 0.1*dist_max
delta_max=25
vel_min=0
vel_max=100
velocity = vel_max
strip_max=70
alpha_max=70
alpha=0
strip_angle_threshold=0.1*strip_max
#lower_white=np.array([0, 0, 200],dtype=np.uint8)                   #for white
#upper_white=np.array([180, 255, 255],dtype=np.uint8)
lower_white=np.array([100, 150, 0],dtype=np.uint8)                  #for blue
upper_white=np.array([140, 255, 255],dtype=np.uint8)
cap.set(5,15)
time.sleep(0.5)
# processing video
while(1):
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
    if len(contours)>0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        rect=cv2.minAreaRect(cnt)
        strip_cen_x=rect[0][0]
        strip_cen_y=rect[0][1]
        [vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
        m=vy/vx
        strip_angle=-(math.degrees(math.atan(vx/vy)))
        c=y-m*x
        k=np.sign((frame_y-m*frame_cen_x-c)*-c)
        box=cv2.boxPoints(rect)
        box=np.int0(box)
        mask=cv2.drawContours(mask,[box],0,(255,255,255),1)
        if vehicle_orientation(strip_angle,mask,m,c,k):
            cap.release()
            cv2.destroyAllWindows()
        else:
            continue
    else :
        print('no contours detected')
