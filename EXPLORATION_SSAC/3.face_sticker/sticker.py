import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import dlib 
from math import *#삼각함수를 이용하기 때문에 math lib를 가져온다. 


#이미지 불러오기 
my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/family.jpg'
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_bgr = cv2.resize(img_bgr, (480,680))    # 480 x 680 의 크기로 Resize
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
#print("BGR image 출력 : ")#
#plt.imshow(img_bgr)#
#plt.show()#



#bgr->rgb
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#print("RGB image 출력 : ")#
#plt.imshow(img_rgb)#
#plt.show()#




#- detector 선언
detector_hog = dlib.get_frontal_face_detector()   





#face_bounding box coordinate
dlib_rects = detector_hog(img_rgb, 1)   #- (image, num of img pyramid)
print("Coordinate of Bounding box :",dlib_rects,"\n")




#print Bounding box on RGB Image 
for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
#plt.imshow(img_show_rgb)
#print("Bounding Box on RGB :")#
#plt.show()#


#import Landmark Model - bz2
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

#make list of points about landmarks 
list_landmarks = []
i=0
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)


    
    
    

    
    

#Landmarks on RGB Image    
for landmark in list_landmarks:
    
    for idx, point in enumerate(landmark):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

    img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_show_rgb)
#print("Landmarks on RGB Image : ")#
#plt.show()#

#find coordinate of nose
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print ("coordinate of nose : ",landmark[30]) # nose center index : 30
    x_eye_l=landmark[37][0] # 코를 중심으로 스티커를 만들지만 회전각을 이용하는 두 점은 눈 위쪽 중앙 landmark를 이용하여
    y_eye_l=landmark[37][1] # 구하기 때문에 두점의 x, y좌표 변수 저장
    x_eye_r=landmark[43][0]
    y_eye_r=landmark[43][1]
    x = landmark[30][0]
    y = landmark[30][1]
    w = dlib_rect.width()
    h = dlib_rect.width()
    #print ('(x,y) : (%d,%d)'%(x,y))
    #print ('(w,h) : (%d,%d)'%(w,h))
    
    
    #find angle 
    wid=abs(x_eye_l-x_eye_r) # 삼각함수(tan)를 이용하기 위해 두 점의 x와 y의 차를 구한다. 
    hei=abs(y_eye_l-y_eye_r)
    if y_eye_l >=y_eye_r: # y_eye_l(오른쪽 눈 좌표의 y값)이 y_eye_l(왼쪽 눈 좌표의 y값) 보다 크다면 고개가 왼쪽으로 회전한 것이기 
        ang=np.arctan(hei/wid) # 때문에 이럴 경우 주어진 각도를 +값으로 저장한다. (회전 방향이 반시계 방향이기 떄문)
        ang=degrees(ang)
    else:
        ang=np.arctan(hei/wid) # 위와는 반대의 경우로 오른쪽 눈 좌표의 y값이 더 크다면 고개가 오른쪽으로 회전한 것이기 때문에  
        ang=-degrees(ang) # 이 경우에는 주어진 각도를 -값으로 저장한다. 

    print("각도는",ang)




    #import sticker
    sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/whisker.png'
    img_sticker = cv2.imread(sticker_path)
    img_sticker = cv2.resize(img_sticker, (w,h))
    print(img_sticker.shape)

    whisker_c=10 # 배경처리에 문제가 있었기 때문에 이미지의 검정부분( 코와 수염부분이며 rgb로는 0,0,0)을 다른 rgb값으로 바꾸어준다.
                # ex ) (10,10,10)   

    img_sticker=np.where(img_sticker==0 ,whisker_c,img_sticker).astype(np.uint8)  # where을 사용해 바꿈



    #rotation of sticker
    height_s,width_s,channel_s=img_sticker.shape #회전을 위해 필요한 값을 따로 저장 
    matrix_s=cv2.getRotationMatrix2D((width_s/2,height_s/2),ang,1)#cv의 함수를 이용하여 회전행렬 생성((회전 중심좌표),각도, 비율)
    img_sticker=cv2.warpAffine(img_sticker,matrix_s,(width_s,height_s))# 회전행렬과 이미지를 연산하여 sticker회전 



    #sticker가 시작하는 지점 x,y
    # 스티커의 모양도 정사각형이므로 dlib_rect과 크기를 같게 해주면 된다. 
    #따라서 rect의 왼쪽 상단을 스티커의 시작점으로 잡고 오른쪽 하단을 끝점으로 잡는다. 

    #coordinate of sticker
    refined_x = x-w//2   # left
    refined_y = y-h//2     # top
    print ('refined (x,y) : (%d,%d)'%(refined_x, refined_y))


    #refined coordinate of sticker@@@@@@@ 조건문으로 수정 필요 
    if refined_y<0:
        img_sticker = img_sticker[-refined_y:]
        print (img_sticker.shape)
        refined_y = 0
        print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
        


    


    #sticker이미지에 대해 선택적으로 적용
    sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        np.where(img_sticker!=whisker_c ,sticker_area,img_sticker).astype(np.uint8)
    # 이미지에 표현해야 하는 부분은 이전에 (10,10,10)으로 바꾸어 주었기 때문에 이를 제외한 부분은 모두 sticker_area로 표현되게 한다. 
    # 이 방법으로 sticker의 기본 배경과 회전했을 경우 나타난는 검정 여백을 한번에 처리할 수 있다. 

    #result




    #final result
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
        np.where(img_sticker!=whisker_c,sticker_area,img_sticker).astype(np.uint8)

   



fig=plt.figure()
rows=1
cols=2
ax1= fig.add_subplot(rows,cols,1)
ax1.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
ax2=fig.add_subplot(rows,cols,2)
ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()