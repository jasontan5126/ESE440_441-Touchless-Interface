import cv2
import numpy as np

t = 2
img1 = cv2.imread('translation_img1.png', 1)
print(img1.shape[0],',',img1.shape[1])
dsize=(201*t,201*t)
img1=cv2.resize(img1,dsize)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
for i in range(0,img1.shape[0]-1):
    for j in range(img1.shape[1]-1):
        if(img1[i][j]>120):
            img1[i][j]=255
        else:
            img1[i][j]=0
cv2.imshow('image1', img1)


img2 = cv2.imread('translation_img2.png',1)
print(img2.shape[0],',',img2.shape[1])
dsize=(201*t,201*t)
img2=cv2.resize(img2,dsize)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
for i in range(0,img2.shape[0]-1):
    for j in range(img2.shape[1]-1):
        if(img2[i][j]>120):
            img2[i][j]=255
        else:
            img2[i][j]=0
cv2.imshow('image2', img2)

cropped= img1[60*t:100*t,21*t:51*t]
cv2.imshow('image3', cropped)

error = 0
error_list=[]
error_list2=[]
row=0
col=0
img_temp = img2[0:40*t,0:30*t]
print(img_temp.shape)
for i in range(0,200*t-40*t):
    for j in range(0,200*t-30*t):
        img_temp = img2[0+i:40*t+i,0+j:30*t+j]

        error = np.sum(np.subtract(img_temp, cropped))
        """if(i>120 and j>10):
            print(i,', ',j,', ',error)
            cv2.imshow('ima1', img_temp)
            cv2.waitKey(0)
            cv2.destroyWindow('ima1')"""
        error_list2.append(error)
        if(error_list):
            if(error_list[0]>error):
                error_list.pop(0)
                error_list.append(error)
                row=i
                col=j
        else:
            error_list.append(error)


print(error_list,', ',row,', ',col)
print(min(error_list2))

cropped2 = img2[0+row:40*t+row,0+col:30*t+col]
#cropped2 = img2[0:40*2,0:30*2]

cv2.imshow('image4', cropped2)

cv2.waitKey(0)
cv2.destroyAllWindows()




