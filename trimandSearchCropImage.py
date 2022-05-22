"""
The main core implementation to what will actually make
the touchless interface work at least when finding the menu
"""

import cv2
import pyrealsense2
import mediapipe as mp
import signal
import sys
import time

from numpy.ma.core import copy

from realsense_depth import *
import numpy as np
from PIL import Image

Coordinates_Menu=[[]]
counter=0
i =0

croptest = 0

menuStore = []
savedMenu = ["Burger","Pizza","Milkshake","Chicken fry","Fries","Hot Coco","Rice","Hot Dog","Soda", "Beef Stew", "Salad", "Done"]

entered = 0
index = 0
cx1 = -1
cy1 = -1


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Camera Intel Realsense
dc = DepthCamera()


class secondHand():

    def show_distance(event, x, y, args, params):
        global point
        point = (x, y)


    # Create mouse event
    # cv2.namedWindow("Color frame")
    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        global Coordinates_Menu
        global counter
        global i

        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            ret, depth_frame, color_frame = dc.get_frame()
            distance = depth_frame[y, x]
            print(x, ' ', y, ' ', distance)

            Coordinates_Menu[i].append([x,y,distance])
            counter = counter + 1
            if(counter==2):
                print(counter)
                counter = 0
                tem=[]
                Coordinates_Menu.append(tem);
                i=i+1
                print(counter)

    def gettingDistance(self):
        ret, depth_frame, color_frame = dc.get_frame()

        new_color = io.imsave('new_color.jpg', color_frame)

        # Convert the color image into a numpy array
        color_image = np.asanyarray(color_frame)

        # Image width and height of the color image
        imageHeight, imageWidth, _ = color_image.shape
        cx = 200
        cy = 200
        center = (cx, cy)
        point1 = (cx - 4, cy - 4)
        point2 = (cx + 4, cy + 4)
        cv2.rectangle(color_frame, point1, point2, (0, 0, 255), 1)
        cv2.rectangle(depth_frame, point1, point2, (0, 0, 255), 1)
        color_depth = [];
        distancearr = np.zeros(81)
        colorarr = np.zeros((81, 3))
        for x in range(point1[0], point2[0] + 1):
            for y in range(point1[1], point2[1] + 1):
                distance = depth_frame[y, x]
                color = color_frame[y, x]
                distancearr = np.append(distancearr, distance)
                colorarr = np.append(colorarr, color)
                print("x,y,z:", x, y, distance, "\n")
                # click_event(x,y,distance);
                #   print("Pixel Value: ", color)
                color_depth.append(color_frame);
                color_depth.append(depth_frame);

        img = cv2.imread('new_color.jpg', 1)
        # displaying the image
        #io.imsave('cali.jpg', img)
        cv2.imshow('image', img)
        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        cv2.waitKey(0)
        cv2.destroyWindow('image')
        print(Coordinates_Menu)

    def cropMenuItem(self, x1, x2, y1, y2):
        global menuStore
        global savedMenu
        global croptest
        image = cv2.imread('new_color.jpg', 1)
        if croptest == 0:
            cropped = image[y2:y1, x1:x2]
            io.imsave('crop.jpg', cropped)
            croptest = 1
        """img = cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)
        img = cv2.flip(img,-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        row,col = img.shape
        print("Image shape =",img.shape)
        grayImage = np.zeros(img.shape)
        for i in range(row-1):
            for j in range(col-1):
                if(img[i][j]>100):
                    grayImage[i][j]=255
                else:
                    grayImage[i][j]=0
        s = pytesseract.image_to_string(img)
        print(s)
        s = pytesseract.image_to_string(img)
        s = s.strip('\n')
        menuStore.append(s)
        cv2.imshow("grayImage",grayImage)
        cv2.waitKey(0)
        cv2.destroyWindow("grayImage")
        print(menuStore)"""

    def callHand(self):
        global entered
        global index
        global cx1,cy1

        Coord_menu_copy = copy.deepcopy(Coordinates_Menu)
        prevx = 0
        prevy = 0
        offset = 0

        image2 = cv2.imread('new_color.jpg', 1)

        cropped = image2[Coordinates_Menu[0][1][1]:Coordinates_Menu[0][0][1],
                  Coordinates_Menu[0][0][0]:Coordinates_Menu[0][1][0]]


        templateGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        while True:
            with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
                ret, depth_frame, color_frame = dc.get_frame()
                # new_color = io.imsave('new_color.jpg', color_frame)
                # Convert the color image into a numpy array
                color_image = np.asanyarray(color_frame)
                # Image width and height of the color image
                imageHeight, imageWidth, _ = color_image.shape

                image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

                #global Coordinates_Menu
                #secHand = secondHand()
                #secHand.gettingDistance()

                # cali = cv2.imread('cali.jpg', 1)

                imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(imageGray, templateGray,
                                           cv2.TM_CCOEFF_NORMED)

                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
                (startX, startY) = maxLoc

                transx = Coord_menu_copy[0][0][0] - startX
                transy = Coord_menu_copy[0][0][1] - startY

                if offset == 0:
                    offy = transy
                    offx = transx
                    offset = 1




                """print(Coord_menu_copy[0][0][0])
                print(Coord_menu_copy[0][0][1])
                print(Coord_menu_copy[0][1][0])
                print(Coord_menu_copy[0][1][1])
                print(" ")"""
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks != None:
                    for i in range(len(Coordinates_Menu) - 1):
                        """print(transx)
                        #print(" ")
                        print(transy)
                        print(" ")"""
                        Coordinates_Menu[i][0][0] = Coord_menu_copy[i][0][0] - transx+offx;
                        Coordinates_Menu[i][1][0] = Coord_menu_copy[i][1][0] - transx+offx;
                        Coordinates_Menu[i][0][1] = Coord_menu_copy[i][0][1] - transy+offy;
                        Coordinates_Menu[i][1][1] = Coord_menu_copy[i][1][1] - transy+offy;

                """print(Coordinates_Menu[0][0][0])
                print(Coordinates_Menu[0][0][1])
                print(Coordinates_Menu[0][1][0])
                print(Coordinates_Menu[0][1][1])
                print(" ")"""
                #secHand.callHand()







                # image = cv2.flip(image, 1)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks != None:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        # To iterate through each hand landmark of a hand
                        for point in mp_hands.HandLandmark:
                            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                             circle_radius=4),
                                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                             circle_radius=2),
                                                      )

                            # To access the list of landmarks of the hand
                            normalizedLandmark = hand.landmark[point]
                            # print("hand:\n",point)

                            # A tumpe with the x and y coordinates of the landmark
                            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                                   normalizedLandmark.y,
                                                                                                   imageWidth,
                                                                                                   imageHeight)

                            # To print the major points of the hand with its 2d coordinates of each of them
                            if (point == 8):
                                # print(point)
                                # distance_of_indexfinger = depth_frame[pixelCoordinatesLandmark[1], pixelCoordinatesLandmark[0]]
                                # print(pixelCoordinatesLandmark, ",", distance_of_indexfinger)
                                try:
                                    if (pixelCoordinatesLandmark[0] is not None):
                                        cx = pixelCoordinatesLandmark[0]
                                        cy = pixelCoordinatesLandmark[1]
                                except TypeError as e:
                                    print("\n")
                                else:
                                    cx = pixelCoordinatesLandmark[0]
                                    cy = pixelCoordinatesLandmark[1]
                                    center = (cx, cy)
                                    point1 = (cx - 4, cy - 4)
                                    point2 = (cx + 4, cy + 4)
                                    #print("Size: ", len(Coordinates_Menu))
                                    # cv2.waitKey(0)
                                    #print("Cx: ", cx)
                                    #print("Cy: ", cy)

                                    if(cx == cx1 and cy == cy1):
                                        print(savedMenu[index])
                                        menuStore.append(savedMenu[index])
                                        x1 = Coordinates_Menu[index][0][0]
                                        x2 = Coordinates_Menu[index][1][0]
                                        y1 = Coordinates_Menu[index][0][1]
                                        y2 = Coordinates_Menu[index][1][1]
                                        # self.cropMenuItem(x1, x2, y1, y2)

                                    for i in range(len(Coordinates_Menu) - 1):
                                        if (Coordinates_Menu[i][0][2] - depth_frame[cy, cx] < 60):
                                            if ((cx > Coordinates_Menu[i][0][0] and cx < Coordinates_Menu[i][1][
                                                0]) and (cy < Coordinates_Menu[i][0][1] and
                                                         cy > Coordinates_Menu[i][1][1])):
                                                time.sleep(0.5)
                                                cx1 = cx
                                                cy1 = cy
                                                index = i

                            # print(normalizedLandmark)

                # cv2.imshow("depth frame", depth_frame)
                # cv2.imshow("Color frame", color_frame)

                # wait for a key to be pressed to exit
                cv2.imshow("Hands", image)
                key = cv2.waitKey(1)

                if key == 27:
                    break



def calibrate(signal, frame):
    global Coordinates_Menu
    secHand = secondHand()
    secHand.gettingDistance()

    #cali = cv2.imread('cali.jpg', 1)
    image = cv2.imread('new_color.jpg', 1)

    cropped = image[Coordinates_Menu[0][1][1]:Coordinates_Menu[0][0][1], Coordinates_Menu[0][0][0]:Coordinates_Menu[0][1][0]]

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(imageGray, templateGray,
                               cv2.TM_CCOEFF_NORMED)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc

    transx = Coordinates_Menu[0][0][0] - startX
    transy = Coordinates_Menu[0][0][1] - startY

    for i in range(len(Coordinates_Menu) - 1):
        Coordinates_Menu[i][0][0] + transx;
        Coordinates_Menu[i][1][0] + transx;
        Coordinates_Menu[i][0][1] + transy;
        Coordinates_Menu[i][1][1] + transy;


    secHand.callHand()


signal.signal(signal.SIGINT, calibrate)

def main():

    secHand = secondHand()
    secHand.gettingDistance()
    secHand.callHand()

#Calls the main function to execute
if __name__ == "__main__":
    main()