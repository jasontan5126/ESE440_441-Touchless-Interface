# importing the module

"""
Don't worry about this main file since I tried testing the secondHand.py on here
but it didn't work as expected on this source file even though I called the secondHand
class and called its functions I made. Other classes I called it worked
"""

import cv2
import mediapipe as mp
from secondHand import *
from part1 import *
from detectDistance import *
from EdgeDetector import *

import cv2
import pyrealsense2
import mediapipe as mp
import pytesseract

from realsense_depth import *
import numpy as np
from PIL import Image

#All the instantiate variables extracted from the other classes
realSenseDepthVar = DepthCamera()
#secondHandVar = secondHand()
readTextVar = extractText()
edgeDetectVar = EdgeDetector()

class mainFunction():


    # Excecute the class to extract characters from the menu if want to
    def executePart1Class(self):
        # Executed the function
        readTextVar.executeDriverFunction()

    # Excecute the detect distance class if we want to
    def executeDetectDistance(self):
        realSenseDepthVar.dummyMainForMainClass()

    #Excecute the edge detector class if we want to
    def executeEdgeDetector(self):
        edgeDetectVar.dummyMainForMainClass()


def main():
    # Initialize Camera Intel Realsense
    dc = DepthCamera()
    ret, depth_frame, color_frame = dc.get_frame()
    secondHandVar = secondHand()
    secondHandVar.gettingDistance(depth_frame, color_frame )
    secondHandVar.callHand(depth_frame, color_frame)

# driver function
if __name__ == "__main__":
    main()