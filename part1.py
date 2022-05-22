"""
This class is what is responsible for extracting the text from what
will be from the printed menu for each area of the button
"""

# importing the module
import cv2
from realsense_depth import *
import mediapipe as mp

class extractText:

    # function to display the coordinates of
    # of the points clicked on the image
    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            """font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img)"""

    def executeDriverFunction(self):
        # reading the image
        #  ret, depth_frame, color_frame = dc.get_frame()
        #  new_color = io.imsave('new_color.jpg', color_frame)

        img = cv2.imread('new_color.jpg', 1)

        # displaying the image
        cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

def main():
    #instantiated variable
    cropImageVar = extractText()

    #Executed the function
    cropImageVar.executeDriverFunction();


# driver function
if __name__ == "__main__":
    main()