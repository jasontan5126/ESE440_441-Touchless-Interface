"""
This class is what is responsible for setting the edges or boundary
of what button is which and another for each
"""

import cv2 as cv
import pytesseract
import pytesseract as pt


class EdgeDetector:

    #To read the menu as the input image
    def readMenu(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        menu = cv.imread('menu1.png')
        return menu;

    #To turn this into a gray level image of the menu
    def grayLevelMenuAndEdgeMenu(self, menu):

        #Display the gray leve image for the menu
        gray_menu = cv.cvtColor(menu, cv.COLOR_BGR2GRAY)
        cv.imshow('Gray Menu', gray_menu)

        #Display the canny edge for the menu
        canny = cv.Canny(gray_menu, 100, 175)
        cv.imshow('Edges of Menu', canny)

        return gray_menu, canny


    def extractMenuCharacterText(self, menu, canny, gray_menu):
        cv.imshow('Menu', menu)
        corner_menu = cv.cornerHarris(canny, 5, 5, 0.04)
        i = 0
        _, threshold = cv.threshold(gray_menu, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if i == 0:
                i = 1
                continue
            approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
            # using drawContours() function
            cv.drawContours(menu, [contour], 0, (0, 0, 255), 1)

            # finding center point of shape
            M = cv.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            # putting shape name at center of each shape

            if len(approx) == 4:
                cv.putText(menu, 'Quadrilateral', (x, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow('shapes', menu)
        text = pt.image_to_string(canny)
        print(text)

    def dummyMainForMainClass(self):
        # New instantiated Object variable to refer to the class function
        menuClass = EdgeDetector()

        while True:
            # To read the image menu input file
            menu = menuClass.readMenu();

            # To return the gray level image and the canny edge detector image result of the menu
            gray_menu, canny = menuClass.grayLevelMenuAndEdgeMenu(menu);

            # Extract the characters from the menu and display it in the output console
            menuClass.extractMenuCharacterText(menu, canny, gray_menu)

            key = cv.waitKey(0)


def main():
    #New instantiated Object variable to refer to the class function
    menuClass = EdgeDetector()

    while True:
        #To read the image menu input file
        menu = menuClass.readMenu();

        #To return the gray level image and the canny edge detector image result of the menu
        gray_menu, canny = menuClass.grayLevelMenuAndEdgeMenu(menu);

        #Extract the characters from the menu and display it in the output console
        menuClass.extractMenuCharacterText(menu, canny, gray_menu)

        key = cv.waitKey(0)


#Calls the main function to execute
if __name__ == "__main__":
    main()
