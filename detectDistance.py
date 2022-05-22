"""
This class is what is responsible for detecting the distance of
whatever target
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from skimage import io, color

class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
    def get_frame(self):
        align = rs.align(rs.stream.color)
        frames = self.pipeline.wait_for_frames()
        frames = align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not aligned_depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()

    def show_distance(event, x, y, z, args, params):
        global point
        point = (x,y)

    def dummyMainForMainClass(self):
        dc = DepthCamera()
        cv2.namedWindow("Color frame")
        while True:
            ret, depth_frame, color_frame = dc.get_frame()
            cx = 200
            cy = 300
            center = (cx, cy)
            point1 = (cx - 4, cy - 4)
            point2 = (cx + 4, cy + 4)
            cv2.rectangle(color_frame, point1, point2, (0, 0, 255), 1)
            cv2.rectangle(depth_frame, point1, point2, (0, 0, 255), 1)
            # distance = depth_frame[point1[1], point1[0]]
            distancearr = np.zeros(81)
            colorarr = np.zeros((81, 3))
            for x in range(point1[0], point2[0] + 1):
                for y in range(point1[1], point2[1] + 1):
                    distance = depth_frame[y, x]
                    color = color_frame[y, x]
                    distancearr = np.append(distancearr, distance)
                    colorarr = np.append(colorarr, color)
                    print("x,y,z:", x, y, distance, "\n")
                    print("Pixel Value: ", color)

            print("Color Array: \n", colorarr)
            print("Distance Array: \n", distancearr)
            cv2.imshow("depth frame", depth_frame)
            cv2.imshow("Color frame", color_frame)

            key = cv2.waitKey(1)
            if key == 27:
                break


def main():
    dc = DepthCamera()
    cv2.namedWindow("Color frame")
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        cx = 200
        cy = 300
        center = (cx,cy)
        point1 = (cx-4,cy-4)
        point2 = (cx+4,cy+4)
        cv2.rectangle(color_frame, point1, point2, (0,0,255), 1)
        cv2.rectangle(depth_frame, point1, point2, (0, 0, 255), 1)
       # distance = depth_frame[point1[1], point1[0]]
        distancearr = np.zeros(81)
        colorarr = np.zeros((81,3))
        for x in range(point1[0], point2[0]+1):
            for y in range(point1[1], point2[1]+1):
                distance = depth_frame[y, x]
                color = color_frame[y, x]
                distancearr=np.append(distancearr,distance)
                colorarr = np.append(colorarr,color)
                print("x,y,z:",x,y,distance,"\n")
                print("Pixel Value: ", color)


        print("Color Array: \n",colorarr)
        print("Distance Array: \n",distancearr)
        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("Color frame", color_frame)


        key = cv2.waitKey(1)
        if key == 27:
            break

#Calls the main function to execute
if __name__ == "__main__":
    main()