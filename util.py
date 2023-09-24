import cv2 as cv
from flask import Flask, Response
import numpy as np
import matplotlib.pyplot as plt


class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)
        # self.video = cv.VideoCapture(0, cv.CAP_DSHOW)
        # self.video.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
        # self.video.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()


class ImageDetector():
    def __init__(self, server):

        @server.route("/video_feed")
        def video_feed():
            return Response(self.gen(VideoCamera()),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        img = cv.imread('beer_bottle.jpg', cv.IMREAD_GRAYSCALE)

        self.ght = cv.createGeneralizedHoughBallard()
        self.ght.setTemplate(img)

        self.ght.setMinDist(500)
        self.ght.setLevels(360)
        self.ght.setMaxBufferSize(8192)
        self.ght.setVotesThreshold(30)
        self.ght.setCannyLowThresh(100)
        self.ght.setCannyHighThresh(200)

    def gen(self, camera):
        while True:
            frame = camera.get_frame()
            # print(frame)

            frame = np.asarray(bytearray(frame), dtype=np.uint8)

            color_frame = cv.imdecode(frame, flags=cv.IMREAD_COLOR)
            gray_frame = cv.imdecode(frame, flags=cv.IMREAD_GRAYSCALE)

            frame = self.detect_image(gray_frame, color_frame)

            ret, jpeg = cv.imencode('.jpg', frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    def detect_image(self, gray_frame, color_frame):
        detection = self.ght.detect(gray_frame)

        if detection[0] is not None:
            cv.circle(img=color_frame, center=(int(detection[0][0][0][0]), int(detection[0][0][0][1])), radius=150, color=(0, 0, 255), thickness=20)

        return color_frame
