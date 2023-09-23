from dash import Dash, html, dcc, callback, Input, Output, State, register_page

from flask import Flask, Response
import cv2

register_page(__name__, title="Generalized Hough transform")

layout = html.Div([
    html.H1("Generalized Hough transform"),
    html.Img(id="webcam-img", src="/video_feed")
])
