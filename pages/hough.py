from dash import Dash, html, dcc, callback, Input, Output, State, register_page
import dash_bootstrap_components as dbc

import cv2

hough_paragraph = """## Generalized Hough transform

The Generalized Hough Transform (GHT) is a feature discovery technique used in image processing and computer vision. It was proposed by Dana H. Ballard in his 1981 paper *“Generalizing the Hough transform to detect arbitrary shapes”*. It is based on an idea by Paul V. C. Hough, which first designed an approach for line discovery in images.

Richard O. Duda and Peter E. Hart proposed in their 1972 article *“Use of the Hough transformation to detect lines and curves in pictures”* a widely adopted approach able to deal with the discovery of geometric shapes which could be described analytically - lines, circles, ellipses and so on.

GHT allows for the detection of a target shape by first generating an edge image of the target (e.g. with Canny edge detector) and then computing the gradient of such edge image. This information is used on a test image to perform a voting procedure among every pixel, in order to find the position of the target.

It is easy to also keep into account the fact that a target shape might also occur scaled or rotated with respect to its original size and orientation. Unfortunately, such an extension makes the problem more difficult from a computational point of view, and not suitable for real-time detection.

GHT is pretty robust to noise in the target image and also to partial object occlusion.

Toggle your webcam on to experiment!"""

WEBCAM_FEED = "/video_feed"
OFF = "/assets/black.jpg"

register_page(__name__, title="Generalized Hough transform")

paragraph = html.Div([dcc.Markdown(hough_paragraph, mathjax=True)],
                     style={}, className="col", id="hough-paragraph")

webcam_output = html.Img(id="webcam-img", src=OFF, style={"border-radius": "15px"})

webcam_toggle_switch = dbc.Checklist(
    options=[
        {"label": "Toggle webcam", "value": 1},
    ],
    value=[],
    id="webcam_switch",
    switch=True,
    style={"padding-top": "10px"}
)

webcam_div = html.Div([webcam_output, webcam_toggle_switch], id="webcam-div",
                      style={"display": "flex",
                             "flex-direction": "column",
                             "justify-content": "center",
                             "align-items": "center"}, className="col")

layout = dbc.Row([webcam_div, paragraph])


@callback(
    Output(component_id='webcam-img', component_property="src"),
    Input(component_id='webcam_switch', component_property='value'),
    prevent_initial_call=True
)
def toggle_webcam(value):
    if len(value) > 0:
        return WEBCAM_FEED

    return OFF
