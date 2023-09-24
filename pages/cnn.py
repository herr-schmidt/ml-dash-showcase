from dash import html, dcc, callback, Input, Output, State, register_page
import plotly.express as px
import pandas as pd
from svm import SVM
import numpy as np
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

register_page(__name__, title="Convolutional neural networks")

paragraph = """## Convolutional neural networks

Convolutional Neural Networks (CNNs) are a particular type of neural network inspired by the way visual cortices behave in some mammals - as studied, for instance, in the 1959 paper *“Receptive fields of single neurones in the cat's striate cortex”* by David Hubel and Torsten Wiesel. CNNs proved to be particularly effective, among other things, in the field of Computer Vision, where they are employed in a wide variety of tasks such as image classification, image segmentation, object detection.

The use of convolutional filters allows for the utilization of the whole information available along the two dimensions (or more, if other channels other than intensity are involved) of an image.

The graph on the left represents a CNN built and trained on the CIFAR-10 dataset by the Tensorflow module named Keras. As one can see, CNNs can reach a considerable depth...
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
#### Deep indeed.
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
#### *Very deep*"""

layout = dbc.Row([html.Img(src="assets/cnn_model.svg", className="col", style={"max-width": "40%"}),
                   html.Div([dcc.Markdown(paragraph, mathjax=True, dangerously_allow_html=True)], className="col", style={"max-width": "50%"})])

