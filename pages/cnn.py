from dash import html, dcc, callback, Input, Output, State, register_page
import plotly.express as px
import pandas as pd
from svm import SVM
import numpy as np
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

register_page(__name__, title="Convolutional neural networks")

layout = html.P("Convolutional neural networks page!")
