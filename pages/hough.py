from dash import html, dcc, callback, Input, Output, register_page
import dash_bootstrap_components as dbc
from constants import hough_paragraph, WEBCAM_FEED_ON, WEBCAM_FEED_OFF

register_page(__name__, title="Generalized Hough transform")

paragraph = html.Div([dcc.Markdown(hough_paragraph, mathjax=True)],
                     style={}, className="col", id="hough-paragraph")

webcam_output = html.Img(id="webcam-img", src=WEBCAM_FEED_OFF, style={"border-radius": "15px"})

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
        return WEBCAM_FEED_ON

    return WEBCAM_FEED_OFF
