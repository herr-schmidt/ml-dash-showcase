# Import packages
from dash import Dash, html, dcc, Input, Output, page_registry, page_container
import dash_bootstrap_components as dbc
from flask import Flask, Response
from util import VideoCamera


# Initialize the app
server = Flask(__name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True, server=server)

# navbar
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Dash showcase", className="display-4"),
        html.Hr(),
        html.P(
            "A simple showcase made with Dash and Bootstrap", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink(f"{page['title']}", href=page["relative_path"], active="exact") for page in page_registry.values()
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div([page_container], style=CONTENT_STYLE, id="page-content")

# App layout
app.layout = html.Div([sidebar, content], id="main-div")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, )
