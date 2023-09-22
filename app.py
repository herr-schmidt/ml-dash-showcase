# Import packages
from dash import Dash, html, dcc, callback, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
from svm import SVM
import numpy as np
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

svm_paragraph = '''## Support Vector Machines
Support Vector Machines (SVMs) are a kind of supervised learning model able to perform regression and binary classification tasks. They were originally proposed by V. Vapnik in 1964, but became popular only in the early 90s.
SVM are trained by determining the hyperplane which is the most distant from the two nearest data points belonging to the two different classes at hand.
The training task over a training set $T$ can be performed by solving the Lagrangian dual of a convex optimization problem, as follows.

$$
\\begin{aligned}
\\text{maximize} \\quad & \\sum_{i \\in T} \\alpha_i - \\frac{1}{2} \\sum_{i \\in T} \\sum_{j \\in T} \\alpha_i \\alpha_j y_i y_j \\bar x_i \\bar x_j \\\\
   \\text{subject to} \\quad & \\sum_{i \\in T} \\alpha_i y_i = 0 \\\\
   & 0 \\leq \\alpha_i \\leq C, \\quad  i \\in T
\\end{aligned}
$$
\
The $\\alpha_i$ terms are the lagrangian coefficients of the dual program, each one for every input vector $\\bar x_i$; the $y_i$ terms represent the true class ($+1$ or $-1$) of each input instance. Finally, the penalization constant $C$ is used for allowing training when input samples happen to be not linearly separable (soft margin approach). As $C$ approaches infinity, the 
original hard margin SVM version is enabled.'''

# Incorporate data
df = pd.read_csv('iris.csv')

# Initialize the app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                dbc.NavLink("SVMs", href="/", active="exact"),
                dbc.NavLink("CNNs", href="/page-1", active="exact"),
                dbc.NavLink("Hough Transform", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

optimizer_radio_buttons = html.Div([
    dbc.Label("Optimizer:"),
    dbc.RadioItems(id='optimizer-dropdown',
                   options=[
                       {"label": "Ipopt", "value": "Ipopt"},
                       {"label": "CPLEX", "value": "CPLEX"},
                   ],
                   value="Ipopt",
                   style={"margin-right": "10px"})
]
)

svm_content = dbc.Row([html.Div([dcc.Markdown(svm_paragraph, mathjax=True)],
                                style={'float': 'left'}, className="col"),
                       html.Div([dcc.Graph(figure=px.scatter(df, x='SepalLength', y='SepalWidth', color="Name"), id='scatter-plot'),
                                 html.Div([dbc.Button('Fit', id='fit-button',
                                                      style={'float': 'right', 'margin': 'auto', 'padding': '5px 28px', "border-radius": "4px"}),
                                           optimizer_radio_buttons,
                                           ]
                                          )],
                                style={'float': 'left', "border-left-style": "solid", "border-left-width": "1px"}, className="col")],
                      id="svm-content")
content = html.Div([svm_content], style=CONTENT_STYLE, id="page-content")

# App layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content], id="main-div")

# Add controls to build the interaction


@callback(
    Output(component_id='scatter-plot', component_property="figure"),
    Input(component_id='fit-button', component_property='n_clicks'),
    State(component_id='optimizer-dropdown', component_property='value'),
    State(component_id='scatter-plot', component_property='figure'),
    prevent_initial_call=True
)
def launch_optimization(n_clicks, optimizer, figure):
    print("Optimizing with " + optimizer)

    # take only first 2 attributes
    class_1 = df[df["Name"] == "Iris-setosa"].iloc[:, [0, 1, 4]]
    class_2 = df[df["Name"] == "Iris-versicolor"].iloc[:, [0, 1, 4]]

    class_1_vectors = class_1.iloc[:, [0, 1]].to_numpy()
    class_2_vectors = class_2.iloc[:, [0, 1]].to_numpy()
    class_1_y = class_1["Name"].to_numpy()
    class_2_y = class_2["Name"].to_numpy()

    samples = np.concatenate((class_1_vectors, class_2_vectors), axis=0)
    classes = np.concatenate(([1 for _ in class_1_y], [-1 for _ in class_2_y]), axis=0)

    indices = [i for i in range(len(samples))]
    np.random.seed(781015)
    np.random.shuffle(indices)
    train_indices = indices[:50]

    X = samples[train_indices]
    Y = classes[train_indices]

    svm = SVM(str.lower(optimizer))
    svm.train(X, Y)

    # update plot with SVM margin and
    min_y = min(df.iloc[:, 1]) - 5
    max_y = max(df.iloc[:, 1]) + 5

    min_x_margin = - svm.w[1] / svm.w[0] * (min_y + svm.b / svm.w[1])
    max_x_margin = - svm.w[1] / svm.w[0] * (max_y + svm.b / svm.w[1])

    min_x_positive_sv = 1 / svm.w[0] * (1 - svm.w[1] * min_y - svm.b)
    max_x_positive_sv = 1 / svm.w[0] * (1 - svm.w[1] * max_y - svm.b)

    min_x_negative_sv = - 1 / svm.w[0] * (1 + svm.w[1] * min_y + svm.b)
    max_x_negative_sv = - 1 / svm.w[0] * (1 + svm.w[1] * max_y + svm.b)

    _figure = go.Figure(figure)
    _figure.add_trace(go.Scatter(x=[min_x_margin, max_x_margin], y=[min_y, max_y], name="Margin", line_shape='linear'))
    _figure.add_trace(go.Scatter(x=[min_x_positive_sv, max_x_positive_sv], y=[min_y, max_y], name="Positive SV", line_shape='linear', line=dict(dash='dot')))
    _figure.add_trace(go.Scatter(x=[min_x_negative_sv, max_x_negative_sv], y=[min_y, max_y], name="Negative SV", line_shape='linear', line=dict(dash='dot')))
    _figure.update_yaxes(autorange=False)
    _figure.update_xaxes(autorange=False)

    return _figure


@app.callback(Output("page-content", "children"), Input("url", "pathname"), prevent_initial_call=True)
def render_page_content(pathname):
    if pathname == "/":
        return svm_content
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
