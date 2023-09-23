from dash import html, dcc, callback, Input, Output, State, register_page
import plotly.express as px
import pandas as pd
from svm import SVM
import numpy as np
import plotly.graph_objects as go

import dash_bootstrap_components as dbc

register_page(__name__, path="/", title="Support vector machines")

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
df = pd.read_csv('assets/iris.csv')

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

layout = dbc.Row([html.Div([dcc.Markdown(svm_paragraph, mathjax=True)],
                           style={'float': 'left'}, className="col"),
                  html.Div([dcc.Graph(figure=px.scatter(df, x='SepalLength', y='SepalWidth', color="Name"), id='scatter-plot'),
                            html.Div([dbc.Button('Fit', id='fit-button',
                                                 style={'float': 'right', 'margin': 'auto', 'padding': '5px 28px', "border-radius": "4px"}),
                                      optimizer_radio_buttons,
                                      ]
                                     )],
                           style={'float': 'left', "border-left-style": "solid", "border-left-width": "1px"}, className="col")],
                 id="svm-content")


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
