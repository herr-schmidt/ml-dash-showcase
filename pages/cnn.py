from dash import html, dcc, callback, Input, Output, register_page
import tensorflow as tf
from PIL import Image
from io import BytesIO
from constants import Classes, cnn_intro_paragraph, cnn_inference_paragraph, RIGHT_INFERENCE, WRONG_INFERENCE, UNKNOWN
import base64
import dash_bootstrap_components as dbc

register_page(__name__, title="Convolutional neural networks")


def create_image_table(tag_list=None, rows=5, columns=10):
    if not tag_list:
        tag_list = [UNKNOWN] * (rows * columns)

    image_rows = []

    for row in range(rows):
        images = []
        for column in range(columns):
            image = Image.fromarray(x_test[row * columns + column], mode="RGB")
            image = image.resize((64, 64))
            output = BytesIO()
            image.save(output, format='JPEG')
            im_data = output.getvalue()

            image_data = base64.b64encode(im_data)
            if not isinstance(image_data, str):
                image_data = image_data.decode()
            data_url = 'data:image/jpg;base64,' + image_data

            image_component = html.Div([html.Img(src=data_url, style={"border-radius": "5px"}),
                                        html.Label(tag_list[row * columns + column])], style={"display": "flex",
                                                                                              "flex-direction": "column",
                                                                                              "align-items": "center"}, className="col")
            images.append(image_component)

        html_row = dbc.Row(images, style={"padding-bottom": "15px"})
        image_rows.append(html_row)

    return html.Div(image_rows, id="image-table")


neural_model = tf.keras.models.load_model("./assets/trained_convolutional_80.keras")

_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)


###########################################


@callback(
    Output("image-table", "style"),
    Output(component_id="image-table-div", component_property='children'),
    Input(component_id="infer-button", component_property='n_clicks'),
    prevent_initial_call=True
)
def display_output(n_clicks):

    prediction_vectors = neural_model.predict(x_test)

    labels = []

    for i in range(len(prediction_vectors)):
        predicted_class = prediction_vectors[i].argmax()
        if(predicted_class == y_test[i]):
            labels.append(str.capitalize(str.lower(Classes(predicted_class).name)) + " " + RIGHT_INFERENCE)
        else:
            labels.append(str.capitalize(str.lower(Classes(predicted_class).name)) + " " + WRONG_INFERENCE)

    image_table = create_image_table(tag_list=labels)

    return {"display": "none"}, image_table

############################# VIEW ##############################


first_row = dbc.Row([html.Img(src="assets/cnn_model.svg", className="col", style={"max-width": "39.75%", "padding-top": "48px"}),
                     html.Div([dcc.Markdown(cnn_intro_paragraph, mathjax=True, dangerously_allow_html=True)], className="col")])

image_table = create_image_table()

image_table_div = html.Div(image_table, id="image-table-div")

infer_button = dbc.Button("Predict", style={"width": "100px"}, id="infer-button")
button_row = dbc.Row([infer_button], style={"display": "flex",
                                            "justify-content": "space-evenly",
                                            "padding-top": "40px"})


final_paragraph_div = html.Div([dcc.Markdown(cnn_inference_paragraph, dangerously_allow_html=True, style={"padding-bottom": "40px"})])

layout = html.Div([first_row, html.Hr(), final_paragraph_div, image_table_div, button_row])
