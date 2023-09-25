from dash import html, dcc, callback, Input, Output, register_page
import tensorflow as tf
from PIL import Image
from io import BytesIO
from constants import Classes
import base64
import dash_bootstrap_components as dbc

register_page(__name__, title="Convolutional neural networks")

paragraph = """## Convolutional neural networks

Convolutional Neural Networks (CNNs) are a particular type of neural network inspired by the way visual cortices behave in some mammals - as studied, for instance, in the 1959 paper *“Receptive fields of single neurones in the cat's striate cortex”* by David Hubel and Torsten Wiesel. CNNs proved to be particularly effective, among other things, in the field of Computer Vision, where they are employed in a wide variety of tasks such as image classification, image segmentation, object detection.

The use of convolutional filters allows for the utilization of the whole information available along the two dimensions (or more, if other channels other than intensity are involved) of an image.

The graph on the left represents a CNN built and trained on the CIFAR-10 dataset by the Tensorflow module named Keras. As one can see, CNNs can reach a considerable depth...
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/>
#### Deep indeed.
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/>
#### *Very* deep."""

###################################


def create_image_table(tag_list=None, rows=5, columns=10):
    if not tag_list:
        tag_list = [unknown] * (rows * columns)

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
                # Python 3, decode from bytes to string
                image_data = image_data.decode()
            data_url = 'data:image/jpg;base64,' + image_data

            image_component = html.Div([html.Img(src=data_url, style={"border-radius": "5px"}), html.Label(tag_list[row * columns + column])], style={"display": "flex",
                                                                                                                                                      "flex-direction": "column",
                                                                                                                                                      "align-items": "center"}, className="col")

            images.append(image_component)

        html_row = dbc.Row(images)
        image_rows.append(html_row)

    return html.Div(image_rows, id="image-table")


neural_model = tf.keras.models.load_model("./assets/trained_convolutional_80.keras")

_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)

wrong = "❌"
right = "✅"
unknown = "❔"

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
            labels.append(str.capitalize(str.lower(Classes(predicted_class).name)) + " " + right)
        else:
            labels.append(str.capitalize(str.lower(Classes(predicted_class).name)) + " " + wrong)

    image_table = create_image_table(tag_list=labels)

    return {"display": "none"}, image_table

############################# VIEW ##############################


first_row = dbc.Row([html.Img(src="assets/cnn_model.svg", className="col", style={"max-width": "39.75%", "padding-top": "48px"}),
                     html.Div([dcc.Markdown(paragraph, mathjax=True, dangerously_allow_html=True)], className="col")])

image_table = create_image_table()

image_table_div = html.Div(image_table, id="image-table-div")

infer_button = dbc.Button("Predict", style={"width": "80px"}, id="infer-button")
button_row = dbc.Row([infer_button], style={"display": "flex",
                                            "justify-content": "space-evenly"})

final_paragraph = """## Making inference
You can try an inference on a loaded Keras model...
"""

final_paragraph_div = html.Div([dcc.Markdown(final_paragraph, dangerously_allow_html=True)])

layout = html.Div([first_row, html.Hr(), final_paragraph_div, image_table_div, button_row])
