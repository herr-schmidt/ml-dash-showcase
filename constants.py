from enum import Enum

WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 360
WEBCAM_FEED_ON = "/video_feed"
WEBCAM_FEED_OFF = "/assets/black.jpg"


class Classes(Enum):

    PLANE = 0
    CAR = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9

WRONG_INFERENCE = "❌"
RIGHT_INFERENCE = "✅"
UNKNOWN = "❔"


################### MARKDOWN PARAGRAPHS ###################


### SVM ###

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
original hard margin SVM version is enabled.

By pressing on the “Fit” button in the panel on the right, you can train a model for the famous Iris dataset. The underlying implementation is based on the above convex mathematical program, defined by means of Pyomo, which allows for various solvers to be used for the actual resolution (Ipopt and CPLEX are available here, but many more are supported). After fitting has been performed, you should see the margin appear in the scatter plot.'''

### HOUGH ###

hough_paragraph = """## Generalized Hough transform

The Generalized Hough Transform (GHT) is a feature discovery technique used in image processing and computer vision. It was proposed by Dana H. Ballard in his 1981 paper *“Generalizing the Hough transform to detect arbitrary shapes”*. It is based on an idea by Paul V. C. Hough, which first designed an approach for line discovery in images.

Richard O. Duda and Peter E. Hart proposed in their 1972 article *“Use of the Hough transformation to detect lines and curves in pictures”* a widely adopted approach able to deal with the discovery of geometric shapes which could be described analytically - lines, circles, ellipses and so on.

GHT allows for the detection of a target shape by first generating an edge image of the target (e.g. with Canny edge detector) and then computing the gradient of such edge image. This information is used on a test image to perform a voting procedure among every pixel, in order to find the position of the target.

It is easy to also keep into account the fact that a target shape might also occur scaled or rotated with respect to its original size and orientation. Unfortunately, such an extension makes the problem more difficult from a computational point of view, and not suitable for real-time detection.

GHT is pretty robust to noise in the target image and also to partial object occlusion.

Toggle your webcam on to experiment!"""

### CNN ###

cnn_intro_paragraph = """## Convolutional neural networks

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

cnn_inference_paragraph = """## Making inference
You can try an inference on the images below (belonging to the CIFAR-10 dataset) using a pretrained Keras model by clicking the “Predict” button below. Wait a few seconds and see how it behaves!
"""
