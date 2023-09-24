## Generalized Hough transform

The Generalized Hough Transform (GHT) is a feature discovery technique used in image processing and computer vision. It was proposed by Dana H. Ballard in his 1981 paper *“Generalizing the Hough transform to detect arbitrary shapes”*. It is based on an idea by Paul V. C. Hough, which first designed an approach for line discovery in images.

Richard O. Duda and Peter E. Hart proposed in their 1972 article *“Use of the Hough transformation to detect lines and curves in pictures”* a widely adopted approach able to deal with the discovery of geometric shapes which could be described analytically - lines, circles, ellipses and so on.

GHT allows for the detection of a target shape by first generating an edge image of the target (e.g. with Canny edge detector) and then computing the gradient of such edge image. This information is used on a test image to perform a voting procedure among every pixel, in order to find the position of the target.

It is easy to also keep into account the fact that a target shape might also occur scaled or rotated with respect to its original size and orientation. Unfortunately, such an extension makes the problem more difficult from a computational point of view, and not suitable for real-time detection.

GHT is pretty robust to noise in the target image, presence of other shapes and also to partial object occlusion.

Toggle your webcam on to experiment!