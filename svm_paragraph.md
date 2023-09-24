## Support Vector Machines
Support Vector Machines (SVMs) are a kind of supervised learning model able to perform regression and binary classification tasks. They were originally proposed by V. Vapnik during in 1964, but became popular only in the early 90s.
SVM are trained by determining the hyperplane which is the most distant from the two nearest data points belonging to the two different classes at hand.
The training task over a training set $T$ can be performed by solving the Lagrangian dual of a convex optimization problem, as follows.
$$\begin{aligned}
\text{maximize} \quad & \sum_{i \in T} \alpha_i - \frac{1}{2} \sum_{i \in T} \sum_{j \in T} \alpha_i \alpha_j y_i y_j \bar x_i \bar x_j \\
   \text{subject to} \quad & \sum_{i \in T} \alpha_i y_i = 0 \\
   & 0 \leq \alpha_i \leq C, \quad  i \in T
\end{aligned}$$
\
The $\alpha_i$ terms are the lagrangian coefficients of the dual program, each one for every input vector $\bar x_i$; the $y_i$ terms represent the true class ($+1$ or $-1$) of each input instance. Finally, the penalization constant $C$ is used for allowing training when input samples happen to be not linearly separable (soft margin approach). As $C$ approaches infinity, the 
original hard margin SVM version is enabled.
