---
title: "ml novo"
date: 2020-08-10
layout: post
categories: notas
tags: notas
---

# Table of Contents

1.  [primeira semana](#org10e4940)
    1.  [o que é machine learning?](#org7b2a971)
        1.  [leitura](#org4df5af7)
    2.  [tipos de algoritmos](#org0dbd02a)
    3.  [supervised learning](#org5478aed)
        1.  [leitura](#org5a2e5cd)
        2.  [regression problem](#org874b4d9)
        3.  [classification problem](#orgb500ebb)
    4.  [unsupervised learning](#org5dd5ca8)
        1.  [leitura](#org29d8588)
    5.  [regressão linear](#org4ded1dd)
        1.  [model and cost function](#orgd00ecc5)
        2.  [cost function](#org62cdb4a)
        3.  [cost function intuition I](#orgcb6ae7d)
        4.  [cost function intuition II](#org83d9ed9)
    6.  [parameter learning](#orgc446cc2)
        1.  [gradient descent](#org4b252d5)
        2.  [gradient descent intuition](#org2092390)
        3.  [gradiente descent for linear regression](#org4c6a073)
    7.  [linear algebra review](#org1721777)
        1.  [☛ TODO matrices and vectors](#orgb135b44)
        2.  [addition and scalar multiplication](#orgdd77664)
        3.  [matrix and vector multiplication](#orge0157b1)
        4.  [matrix matrix multiplication](#org6a2afdd)
        5.  [matrix multiplication properties](#org3468f31)
        6.  [inverse and transpose](#orgcc5a35d)
2.  [segunda semana](#orgfa5914d)
    1.  [multivariate linear regression](#org055e0e9)
        1.  [multiple features](#orge039f15)
        2.  [gradient descent for multiple variables](#orga44cd8d)
        3.  [gradient descent in practice I - feature scaling](#org12c2381)
        4.  [gradient descent in practice II - learning rate](#orgfc355cf)
        5.  [features and polynomial regression](#org6c68c6e)
    2.  [computing parameters analytically](#org2e43a6e)
        1.  [normal equation](#org729a754)
        2.  [normal equation noninvertibility](#org2b7494d)
    3.  [programming tips from mentors](#orgc0d6adf)
3.  [terceira semana](#org4e1b78b)
    1.  [classification and representation](#org3eb116f)
        1.  [classification](#org2b9088c)
        2.  [hypothesis representation](#org5279361)
        3.  [decision boundary](#org41e1eff)
    2.  [logistic regression model](#org60e243d)
        1.  [cost function](#org852cf0e)
        2.  [simplified cost function and gradient descent](#org578bc51)
        3.  [advanced optimization](#org21cb705)
    3.  [multiclass classification: one-vs-all](#orgd15d7a7)
    4.  [solving the problem of overfitting](#org808b418)
        1.  [the problem of overfitting](#orgaccc123)
        2.  [cost function](#org3deb568)
        3.  [regularized linear regression](#org4f5c666)
        4.  [regularized logistic regression](#orged2226b)
4.  [quarta semana](#org385c12d)
    1.  [neural networks](#orgf09d38a)
        1.  [model representation I](#org03d761c)
        2.  [model representation II](#org02b7539)
    2.  [applications](#orge10dc70)
        1.  [examples and intuitions I](#orgac8f3f7)
        2.  [examples and intuitions II](#org9b42e24)
        3.  [multiclass classification](#org56762e2)
5.  [quinta semana](#org6f5bfa0)
    1.  [cost function and backpropagation](#org6216e75)
        1.  [cost function](#org09f959d)
        2.  [backpropagation algorithm](#org03fa012)
        3.  [backpropagation intuition](#org93db040)
    2.  [backpropagation in practice](#orgb4484cf)
        1.  [implementation note: unrolling parameters](#org7c5fe93)
        2.  [gradient checking](#orge38541f)
        3.  [random initialization](#org95413ab)
        4.  [putting it together](#org61fb001)
6.  [sexta semana](#org0dedd65)
    1.  [evaluating a learning algorithm](#orgeaa1584)
        1.  [evaluating a hypothesis](#orgb911536)
        2.  [model selection and train/validation/test sets](#orgf4649e3)
    2.  [bias vs variance](#orgb62f429)
        1.  [diagnosing bias vs variance](#org7eb0b95)
        2.  [regulatization and bias/variance](#orgb4fa3a3)
        3.  [learning curves](#org16e36b9)
        4.  [deciding what to do next revisited](#org1957621)
    3.  [building a spam classifier](#org8652b13)
        1.  [prioritizing what to work on](#orgc5747af)
        2.  [error analysis](#org569bda7)
7.  [sétima semana](#orgbb245c7)
    1.  [optimization objective](#org562890a)
    2.  [large margin intuition](#orgb46d5cd)
    3.  [Mathematics Behind Large Margin Classification (Optional)](#orge4c34c7)
    4.  [kernels I](#org0ab8d6a)
    5.  [kernels II](#org41c2fd9)
    6.  [choosing SVM parameters](#org34462cb)
    7.  [Using An SVM](#orge43e75e)
    8.  [multi-class classification](#org150493f)
    9.  [logistic regression vs. SVMs](#org9bd9a23)
8.  [oitava semana](#org6bda59f)
    1.  [ML:Clustering](#org9701f23)
        1.  [Unsupervised Learning: Introduction](#org5bdb782)
        2.  [k-means algorithm](#org9f5d159)
        3.  [optimization objective](#org611688e)
        4.  [random initialization](#orgaf27add)
        5.  [choosing the number of clusters](#org63dbe35)
        6.  [bonus: discussion of the drawbacks of k-means](#orgf57ddfa)
    2.  [ML:Dimensionality Reduction](#org361b7d7)
        1.  [principal component analysis problem formulation](#orgb09de29)
        2.  [principal component analysis algorithm](#org3c2a996)
        3.  [reconstruction from compressed representation](#org2e8c879)
        4.  [choosing the number of principal components](#org185d683)
        5.  [advice for applying PCA](#orgf381aea)
9.  [nona semana](#org4526617)
    1.  [ML:Anomaly detection](#orgbdd8465)
        1.  [problem motivation](#org1f3a356)
        2.  [gaussian distribution](#org5b60b87)
        3.  [algorithm](#orgf83c08a)
        4.  [Developing and Evaluating an Anomaly Detection System](#orge3a0f44)
        5.  [anomaly detection vs. supervised learning](#orgf1cd9f4)
        6.  [choosing what features to use](#org1120872)
        7.  [Multivariate Gaussian Distribution (Optional)](#orgd911187)
        8.  [Anomaly Detection using the Multivariate Gaussian Distribution (Optional)](#org376aafa)
    2.  [ML: Recommender Systems](#orgbcfeb10)
        1.  [problem formulation](#org749267a)
        2.  [content based recommendations](#org8fb2ecc)
        3.  [collaborative filtering algorithm](#orgae5bd3e)
        4.  [vectorization: low rank matrix factorization](#org67a6b80)
        5.  [implementation detail: mean normalization](#orgdab2af8)
10. [décima semana](#org073dd44)
    1.  [learning with large datasets](#orgc6989a5)
    2.  [stochastic gradient descent](#org7a69b39)
    3.  [mini-batch gradient descent](#orgf5a0b26)
    4.  [stochastic gradient descent convergence](#orga981d14)
    5.  [online learning](#orgf520bab)
    6.  [map reduce and data parallelism](#org2c95ad7)

<a id="org10e4940"></a>

# primeira semana


<a id="org7b2a971"></a>

## o que é machine learning?

He says, a computer program is said to learn from experience E with
respect to some task T and some performance measure P, if its
performance on T, as measured by P, improves with experience E. 

Ele diz que um programa de computador aprende com uma experiência E em
relação a uma tarefa T e uma medida de performance P, se sua
performance em T, medida por P, melhora com a experiência E.


<a id="org4df5af7"></a>

### leitura

Two definitions of Machine Learning are offered. Arthur Samuel
described it as: "the field of study that gives computers the ability
to learn without being explicitly programmed." This is an older,
informal definition.

Tom Mitchell provides a more modern definition: "A computer program is
said to learn from experience E with respect to some class of tasks T
and performance measure P, if its performance at tasks in T, as
measured by P, improves with experience E."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two
broad classifications:

Supervised learning and Unsupervised learning.


<a id="org0dbd02a"></a>

## tipos de algoritmos

`unsupervised learning`
`supervised learning`
reinforced learning
recommender systems


<a id="org5478aed"></a>

## supervised learning


<a id="org5a2e5cd"></a>

### leitura

In supervised learning, we are given a data set and already know what
our correct output should look like, having the idea that there is a
relationship between the input and the output.

Supervised learning problems are categorized into "regression" and
"classification" problems. In a regression problem, we are trying to
predict results within a continuous output, meaning that we are trying
to map input variables to some continuous function. In a
classification problem, we are instead trying to predict results in a
discrete output. In other words, we are trying to map input variables
into discrete categories.

Example 1:

We could turn this example into a classification problem by instead
making our output about whether the house "sells for more or less than
the asking price." Here we are classifying the houses based on price
into two discrete categories.

Given data about the size of houses on the real estate market, try to
predict their price. Price as a function of size is a continuous
output, so this is a regression problem.

Example 2:

(a) Regression - Given a picture of a person, we have to predict their
age on the basis of the given picture

(b) Classification - Given a patient with a tumor, we have to predict
whether the tumor is malignant or benign.


<a id="org874b4d9"></a>

### regression problem

encaixar uma equação num gráfico com os dados


<a id="orgb500ebb"></a>

### classification problem

valores discretos para separar classificações diferentes


<a id="org5dd5ca8"></a>

## unsupervised learning


<a id="org29d8588"></a>

### leitura

Unsupervised learning allows us to approach problems with little or no
idea what our results should look like. We can derive structure from
data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on
relationships among the variables in the data.

With unsupervised learning there is no feedback based on the
prediction results.

Example:

Clustering: Take a collection of 1,000,000 different genes, and find a
way to automatically group these genes into groups that are somehow
similar or related by different variables, such as lifespan, location,
roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find
structure in a chaotic environment. (i.e. identifying individual
voices and music from a mesh of sounds at a cocktail party).


<a id="org4ded1dd"></a>

## regressão linear


<a id="orgd00ecc5"></a>

### model and cost function

1.  model representation

    1.  leitura
    
        To establish notation for future use, we’ll use x<sup>(i)</sup> to denote
        the “input” variables (living area in this example), also called input
        features, and y<sup>(i)</sup> to denote the “output” or target variable that
        we are trying to predict (price). A pair (x<sup>(i)</sup> , y<sup>(i)</sup> ) is called a
        training example, and the dataset that we’ll be using to
        learn—a list of m training examples (x<sup>(i)</sup>,y<sup>(i)</sup>);i=1,&#x2026;,m—is called a
        training set. Note that the superscript “(i)” in the notation is
        simply an index into the training set, and has nothing to do with
        exponentiation. We will also use X to denote the space of input
        values, and Y to denote the space of output values. In this example, X
        = Y = ℝ.
        
        To describe the supervised learning problem slightly more formally,
        our goal is, given a training set, to learn a function h : X → Y so
        that h(x) is a “good” predictor for the corresponding value of y. For
        historical reasons, this function h is called a hypothesis. Seen
        pictorially, the process is therefore like this:
        
        ![img](primeira_semana/H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20.14.58_2019-08-08_19-15-47.png)
        
        When the target variable that we’re trying to predict is continuous,
        such as in our housing example, we call the learning problem a
        regression problem. When y can take on only a small number of discrete
        values (such as if, given the living area, we wanted to predict if a
        dwelling is a house or an apartment, say), we call it a classification
        problem.


<a id="org62cdb4a"></a>

### cost function

1.  leitura

    We can measure the accuracy of our hypothesis function by using a cost
    function. This takes an average difference (actually a fancier version
    of an average) of all the results of the hypothesis with inputs from
    x's and the actual output y's.
    
    \[ J(\theta_0, \theta_1) = \frac{1}{2m} \sum^{m}_{i=1} (\hat{y}_i - y_i)^2 =
    \frac{1}{2m}\sum^{m}_{i=1} (h_\theta(x_i)-y_i)^2\]
    
    To break it apart, it is \[ \frac{1}{2} \hat{x} \] where \[ \hat{x} \] is the mean
    of the squares of \[ h_\theta (x_{i}) - y_{i} \] , or the difference between
    the predicted value and the actual value.
    
    This function is otherwise called the "Squared error function", or
    "Mean squared error". The mean is halved \[ \left(\frac{1}{2}\right)
    \] as a convenience for the
     computation of the gradient descent, as the derivative term of the
    square function will cancel out the \[ \frac{1}{2} \] term. The
    following image summarizes what the cost function does:
    
    ![img](primeira_semana/R2YF5Lj3EeajLxLfjQiSjg_110c901f58043f995a35b31431935290_Screen-Shot-2016-12-02-at-5.23.31-PM_2019-08-08_20-42-43.png)


<a id="orgcb6ae7d"></a>

### cost function intuition I

1.  leitura

    If we try to think of it in visual terms, our training data set is
    scattered on the x-y plane. We are trying to make a straight line
    (defined by h<sub>&theta;</sub>(x)) which passes through these scattered data points.
    
    Our objective is to get the best possible line. The best possible line
    will be such so that the average squared vertical distances of the
    scattered points from the line will be the least. Ideally, the line
    should pass through all the points of our training data set. In such a
    case, the value of J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) will be 0. The following example shows
    the ideal situation where we have a cost function of 0.
    
    ![img](primeira_semana/_B8TJZtREea33w76dwnDIg_3e3d4433e32478f8df446d0b6da26c27_Screenshot-2016-10-26-00.57.56_2019-08-08_21-16-29.png)
    
    When &theta;<sub>1</sub> = 1, we get a slope of 1 which goes through every
    single data point in our model. Conversely, when &theta;<sub>1</sub> = 0.5, we
    see the vertical distance from our fit to the data points increase.
    
    ![img](primeira_semana/8guexptSEeanbxIMvDC87g_3d86874dfd37b8e3c53c9f6cfa94676c_Screenshot-2016-10-26-01.03.07_2019-08-08_21-25-47.png)
    
    This increases our cost function to 0.58. Plotting several other
    points yields to the following graph:
    
    ![img](primeira_semana/fph0S5tTEeajtg5TyD0vYA_9b28bdfeb34b2d4914d0b64903735cf1_Screenshot-2016-10-26-01.09.05_2019-08-08_21-26-02.png)
    
    Thus as a goal, we should try to minimize the cost function. In this
    case, &theta;<sub>1</sub> = 1 is our global minimum.


<a id="org83d9ed9"></a>

### cost function intuition II

1.  leitura

    A contour plot is a graph that contains many contour lines. A contour
    line of a two variable function has a constant value at all points of
    the same line. An example of such a graph is the one to the right
    below.
    
    ![img](primeira_semana/N2oKYp2wEeaVChLw2Vaaug_d4d1c5b1c90578b32a6672e3b7e4b3a4_Screenshot-2016-10-29-01.14.37_2019-08-08_21-44-54.png)
    
    Taking any color and going along the 'circle', one would expect to get
    the same value of the cost function. For example, the three green
    points found on the green line above have the same value for
    J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) and as a result, they are found along the
    same line. The circled x displays the value of the cost function for
    the graph on the left when &theta;<sub>0</sub> = 800 and &theta;<sub>1</sub> = -0.15. Taking another
    h(x) and plotting its contour plot, one gets the following graphs:
    
    ![img](primeira_semana/26RZhJ34EeaiZBL80Yza_A_0f38a99c8ceb8aa5b90a5f12136fdf43_Screenshot-2016-10-29-01.14.57_2019-08-08_21-45-52.png)
    
    When &theta;<sub>0</sub> = 360 and &theta;<sub>1</sub> = 0, the value of J(&theta;<sub>0</sub>, &theta;<sub>1</sub>) the contour plot gets
    closer to the center thus reducing the cost function error. Now giving
    our hypothesis function a slightly positive slope results in a better
    fit of the data.
    
    ![img](primeira_semana/hsGgT536Eeai9RKvXdDYag_2a61803b5f4f86d4290b6e878befc44f_Screenshot-2016-10-29-09.59.41_2019-08-08_21-53-32.png)
    
    The graph above minimizes the cost function as much as possible and
    consequently, the result of &theta;<sub>1</sub> and &theta;<sub>0</sub> tend to be around 0.12 and 250
    respectively. Plotting those values on our graph to the right seems to
    put our point in the center of the inner most 'circle'.


<a id="orgc446cc2"></a>

## parameter learning


<a id="org4b252d5"></a>

### gradient descent

1.  leitura

    So we have our hypothesis function and we have a way of measuring how
    well it fits into the data. Now we need to estimate the parameters in
    the hypothesis function. That's where gradient descent comes in.
    
    Imagine that we graph our hypothesis function based on its fields &theta;<sub>0</sub>
    and &theta;<sub>1</sub> (actually we are graphing the cost function as a function of
    the parameter estimates). We are not graphing x and y itself, but the
    parameter range of our hypothesis function and the cost resulting from
    selecting a particular set of parameters.
    
    We put &theta;<sub>0</sub> on the x axis and &theta;<sub>1</sub> on the y axis, with the cost
    function on the vertical z axis. The points on our graph will be the
    result of the cost function using our hypothesis with those specific
    theta parameters. The graph below depicts such a setup.
    
    ![img](primeira_semana/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26_2019-08-08_22-39-33.png)
    
    We will know that we have succeeded when our cost function is at the
    very bottom of the pits in our graph, i.e. when its value is the
    minimum. The red arrows show the minimum points in the graph.
    
    The way we do this is by taking the derivative (the tangential line to
    a function) of our cost function. The slope of the tangent is the
    derivative at that point and it will give us a direction to move
    towards. We make steps down the cost function in the direction with
    the steepest descent. The size of each step is determined by the
    parameter α, which is called the learning rate.
    
    For example, the distance between each 'star' in the graph above
    represents a step determined by our parameter α. A smaller α would
    result in a smaller step and a larger α results in a larger step. The
    direction in which the step is taken is determined by the partial
    derivative of J(&theta;<sub>0</sub>, &theta;<sub>1</sub>). Depending on where one starts on the graph,
    one could end up at different points. The image above shows us two
    different starting points that end up in two different places.
    
    The gradient descent algorithm is:
    
    repeat until convergence:
    
    \[ \theta_j := \theta_j - \alpha\frac{\delta}{\delta\theta_j}J(\theta_0, \theta_1) \]
    
    where
    
    j = 0, 1 represents the feature index number.
    
    At each iteration j, one should simultaeously update the parameters
    &theta;<sub>1</sub>, &theta;<sub>2</sub>,&#x2026;, &theta;<sub>n</sub>. Updating a specific parameter prior to calculating
    another one on the j<sup>(th)</sup> iteration would yield to a wrong
    implementation.
    
    ![img](primeira_semana/yr-D1aDMEeai9RKvXdDYag_627e5ab52d5ff941c0fcc741c2b162a0_Screenshot-2016-11-02-00.19.56_2019-08-08_22-49-00.png)


<a id="org2092390"></a>

### gradient descent intuition

In this video we explored the scenario where we used one parameter &theta;<sub>1</sub>
and plotted its cost function to implement a gradient descent. Our
formula for a single parameter was :

Repeat until convergence:

\[ \theta_1 := \theta_1 - \alpha\frac{d}{d\theta_1}J(\theta_1) \]

Regardless of the slope's sign for \[ \frac{d}{d\theta_1}J(\theta_1) \], &theta;<sub>1</sub>
eventually converges to its minimum value. The following grapg shows
that when the slope is negative, the value of &theta;<sub>1</sub> increases and when it
is positive, the value of &theta;<sub>1</sub> decreases.

![img](primeira_semana/SMSIxKGUEeav5QpTGIv-Pg_ad3404010579ac16068105cfdc8e950a_Screenshot-2016-11-03-00.05.06_2019-08-09_10-33-08.png)

On a side note, we should adjust our parameter &alpha; to ensure that
the gradient descent algorithm converges in a reasonable time. Failure
to converge or too much time to obtain the minimum value imply that
our step size is wrong.

![img](primeira_semana/UJpiD6GWEeai9RKvXdDYag_3c3ad6625a2a4ec8456f421a2f4daf2e_Screenshot-2016-11-03-00.05.27_2019-08-09_12-32-34.png)

How does gradient descent converge with a fixed step size &alpha;?

The intuition behind the convergence is that \[ \frac{d}{d\theta_1} J(\theta_1) \]
approaches 0 as we approach the bottom of our convex function. At the
minimum, the derivative will always be 0 and thus we get:

\[ \theta_1 := \theta_1 - \alpha.0 \]

![img](primeira_semana/RDcJ-KGXEeaVChLw2Vaaug_cb782d34d272321e88f202940c36afe9_Screenshot-2016-11-03-00.06.00_2019-08-09_12-34-41.png)


<a id="org4c6a073"></a>

### gradiente descent for linear regression

When specifically applied to the case of linear regression, a new form
of the gradient descent equation can be derived. We can substitute our
actual cost function and our actual hypothesis function and modify the
equation to :

\[ \begin{align*} \text{repeat until convergence: } \lbrace & \\
\theta_0 := & \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) \\
\theta_1 := & \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i})
x_{i}\right) \\ \rbrace& \end{align*} \]

Where m is the size of the training set, &theta;<sub>0</sub> a constant that will be
changing simultaneously with &theta;<sub>1</sub> and x<sub>i</sub>, y<sub>i</sub> are values of the given
training set (data).

Note that we have separated out the two cases for &theta;<sub>j</sub> into separte
equations for &theta;<sub>0</sub> and &theta;<sub>1</sub>; and that for &theta;<sub>1</sub> we are multiplying x<sub>i</sub> at the
end due to the derivative. The following is a derivation of \[
\frac{\delta}{\delta\theta_j} J(\theta) \] for a single example:

![img](primeira_semana/QFpooaaaEea7TQ6MHcgMPA_cc3c276df7991b1072b2afb142a78da1_Screenshot-2016-11-09-08.30.54_2019-08-09_13-29-59.png)

The point of all this is that if we start with a guess for our
hypothesis and then repeatedly apply these gradient descent equations,
our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function
J. This method looks at every example in the entire training set on
every step, and is called batch gradient descent. Note that, while
gradient descent can be susceptible to local minima in general, the
optimization problem we have posed here for linear regression has only
one global, and no other local, optima; thus gradient descent always
converges (assuming the learning rate α is not too large) to the
global minimum. Indeed, J is a convex quadratic function. Here is an
example of gradient descent as it is run to minimize a quadratic
function.

![img](primeira_semana/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49_2019-08-09_13-34-35.png)

The ellipses shown above are the contours of a quadratic
function. Also shown is the trajectory taken by gradient descent,
which was initialized at (48,30). The x’s in the figure (joined by
straight lines) mark the successive values of θ that gradient descent
went through as it converged to its minimum.


<a id="org1721777"></a>

## linear algebra review


<a id="orgb135b44"></a>

### ☛ TODO matrices and vectors

Matrices are 2-dimensional arrays:

\[\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \\ j & k & l\end{bmatrix} \]

The above matrix has four rows and three columns, so it is a 4 x 3 matrix.

A vector is a matrix with one column and many rows:

\[ \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix} \]

Notation and terms:

-   A<sub>ij</sub> refers to the element in the ith row and jth column of matrix A.
-   A vector with 'n' rows is referred to as an 'n'-dimensional

vector.

-   v<sub>iv</sub> refers to the element in the ith row of the vector.
-   In general, all our vectors and matrices will be 1-indexed. Note that

for some programming languages, the arrays are 0-indexed.  

-   Matrices are usually denoted by uppercase names while vectors are lowercase.
-   "Scalar" means that an object is a single value, not a vector or

matrix.

-   \[\Re\] refers to the set of scalar real numbers.
-   \[\Re^n\] refers to the set of n-dimensional vectors of real numbers.

Run the cell below to get familiar with the commands in
Octave/Matlab. Feel free to create matrices and vectors and try out
different things.

{% highlight nillangnilswitchesnilflags %}
nilbody
{% endhighlight %}


<a id="orgdd77664"></a>

### addition and scalar multiplication

Addition and subtraction are element-wise, so you simply add or
subtract each corresponding element:

\[ \begin{bmatrix} a & b \\ c & d \\ \end{bmatrix}
+\begin{bmatrix} w & x \\ y & z \\ \end{bmatrix}
=\begin{bmatrix} a+w & b+x \\ c+y & d+z \\ \end{bmatrix} \]

Subtracting Matrices:

\[ \begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} - \begin{bmatrix} w
& x \\ y & z \\ \end{bmatrix} =\begin{bmatrix} a-w & b-x \\ c-y & d-z
\\ \end{bmatrix} \]

To add or subtract two matrices, their dimensions must be the same.

In scalar multiplication, we simply multiply every element by the
scalar value:

\[ \begin{bmatrix} a & b \\ c & d \\ \end{bmatrix} * x
=\begin{bmatrix} a*x & b*x \\ c*x & d*x \\ \end{bmatrix} \]

In scalar division, we simply divide every element by the scalar
value:

\[ \frac{\begin{bmatrix} a & b \\ c & d \\ \end{bmatrix}}{x}
=\begin{bmatrix} a /x & b/x \\ c /x & d /x \\
\end{bmatrix} \]

Experiment below with the Octave/Matlab commands for matrix addition
and scalar multiplication. Feel free to try out different
commands. Try to write out your answers for each command before
running the cell below.

{% highlight octave %}
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
ans = add_As
{% endhighlight %}


<a id="orge0157b1"></a>

### matrix and vector multiplication

We map the column of the vector onto each row of the matrix,
multiplying each element and summing the result.

$$ \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}
*\begin{bmatrix} x \\ y \\ \end{bmatrix} =\begin{bmatrix}
a*x + b*y \\ c*x + d*y \\ e*x + f*y\end{bmatrix} $$

The result is a vector. The number of columns of the matrix must equal
the number of rows of the vector.

An m x n matrix multiplied by an n x 1 vector results in an m x 1
vector.

Below is an example of a matrix-vector multiplication. Make sure you
understand how the multiplication works. Feel free to try different
matrix-vector multiplications.

{% highlight octave %}
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v
% ans = Av
{% endhighlight %}


<a id="org6a2afdd"></a>

### matrix matrix multiplication

We multiply two matrices by breaking it into several vector
multiplications and concatenating the result.

$$ \begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}
*\begin{bmatrix} w & x \\ y & z \\ \end{bmatrix}
=\begin{bmatrix} a*w + b*y & a*x + b*z \\ c*w + d*y & c*x + d*z
\\ e*w + f*y & e*x + f*z\end{bmatrix} $$

An m x n matrix multiplied by an n x o matrix results in an m x o
matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix
resulted in a 3 x 2 matrix.

To multiply two matrices, the number of columns of the first matrix
must equal the number of rows of the second matrix.

For example:

{% highlight octave %}
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B
ans = mult_AB
% Make sure you understand why we got that result
{% endhighlight %}


<a id="org3468f31"></a>

### matrix multiplication properties

-   Matrices are not commutative: \[ A * B \neq B * A \]
-   Matrices are associative: \[ (A * B) * C = A * (B * C) \]

The identity matrix, when multiplied by any matrix of the same
dimensions, results in the original matrix. It's just like multiplying
numbers by 1. The identity matrix simply has 1's on the diagonal
(upper left to lower right diagonal) and 0's elsewhere.

\[ \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1
\\ \end{bmatrix} \]

When multiplying the identity matrix after some matrix (A∗I), the
square identity matrix's dimension should match the other matrix's
columns. When multiplying the identity matrix before some other matrix
(I∗A), the square identity matrix's dimension should match the other
matrix's rows.

{% highlight octave %}
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 
ans = AB
% Note that IA = AI but AB != BA
{% endhighlight %}


<a id="orgcc5a35d"></a>

### inverse and transpose

The inverse of a matrix A is denoted A<sup>-1</sup>. Multiplying by the
inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute
inverses of matrices in octave with the pinv(A) function and in
Matlab with the inv(A) function. Matrices that don't have an
inverse are singular or degenerate.

The transposition of a matrix is like rotating the matrix 90° in
clockwise direction and then reversing it. We can compute
transposition of matrices in matlab with the transpose(A) function or
A':

\[ A = \begin{bmatrix} a & b \\ c & d \\ e & f
\end{bmatrix} \]

\[ A^T = \begin{bmatrix} a & c & e \\ b & d & f \\
\end{bmatrix} \]

In other words:

A<sub>ij</sub> = A<sup>T</sup><sub>ji</sub>
​	

{% highlight octave %}
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = pinv(A)

% What is A^(-1)*A? 
A_invA = pinv(A)*A

ans = A_inv
{% endhighlight %}


<a id="orgfa5914d"></a>

# segunda semana


<a id="org055e0e9"></a>

## multivariate linear regression


<a id="orge039f15"></a>

### multiple features

Linear regression with multiple variables is also known as
"multivariate linear regression".

We now introduce notation for equations where we can have any number
of input variables.

x<sup>(i)</sup><sub>j</sub> = value of feature j in the ith training example
x<sup>(i)</sup> = the input (features) of the ith training example
m = the number of training examples
n = the number of features

The multivariable form of the hypothesis function accommodating these
multiple features is as follows:

h<sub>θ</sub>(x)=θ<sub>0</sub>+θ<sub>1</sub>\*x<sub>1</sub>+θ<sub>2</sub>\*x<sub>2</sub>+θ<sub>3</sub>\*x<sub>3</sub>+⋯+θ<sub>n</sub>\*x<sub>n</sub> 

In order to develop intuition about this function, we can think about
&theta;<sub>0</sub> as the basic price of a house, &theta;<sub>1</sub> as the price per square meter, &theta;<sub>2</sub>
as the price per floor, etc. x<sub>1</sub> will be the number of square meters in
the house,x<sub>2</sub> the number of floors, etc.

Using the definition of matrix multiplication, our multivariable
hypothesis function can be concisely represented as:

\[ \begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em}
\theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0
\\ x_1 \\ \vdots \\ x_n\end{bmatrix}= \theta^T
x\end{align*} \]

This is a vectorization of our hypothesis function for one training
example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume
x<sup>(i)</sup><sub>0</sub>= 1 for (i∈1,…,m). This allows us to do matrix operations with
theta and x. Hence making the two vectors '&theta;' and x<sup>(i)</sup> match each
other element-wise (that is, have the same number of elements: n+1).]


<a id="orga44cd8d"></a>

### gradient descent for multiple variables

The gradient descent equation itself is generally the same form; we
just have to repeat it for our 'n' features:

\[ \begin{align*} & \text{repeat until convergence:} \; \lbrace
\\ \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) -
y^{(i)}) \cdot x_0^{(i)}\\ \; & \theta_1 := \theta_1 - \alpha \frac{1}{m}
\sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \\ \; & \theta_2
:= \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)}
\\ & \cdots \\ \rbrace \end{align*} \]

In other words:

\[ \begin{align*}& \text{repeat until convergence:} \; \lbrace
\\ \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) -
y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\\
\rbrace\end{align*} \]

The following image compares gradient descent with one variable to
gradient descent with multiple variables:

![img](segunda_semana/MYm8uqafEeaZoQ7hPZtKqg_c974c2e2953662e9578b38c7b04591ed_Screenshot-2016-11-09-09.07.04_2019-08-09_20-01-05.png)


<a id="org12c2381"></a>

### gradient descent in practice I - feature scaling

We can speed up gradient descent by having each of our input values in
roughly the same range. This is because θ will descend quickly on
small ranges and slowly on large ranges, and so will oscillate
inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables
so that they are all roughly the same. Ideally:

-1 &le; x<sub>(i)</sub> &le; 1

or

-0.5 &le; x<sub>(i)</sub> &le; 0.5

These aren't exact requirements; we are only trying to speed things
up. The goal is to get all input variables into roughly one of these
ranges, give or take a few.

Two techniques to help with this are feature scaling and mean
normalization. Feature scaling involves dividing the input values by
the range (i.e. the maximum value minus the minimum value) of the
input variable, resulting in a new range of just 1. Mean normalization
involves subtracting the average value for an input variable from the
values for that input variable resulting in a new average value for
the input variable of just zero. To implement both of these
techniques, adjust your input values as shown in this formula:

\[ x_i := \frac{x_i - \mu_i}{s_i} \]

Where &mu;<sub>i</sub> is the average of all the values for feature (i) and s<sub>i</sub> is
the range of values (max - min), or s<sub>i</sub> is the standard deviation.

Note that dividing by the range, or dividing by the standard
deviation, give different results. The quizzes in this course use
range - the programming exercises use standard deviation.

For example, if x<sub>i</sub> represents housing prices with a range of 100 to
2000 and a mean value of 1000, then, \[ x_i := \dfrac{price-1000}{1900} \].


<a id="orgfc355cf"></a>

### gradient descent in practice II - learning rate

Debugging gradient descent. Make a plot with number of iterations on
the x-axis. Now plot the cost function, J(θ) over the number of
iterations of gradient descent. If J(θ) ever increases, then you
probably need to decrease α.

Automatic convergence test. Declare convergence if J(θ) decreases by
less than E in one iteration, where E is some small value such as
10<sup>-3</sup>. However in practice it's difficult to choose this threshold
value.

![img](segunda_semana/FEfS3aajEea3qApInhZCFg_6be025f7ad145eb0974b244a7f5b3f59_Screenshot-2016-11-09-09.35.59_2019-08-09_22-01-25.png)

It has been proven that if learning rate α is sufficiently small, then
J(θ) will decrease on every iteration.

![img](segunda_semana/rC2jGKgvEeamBAoLccicqA_ec9e40a58588382f5b6df60637b69470_Screenshot-2016-11-11-08.55.21_2019-08-09_22-02-59.png)

To summarize:

If &alpha; is too small: slow convergence.

If &alpha; is too large: ￼may not decrease on every iteration and thus
may not converge.


<a id="org6c68c6e"></a>

### features and polynomial regression

We can improve our features and the form of our hypothesis function in
a couple different ways.

We can combine multiple features into one. For example, we can combine
x<sub>1</sub> and x<sub>2</sub> into a new feature x<sub>3</sub> by taking x<sub>1</sub>⋅x<sub>2</sub>.

Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that
does not fit the data well.

We can change the behavior or curve of our hypothesis function by
making it a quadratic, cubic or square root function (or any other
form).

For example, if our hypothesis function is \[ h_\theta(x) = \theta_0 +
\theta_1 x_1 \]
then we can create additional features based on x<sub>1</sub>, to get the
quadratic function \[ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2
\] or the cubic 
function \[ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 +
\theta_3 x_1^3 \]

In the cubic version, we have created new features x<sub>2</sub> and x<sub>3</sub>
where x<sub>2</sub> = x<sub>1</sub><sup>2</sup> and x<sub>3</sub> = x<sub>1</sub><sup>3</sup> .

To make it a square root function, we could do: \[ h_\theta(x) = \theta_0 +
\theta_1 x_1 + \theta_2 \sqrt{x_1} \] ​ ​

One important thing to keep in mind is, if you choose your features
this way then feature scaling becomes very important.

eg. if x<sub>1</sub> has range 1 - 1000 then range of x<sub>1</sub><sup>2</sup> becomes 1 -
1000000 and that of x<sub>1</sub><sup>3</sup> becomes 1 - 1000000000


<a id="org2e43a6e"></a>

## computing parameters analytically


<a id="org729a754"></a>

### normal equation

Gradient descent gives one way of minimizing J. Let’s discuss a second
way of doing so, this time performing the minimization explicitly and
without resorting to an iterative algorithm. In the "Normal Equation"
method, we will minimize J by explicitly taking its derivatives with
respect to the θj ’s, and setting them to zero. This allows us to find
the optimum theta without iteration. The normal equation formula is
given below:

θ = (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> y

![img](segunda_semana/dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10.06.16_2019-08-09_22-56-41.png)

There is no need to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal
equation:

|---|---|
| Gradient Descent | Normal Equation |
|---|---|
| Needs to choose alpha | No need to choose alpha |
| Needs many iterations | No need to iterate |
| O(kn<sup>2</sup>) | O(n<sup>3</sup>), need to calculate the inverse of X<sup>T</sup> X |
| Works well when n is large | Slow if n is ery large |
|---|---|

With the normal equation, computing the inversion has complexity
O(n³). So if we have a very large number of features, the normal
equation will be slow. In practice, when n exceeds 10,000 it might be
a good time to go from a normal solution to an iterative process.


<a id="org2b7494d"></a>

### normal equation noninvertibility

When implementing the normal equation in octave we want to use the
'pinv' function rather than 'inv.' The 'pinv' function will give you a
value of &theta; even if X<sup>T</sup> X is not invertible.

If X<sup>TX</sup> X is noninvertible, the common causes might be having :

-   Redundant features, where two features are very closely related

(i.e. they are linearly dependent) 

-   Too many features (e.g. m ≤ n). In this case, delete some features or

use "regularization" (to be explained in a later lesson).  

Solutions to the above problems include deleting a feature that is
linearly dependent with another or deleting one or more features when
there are too many features.


<a id="orgc0d6adf"></a>

## programming tips from mentors

Thank you to Machine Learning Mentor, Tom Mosher, for compiling this
list

Subject: Confused about "h(x) = theta' \* x" vs. "h(x) = X \* theta?"

Text: The lectures and exercise PDF files are based on Prof. Ng's
feeling that novice programmers will adapt to for-loop techniques more
readily than vectorized methods. So the videos (and PDF files) are
organized toward processing one training example at a time. The course
uses column vectors (in most cases), so h (a scalar for one training
example) is theta' \* x.

Lower-case x typically indicates a single training example.

The more efficient vectorized techniques always use X as a matrix of
all training examples, with each example as a row, and the features as
columns. That makes X have dimensions of (m x n). where m is the
number of training examples. This leaves us with h (a vector of all
the hypothesis values for the entire training set) as X \* theta, with
dimensions of (m x 1).

X (as a matrix of all training examples) is denoted as upper-case X.

Throughout this course, dimensional analysis is your friend.

Subject: Tips from the Mentors: submit problems and fixing program
errors Text: This post contains some frequently-used tips about the
course, and to help get your programs working correctly.

The Most Important Tip: Search the forum before posting a new
question. If you've got a question, the chances are that someone else
has already posted it, and received an answer. Save time for yourself
and the Forum users by searching for topics before posting a new one.

Running your scripts: At the Octave/Matlab command line, you do not
need to include the ".m" portion of the script file name. If you
include the ".m", you'll get an error message about an invalid
indexing operation. So, run the Exercise 1 script by typing just "ex1"
at the command line.

You also do not need to include parenthesis () when using the submit
script. Just type "submit".

You cannot execute your functions by simply typing the name. All of
the functions you will work on require a set of parameter values,
enter between a set of parenthesis. Your three methods of testing your
code are:

1 - use an exercise script, such as "ex1"

2 - use a Unit Test (see below) where you type-in the entire command
line including the parameters.

3 - use the submit script.

Making the grader happy: The submit grader uses a different test case
than what is in the PDF file. These test cases use a different size of
data set and are more sensitive to small errors than the ex test
cases. Your code must work correctly with any size of data set.

Your functions must handle the general case. This means:

-   You should avoid using hard-coded array indexes.

-   You should avoid having fixed-length arrays and matrices.

It is very common for students to think that getting the same answer
as listed in the PDF file means they should get full credit from the
grader. This is a false hope. The PDF file is just one test case. The
grader uses a different test case.

Also, the grader does not like your code to send any additional
outputs to the workspace. So, every line of code should end with a
semicolon.

Getting Help: When you want help from the Forum community, please use
this two-step procedure:

1 - Search the Forum for keywords that relate to your
problem. Searching by the function name is a good start.

2 - If you don't find a suitable thread, then do this:

2a - Find the unit tests for that exercise (see below), and run the
appropriate test. Attempt to debug your code.

2b - Take a screen capture of your whole console workspace (including
the command line), and post it to the forum, along with any other
useful information (computer type, Octave/Matlab version, other tests
you've tried, etc).

Debugging: If your code runs but gives the wrong answers, you can
insert a "keyboard" command in your script, just before the function
ends. This will cause the program to exit to the debugger, so you can
inspect all your variables from the command line. This often is very
helpful in analysing math errors, or trying out what commands to use
to implement your function.

There are additional test cases and tutorials listed in pinned threads
under "All Course Discussions". The test cases are especially helpful
in debugging in situations where you get the expected output in ex but
get no points or an error when submitting.

Unit Tests: Each programming assignment has a "Discussions" area in
the Forum. In this section you can often find "unit tests". These are
additional test cases, which give you a command to type, and provides
the expected results. It is always a good idea to test your functions
using the unit tests before submitting to the grader.

If you run a unit test and do not get the correct results, you can
most easily get help on the forums by posting a screen capture of your
workspace - including the command line you entered, and the results.

Having trouble submitting your work to the grader?:

-   This section will need to be supplemented with info appropriate to
    the new submission system. If you run the submit script and get a
    message that your identity can't be verified, be sure that you have
    logged-in using your Coursera account email and your Programming
    Assignment submission password.

-   If you get the message "submit undefined", first check that you are
    in the working directory where you extracted the files from the ZIP
    archive. Use "cd" to get there if necessary.

-   If the "submit undefined" error persists, or any other "function
    undefined" messages appear, try using the "addpath(pwd)" command to
    add your present working directory (pwd) to the Octave execution
    path.

-If the submit script crashes with an error message, please see the
thread "Mentor tips for submitting your work" under "All Course
Discussions".

-The submit script does not ask for what part of the exercise you want
to submit. It automatically grades any function you have modified.

Found some errata in the course materials?  This course material has
been used for many previous sessions. Most likely all of the errata
has been discovered, and it's all documented in the 'Errata' section
under 'Supplementary Materials'. Please check there before posting
errata to the Forum.

Error messages with fmincg()

The "short-circuit" warnings are due to use a change in the syntax for
conditional expressions (| and & vs || and &&) in the newer versions
of Matlab. You can edit the fmincg.m file and the warnings may be
resolved.

Warning messages about "automatic broadcasting"?  See this link for
info.

Warnings about "divide by zero" These are normal in some of the
exercises, and do not represent a problem in your function. You can
ignore them - Octave senses the issue and substitutes a +Inf or -Inf
value so your program continues to execute.


<a id="org4e1b78b"></a>

# terceira semana


<a id="org3eb116f"></a>

## classification and representation


<a id="org2b9088c"></a>

### classification

To attempt classification, one method is to use linear regression and
map all predictions greater than 0.5 as a 1 and all less than 0.5 as
a 0. However, this method doesn't work well because classification is
not actually a linear function.

The classification problem is just like the regression problem, except
that the values we now want to predict take on only a small number of
discrete values. For now, we will focus on the binary classification
problem in which y can take on only two values, 0 and 1. (Most of what
we say here will also generalize to the multiple-class case.) For
instance, if we are trying to build a spam classifier for email, then
x<sup>(i)</sup> may be some features of a piece of email, and y may be 1 if
it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. 0 is also
called the negative class, and 1 the positive class, and they are
sometimes also denoted by the symbols “-” and “+.” Given x<sup>(i)</sup>,
the corresponding y<sup>(i)</sup> is also called the label for the training
example.


<a id="org5279361"></a>

### hypothesis representation

We could approach the classification problem ignoring the fact that y
is discrete-valued, and use our old linear regression algorithm to try
to predict y given x. However, it is easy to construct examples where
this method performs very poorly. Intuitively, it also doesn’t make
sense for h<sub>&theta;</sub>(x) to take values larger than 1 or smaller
than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the
form for our hypotheses h<sub>&theta;</sub>(x) to satisfy \[ 0 \leq h_\theta(x)
\leq 1 \]. 
This is accomplished by plugging &theta;<sup>Txθ</sup> T x into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic
Function":

\[ \begin{align*}& h_\theta (x) = g ( \theta^T x ) \\ \\& z
= \theta^T x \\& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*} \]

The following image shows us what the sigmoid function looks like:

![img](terceira_semana/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function_2019-08-11_12-31-29.png)

The function g(z), shown here, maps any real number to the (0, 1)
interval, making it useful for transforming an arbitrary-valued
function into a function better suited for classification.

h<sub>&theta;</sub>(x) will give us the probability that our output is 1. For
example, h<sub>&theta;</sub>(x)=0.7 gives us a probability of 70% that our output
is 1. Our probability that our prediction is 0 is just the complement
of our probability that it is 1 (e.g. if probability that it is 1 is
70%, then the probability that it is 0 is 30%).

\[\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ;
\theta) \\& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}\]


<a id="org41e1eff"></a>

### decision boundary

In order to get our discrete 0 or 1 classification, we can translate
the output of the hypothesis function as follows:

\[ \begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \\&
h_\theta(x) < 0.5 \rightarrow y = 0 \\\end{align*} \]

The way our logistic function g behaves is that when its input is
greater than or equal to zero, its output is greater than or equal to
0.5:

\[ \begin{align*}& g(z) \geq 0.5 \\& when \; z \geq 0\end{align*} \]

Remember.

\[ \begin{align*}z=0, e^{0}=1 \Rightarrow g(z)=1/2\\ z \to \infty, e^{-\infty} \to 0
\Rightarrow g(z)=1 \\ z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0
\end{align*} \]

So if our input to g is &theta;<sup>T</sup> X, then that means:

\[ \begin{align*}& h_\theta(x) = g(\theta^T x) \geq 0.5 \\& when
\; \theta^T x \geq 0\end{align*} \]

From these statements we can now say:

\[ \begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \\& \theta^T x < 0 \Rightarrow y = 0
\\\end{align*} \]

The decision boundary is the line that separates the area where y = 0
and where y = 1. It is created by our hypothesis function.

Example:

\[ \begin{align*}& \theta = \begin{bmatrix}5 \\ -1 \\
0\end{bmatrix} \\ & y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0
\\ & 5 - x_1 \geq 0 \\ & - x_1 \geq -5 \\& x_1 \leq 5
\\ \end{align*} \]

In this case, our decision boundary is a straight vertical line placed
on the graph where x<sub>1</sub> = 5, and everything to the left of that
denotes y = 1, while everything to the right denotes y = 0.

Again, the input to the sigmoid function g(z) (e.g. &theta;<sup>T</sup> X) doesn't
need to be linear, and could be a function that describes a circle
(e.g. z = &theta;<sub>0</sub> + &theta;<sub>1</sub> x<sub>1</sub><sup>2</sup> +&theta;<sub>2</sub> x<sub>2</sub><sup>2</sup>) or any shape to fit our data.


<a id="org60e243d"></a>

## logistic regression model


<a id="org852cf0e"></a>

### cost function

We cannot use the same cost function that we use for linear regression
because the Logistic Function will cause the output to be wavy,
causing many local optima. In other words, it will not be a convex
function.

Instead, our cost function for logistic regression looks like:

\[ \begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m
\mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\ &
\mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1}
\\ & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; &
\text{if y = 0}\end{align*} \]

When y = 1, we get the following plot for J(&theta;) vs h<sub>&theta;</sub> (x):

![img](terceira_semana/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class_2019-08-11_15-50-01.png)

Similarly, when y = 0, we get the following plot for J(&theta;) vs
h<sub>&theta;</sub> (x):

![img](terceira_semana/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class_2019-08-11_15-50-34.png)

\[ \begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if }
h_\theta(x) = y \\ & \mathrm{Cost}(h_\theta(x),y) \rightarrow
\infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1
\\ & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if }
y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \\
\end{align*} \]

If our correct answer 'y' is 0, then the cost function will be 0 if
our hypothesis function also outputs 0. If our hypothesis approaches
1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if
our hypothesis function outputs 1. If our hypothesis approaches 0,
then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ)
is convex for logistic regression.


<a id="org578bc51"></a>

### simplified cost function and gradient descent

We can compress our cost function's two conditional cases into one
case:

\[ Cost(hθ(x),y) = -y.log(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x)) \]

Notice that when y is equal to 1, then the second term
\[ (1-y)log(1-h_{\theta}(x)) \] will be zero and will not affect the result. If
y is equal to 0, then the first term \[ -ylog(h_{\theta}(x)) \] will be zero and
will not affect the result.

We can fully write out our entire cost function as follows:

\[ J(\theta) = - \frac{1}{m}
\sum^{m}_{i=1}[y^{(i)} \log(h_\theta(x^{(i)}))+(1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] \]

A vectorized implementation is:

\[ \begin{align*} & h = g(X\theta)\\ & J(\theta) = \frac{1}{m} \cdot
\left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*} \]

Gradient Descent

Remember that the general form of gradient descent is:

\[ \begin{align*}& Repeat \; \lbrace \\ & \; \theta_j := \theta_j - \alpha
\dfrac{\partial}{\partial \theta_j}J(\theta) \\ & \rbrace\end{align*} \]

We can work out the derivative part using calculus to get:

\[ \begin{align*} & Repeat \; \lbrace \\ & \; \theta_j := \theta_j - \frac{\alpha}{m}
\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \\ & \rbrace \end{align*} \]

Notice that this algorithm is identical to the one we used in linear
regression. We still have to simultaneously update all values in
theta.

A vectorized implementation is:

\[ \theta:= \theta - \frac{\alpha}{m}X^T(g(X\theta)-\hat{y}) \]


<a id="org21cb705"></a>

### advanced optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated,
faster ways to optimize θ that can be used instead of gradient
descent. We suggest that you should not write these more sophisticated
algorithms yourself (unless you are an expert in numerical computing)
but use the libraries instead, as they're already tested and highly
optimized. Octave provides them.

We first need to provide a function that evaluates the following two
functions for a given input value θ:

\[ \begin{align*} & J(\theta) \\ & \dfrac{\partial}{\partial \theta_j}J(\theta)\end{align*} \]

We can write a single function that returns both of these:

{% highlight octave %}
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
{% endhighlight %}

Then we can use octave's "fminunc()" optimization algorithm along with
the "optimset()" function that creates an object containing the
options we want to send to "fminunc()". (Note: the value for MaxIter
should be an integer, not a character string - errata in the video at
7:30)

{% highlight octave %}
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
{% endhighlight %}

We give to the function "fminunc()" our cost function, our initial
vector of theta values, and the "options" object that we created
beforehand.


<a id="orgd15d7a7"></a>

## multiclass classification: one-vs-all

Now we will approach the classification of data when we have more than
two categories. Instead of y = {0,1} we will expand our definition so
that y = {0,1&#x2026;n}.

Since y = {0,1&#x2026;n}, we divide our problem into n+1 (+1 because the
index starts at 0) binary classification problems; in each one, we
predict the probability that 'y' is a member of one of our classes.

\[ \begin{align*}& y \in \lbrace0, 1 ... n\rbrace \\&
h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \\& h_\theta^{(1)}(x) = P(y =
1 | x ; \theta) \\& \cdots \\& h_\theta^{(n)}(x) = P(y = n | x ; \theta)
\\& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x)
)\\\end{align*} \]

We are basically choosing one class and then lumping all the others
into a single second class. We do this repeatedly, applying binary
logistic regression to each case, and then use the hypothesis that
returned the highest value as our prediction.

The following image shows how one could classify 3 classes:

![img](terceira_semana/cqmPjanSEeawbAp5ByfpEg_299fcfbd527b6b5a7440825628339c54_Screenshot-2016-11-13-10.52.29_2019-08-12_00-21-50.png)

To summarize:

Train a logistic regression classifier h<sub>&theta;</sub>(x)for each
class￼ to predict the probability that y = i.

To make a prediction on a new x, pick the class ￼that maximizes
h<sub>&theta;</sub> (x)


<a id="org808b418"></a>

## solving the problem of overfitting


<a id="orgaccc123"></a>

### the problem of overfitting

Consider the problem of predicting y from x ∈ R. The leftmost figure
below shows the result of fitting a y = θ<sub>0</sub>+θ<sub>1</sub> x to a dataset. We see
that the data doesn’t really lie on straight line, and so the fit is
not very good.

![img](terceira_semana/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30_2019-08-12_13-26-18.png)

Instead, if we had added an extra feature x<sup>2</sup>, and fit y = &theta;<sub>0</sub> +
&theta;<sub>1</sub> x + &theta;<sub>2</sub> x<sup>2</sup>, then we obtain a slightly better
fit to the data (See middle figure). Naively, it might seem that the
more features we add, the better. However, there is also a danger in
adding too many features: The rightmost figure is the result of
fitting a 5<sup>th</sup>5 th order polynomial y = &sum;<sup>5</sup><sub>j=0</sub> &theta;<sub>j</sub> x<sup>j</sup>. We see that
even though the fitted curve passes through the data
perfectly, we would not expect this to be a very good predictor of,
say, housing prices (y) for different living areas (x). Without
formally defining what these terms mean, we’ll say the figure on the
left shows an instance of underfitting—in which the data clearly shows
structure not captured by the model—and the figure on the right is an
example of overfitting.

Underfitting, or high bias, is when the form of our hypothesis
function h maps poorly to the trend of the data. It is usually caused
by a function that is too simple or uses too few features. At the
other extreme, overfitting, or high variance, is caused by a
hypothesis function that fits the available data but does not
generalize well to predict new data. It is usually caused by a
complicated function that creates a lot of unnecessary curves and
angles unrelated to the data.

This terminology is applied to both linear and logistic
regression. There are two main options to address the issue of
overfitting:

1.  Reduce the number of features:

2.  Manually select which features to keep.

3.  Use a model selection algorithm (studied later in the course).

4.  Regularization

5.  Keep all the features, but reduce the magnitude of parameters &theta;<sub>j</sub>.
6.  Regularization works well when we have a lot of slightly useful

features.


<a id="org3deb568"></a>

### cost function

If we have overfitting from our hypothesis function, we can reduce the
weight that some of the terms in our function carry by increasing
their cost.

Say we wanted to make the following function more quadratic:

\[ \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \theta_4 x^4 \]

We'll want to eliminate the influence of &theta;<sub>3</sub> x<sup>3</sup> and &theta;<sub>4</sub> x<sup>4</sup>. Without
actually getting rid of these features or changing the form of our
hypothesis, we can instead modify our cost function:

\[ \min_{\theta}​ \dfrac{1}{2m}​ \sum_{i=1}^{m} (h_\theta(x^{(i)})- y^{(i)})^2 +1000.\theta_3^2 ​ +1000.\theta_4^2 ​ \]

We've added two extra terms at the end to inflate the cost of &theta;<sub>3</sub>
and &theta;<sub>4</sub>. Now, in order for the cost function to get close to zero,
we will have to reduce the values of &theta;<sub>3</sub> and &theta;<sub>4</sub> to near
zero. This will in turn greatly reduce the values of &theta;<sub>3</sub> x<sup>3</sup> and
&theta;<sub>4</sub> x<sup>4</sup> in our hypothesis function. As a result, we see that the
new hypothesis (depicted by the pink curve) looks like a quadratic
function but fits the data better due to the extra small terms &theta;<sub>3</sub> x<sup>3</sup>
and &theta;<sub>4</sub> x<sup>4</sup>.

![img](terceira_semana/j0X9h6tUEeawbAp5ByfpEg_ea3e85af4056c56fa704547770da65a6_Screenshot-2016-11-15-08.53.32_2019-08-12_18-28-04.png)

We could also regularize all of our theta parameters in a single summation as:

\[ \min_\theta \dfrac{1}{2m} \sum^{m}_{i=1} (h_\theta (x^{(i)})-y^{(i)})^2 +
\lambda\sum^n_{j=1} \theta^2_j \]

The λ, or lambda, is the regularization parameter. It determines how
much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth
the output of our hypothesis function to reduce overfitting. If lambda
is chosen to be too large, it may smooth out the function too much and
cause underfitting. Hence, what would happen if &lambda; = 0 or is too
small ?


<a id="org4f5c666"></a>

### regularized linear regression

We can apply regularization to both linear regression and logistic
regression. We will approach linear regression first.

Gradient Descent We will modify our gradient descent function to
separate out &theta;<sub>0</sub> from the rest of the parameters because we do not
want to penalize &theta;<sub>0</sub>.

\[ \begin{align*} & \text{Repeat}\ \lbrace \\ & \ \ \ \ \theta_0 :=
\theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\ & \ \
\ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) -
y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in
\lbrace 1,2...n\rbrace\\ & \rbrace \end{align*} \]

The term \[ \dfrac{\lambda}{m}\theta_j \]  performs our regularization. With some
manipulation our update rule can also be represented as:

\[ \theta_j ​ := \theta_j (1 - \dfrac{\alpha\lambda}{m}) - \dfrac{\alpha}{m} \sum^m_{i=1} (h \theta ​ (x^{(i)}
)-y^{(i)})x_j^{(i)} \]

The first term in the above equation, \[ 1 - \alpha\dfrac{\lambda}{m} \] will
always be less than 1. Intuitively you can see it as reducing the
value of &theta;<sub>j</sub> by some amount on every update. Notice that the second
term is now exactly the same as it was before.

-`Normal Equation`-

Now let's approach regularization using the alternate
method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original,
except that we add another term inside the parentheses:

​\[ \begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \\&
\text{where}\ \ L = \begin{bmatrix} 0 & & & & \\ & 1 & & &
\\ & & 1 & & \\ & & & \ddots & \\ & & & & 1
\\\end{bmatrix}\end{align*}
\]

L is a matrix with 0 at the top left and 1's down the diagonal, with
0's everywhere else. It should have dimension
(n+1)×(n+1). Intuitively, this is the identity matrix (though we are
not including x<sub>0</sub>), multiplied with a single real number λ.

Recall that if m < n, then X<sup>T</sup> X is non-invertible. However, when
we add the term λ⋅L, then X<sup>T</sup> X + λ⋅L becomes invertible.


<a id="orged2226b"></a>

### regularized logistic regression

We can regularize logistic regression in a similar way that we
regularize linear regression. As a result, we can avoid
overfitting. The following image shows how the regularized function,
displayed by the pink line, is less likely to overfit than the
non-regularized function represented by the blue line:

![img](terceira_semana/Od9mobDaEeaCrQqTpeD5ng_4f5e9c71d1aa285c1152ed4262f019c1_Screenshot-2016-11-22-09.31.21_2019-08-12_20-58-08.png)

Cost Function

Recall that our cost function for logistic regression was:

\[ J(\theta) = -\dfrac{1}{m}\sum^m_{i=1}[y^{(i)}
\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))] \]

We can regularize this equation by adding a term to the end:

\[ J(\theta) = -\dfrac{1}{m}\sum^m_{i=1}[y^{(i)}
\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))] +
\dfrac{\lambda}{2m}\sum^n_{j=1}\theta^2_j \]

The second sum, &sum;<sub>j=1</sub><sup>n</sup> &theta;<sub>j</sub><sup>2</sup> means to explicitly exclude the bias
term, &theta;<sub>0</sub>. I.e. the θ vector is indexed from 0 to n (holding n+1
values, &theta;<sub>0</sub> through &theta;<sub>n</sub>), and this sum explicitly skips &theta;<sub>0</sub>, by running
from 1 to n, skipping 0. Thus, when computing the equation, we should
continuously update the two following equations:

![img](terceira_semana/dfHLC70SEea4MxKdJPaTxA_306de28804a7467f7d84da0fe3ee9c7b_Screen-Shot-2016-12-07-at-10.49.02-PM_2019-08-12_21-11-32.png)


<a id="org385c12d"></a>

# quarta semana


<a id="orgf09d38a"></a>

## neural networks


<a id="org03d761c"></a>

### model representation I

Let's examine how we will represent a hypothesis function using neural
networks. At a very simple level, neurons are basically computational
units that take inputs (dendrites) as electrical inputs (called
"spikes") that are channeled to outputs (axons). In our model, our
dendrites are like the input features x1⋯xn, and the output is the
result of our hypothesis function. In this model our x<sub>0</sub> input node
is sometimes called the "bias unit." It is always equal to 1. In
neural networks, we use the same logistic function as in
classification,\[ \frac{1}{1 + e^{-\theta^Tx}}\], yet we sometimes call it a sigmoid (logistic) activation
function. In this situation, our "theta" parameters are sometimes
called "weights".

Visually, a simplistic representation looks like:

\[ \begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \\ \end{bmatrix}\rightarrow h_\theta(x) \]

Our input nodes (layer 1), also known as the "input layer", go into
another node (layer 2), which finally outputs the hypothesis function,
known as the "output layer".

We can have intermediate layers of nodes between the input and output
layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes
a<sup>2</sup><sub>0</sub>⋯a<sup>2</sup><sub>n</sub> and call them "activation units."

\[ \begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \\& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*} \]

If we had one hidden layer, it would look like:

\[ \begin{bmatrix}x_0 \\ x_1 \\ x_2 \\ x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \\ \end{bmatrix}\rightarrow h_\theta(x) \]

The values for each of the "activation" nodes is obtained as follows:

\[ \begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\ a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\ h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{align*} \]

This is saying that we compute our activation nodes by using a 3×4
matrix of parameters. We apply each row of the parameters to our
inputs to obtain the value for one activation node. Our hypothesis
output is the logistic function applied to the sum of the values of
our activation nodes, which have been multiplied by yet another
parameter matrix Θ<sup>(2)</sup> containing the weights for our second layer of
nodes.

Each layer gets its own matrix of weights, Θ<sup>(j)</sup>.

The dimensions of these matrices of weights is determined as follows:

If network has s<sub>j</sub> units in layer j and s<sub>j+1</sub> units in layer j+1, then
&Theta;<sup>(j)</sup> will be of dimension s<sub>j+1</sub>×(s<sub>j</sub>+1). The +1 comes from the addition
in &Theta;(j) of the "bias nodes," x<sub>0</sub> and Θ<sup>(j)</sup><sub>0</sub>. In other words the
output nodes will not include the bias nodes while the inputs
will. The following image summarizes our model representation:

![img](quarta_semana/0rgjYLDeEeajLxLfjQiSjg_0c07c56839f8d6e8d7b0d09acedc88fd_Screenshot-2016-11-22-10.08.51_2019-08-14_10-54-08.png)

Example: If layer 1 has 2 input nodes and layer 2 has 4 activation
nodes. Dimension of Θ<sup>(1)</sup> is going to be 4×3 where s<sub>j</sub> = 2 and s<sub>j+1</sub> = 4,
so s<sub>j+1</sub> &times; (s<sub>j</sub> + 1) = 4 &times; 3.


<a id="org02b7539"></a>

### model representation II

To re-iterate, the following is an example of a neural network:

\[ \begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)
\\ a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \\
a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \\
h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} +
\Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \\ \end{align*} \]

In this section we'll do a vectorized implementation of the above
functions. We're going to define a new variable z<sub>k</sub><sup>(j)</sup> that encompasses
the parameters inside our g function. In our previous example if we
replaced by the variable z for all the parameters we would get:

\[ \begin{align*}a_1^{(2)} = g(z_1^{(2)}) \\ a_2^{(2)} = g(z_2^{(2)}) \\
a_3^{(2)} = g(z_3^{(2)}) \\ \end{align*} \]

In other words, for layer j=2 and node k, the variable z will be:

\[ z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n \]

The vector representation of x and z<sup>j</sup> is:

\[ \begin{align*}x = \begin{bmatrix}x_0 \\ x_1 \\\cdots
\\ x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \\ z_2^{(j)}
\\\cdots \\ z_n^{(j)}\end{bmatrix}\end{align*} \]

Setting x = a<sup>(1)</sup>, we can rewrite the equation as:

\[ z^{(j)} = \Theta^{(j-1)}a^{(j-1)} \]

We are multiplying our matrix Θ(j−1) with dimensions s<sub>j</sub> &times; (n+1)
(where s<sub>j</sub> is the number of our activation nodes) by our vector a<sup>(j-1)</sup>
with height (n+1). This gives us our vector z<sup>(j)</sup> with height s<sub>j</sub>. Now
we can get a vector of our activation nodes for layer j as follows:

a<sup>(j)</sup> =g(z<sup>(j)</sup>)

Where our function g can be applied element-wise to our vector z<sup>(j)</sup>.

We can then add a bias unit (equal to 1) to layer j after we have
computed a<sup>(j)</sup> . This will be element a<sub>0</sub><sup>(j)</sup> and will be equal to 1. To
compute our final hypothesis, let's first compute another z vector:

\[ z^{(j+1)} = \Theta^{(j)}a^{(j)} \]

Notice that in this last step, between layer j and layer j+1, we are
doing exactly the same thing as we did in logistic regression. Adding
all these intermediate layers in neural networks allows us to more
elegantly produce interesting and more complex non-linear hypotheses.


<a id="orge10dc70"></a>

## applications


<a id="orgac8f3f7"></a>

### examples and intuitions I

A simple example of applying neural networks is by predicting x<sub>1</sub> AND
x<sub>2</sub>, which is the logical 'and' operator and is only true if
both x<sub>1</sub> and x<sub>2</sub> are 1.

The graph of our functions will look like:

\[ \begin{align*}\begin{bmatrix}x_0 \\ x_1 \\ x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*} \]

Remember that x<sub>0</sub> is our bias variable and is always 1.

Let's set our first theta matrix as:

\[ \Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \]

This will cause the output of our hypothesis to only be positive if
both x<sub>1</sub> and x<sub>2</sub> are 1. In other words:

\[ \begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \\ \\ & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \\ & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \\ & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \\ & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*} \]

So we have constructed one of the fundamental operations in computers
by using a small neural network rather than using an actual AND
gate. Neural networks can also be used to simulate all the other
logical gates. The following is an example of the logical operator
'OR', meaning either x<sub>1</sub> is true or x<sub>2</sub> is true, or both:

![img](quarta_semana/f_ueJLGnEea3qApInhZCFg_a5ff8edc62c9a09900eae075e8502e34_Screenshot-2016-11-23-10.03.48_2019-08-14_11-58-31.png)

Where g(z) is the following:

![img](quarta_semana/wMOiMrGnEeajLxLfjQiSjg_bbbdad80f5c95068bde7c9134babdd77_Screenshot-2016-11-23-10.07.24_2019-08-14_11-58-55.png)


<a id="org9b42e24"></a>

### examples and intuitions II

The Θ(1) matrices for AND, NOR, and OR are:

\[ \begin{align*}AND:\\\Theta^{(1)} &=\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \\ NOR:\\\Theta^{(1)} &= \begin{bmatrix}10 & -20 & -20\end{bmatrix} \\ OR:\\\Theta^{(1)} &= \begin{bmatrix}-10 & 20 & 20\end{bmatrix} \\\end{align*} \]

We can combine these to get the XNOR logical operator (which gives 1
if x<sub>1</sub> and x<sub>2</sub> are both 0 or both 1).

\[ \begin{align*}\begin{bmatrix}x_0 \\ x_1 \\ x_2\end{bmatrix} \rightarrow\begin{bmatrix}a_1^{(2)} \\ a_2^{(2)} \end{bmatrix} \rightarrow\begin{bmatrix}a^{(3)}\end{bmatrix} \rightarrow h_\Theta(x)\end{align*} \]

For the transition between the first and second layer, we'll use a
Θ<sup>(1)</sup> matrix that combines the values for AND and NOR:

\[ \Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20 \\ 10 & -20 & -20\end{bmatrix} \]

For the transition between the second and third layer, we'll use a
Θ<sup>(2)</sup> matrix that uses the value for OR:

\[ \Theta^{(2)} =\begin{bmatrix}-10 & 20 & 20\end{bmatrix} \]

Let's write out the values for all our nodes:

\[ \begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \\& a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \\& h_\Theta(x) = a^{(3)}\end{align*} \]

And there we have the XNOR operator using a hidden layer with two
nodes! The following summarizes the above algorithm:

![img](quarta_semana/rag_zbGqEeaSmhJaoV5QvA_52c04a987dcb692da8979a2198f3d8d7_Screenshot-2016-11-23-10.28.41_2019-08-14_13-16-52.png)


<a id="org56762e2"></a>

### multiclass classification

To classify data into multiple classes, we let our hypothesis function
return a vector of values. Say we wanted to classify our data into one
of four categories. We will use the following example to see how this
classification is done. This algorithm takes as input an image and
classifies it accordingly:

![img](quarta_semana/9Aeo6bGtEea4MxKdJPaTxA_4febc7ec9ac9dd0e4309bd1778171d36_Screenshot-2016-11-23-10.49.05_2019-08-14_14-45-35.png)

We can define our set of resulting classes as y:

![img](quarta_semana/KBpHLXqiEealOA67wFuqoQ_95654ff11df1261d935ab00553d724e5_Screenshot-2016-09-14-10.38.27_2019-08-14_14-45-49.png)

Each y<sup>(i)</sup> represents a different image corresponding to either a
car, pedestrian, truck, or motorcycle. The inner layers, each provide
us with some new information which leads to our final hypothesis
function. The setup looks like:

![img](quarta_semana/VBxpV7GvEeamBAoLccicqA_3e7f67888330b131426ecffd27936f61_Screenshot-2016-11-23-10.59.19_2019-08-14_14-46-22.png)

Our resulting hypothesis for one set of inputs may look like:

\[ h_\Theta(x) =\begin{bmatrix}0 \\ 0 \\ 1 \\ 0 \\\end{bmatrix} \]

In which case our resulting class is the third one down, or h<sub>Θ</sub>(x)<sub>3</sub>,
which represents the motorcycle.


<a id="org6f5bfa0"></a>

# quinta semana


<a id="org6216e75"></a>

## cost function and backpropagation


<a id="org09f959d"></a>

### cost function

Let's first define a few variables that we will need to use:

L = total number of layers in the network 
s<sub>l</sub> = number of units (not counting bias unit) in layer l 
K = number of output units/classes

Recall that in neural networks, we may have many output nodes. We
denote hΘ(x)<sub>k</sub> as being a hypothesis that results in the k<sup>th</sup>
output. Our cost function for neural networks is going to be a
generalization of the one we used for logistic regression. Recall that
the cost function for regularized logistic regression was:

\[ J(\theta) = - \frac{1}{m}
\sum^{m}_{i=1}[y^{(i)} \log(h_\theta(x^{(i)}))+(1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] \]

For neural networks, it is going to be slightly more complicated:

\[ J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2 \]

We have added a few nested summations to account for our multiple
output nodes. In the first part of the equation, before the square
brackets, we have an additional nested summation that loops through
the number of output nodes.

In the regularization part, after the square brackets, we must account
for multiple theta matrices. The number of columns in our current
theta matrix is equal to the number of nodes in our current layer
(including the bias unit). The number of rows in our current theta
matrix is equal to the number of nodes in the next layer (excluding
the bias unit). As before with logistic regression, we square every
term.

Note:

-   the double sum simply adds up the logistic regression costs calculated

for each cell in the output layer

-   the triple sum simply adds up the squares of all the individual Θs in

the entire network.

-   the i in the triple sum does not refer to training example i


<a id="org03fa012"></a>

### backpropagation algorithm

"Backpropagation" is neural-network terminology for minimizing our
cost function, just like what we were doing with gradient descent in
logistic and linear regression. Our goal is to compute:

\[ \min_\Theta J(\Theta) \] 

That is, we want to minimize our cost function J using an optimal set
of parameters in theta. In this section we'll look at the equations we
use to compute the partial derivative of J(Θ):

\[ \dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta) \]

To do so, we use the following algorithm:

![img](quinta_semana/Ul6i5teoEea1UArqXEX_3g_a36fb24a11c744d7552f0fecf2fdd752_Screenshot-2017-01-10-17.13.27_2019-08-15_21-35-41.png)

Back propagation Algorithm

Given training set \[ \lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace \]

-   Set &Delta;<sup>(l)</sup><sub>i,j</sub> := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

For training example t =1 to m:

1.  Set a<sup>(1)</sup> := x<sup>(t)</sup>
2.  Perform forward propagation to compute a<sup>(l)</sup> for l=2,3,…,L

![img](quinta_semana/bYLgwteoEeaX9Qr89uJd1A_73f280ff78695f84ae512f19acfa29a3_Screenshot-2017-01-10-18.16.50_2019-08-15_21-38-10.png)

1.  Using y<sup>(t)</sup>, compute &delta;<sup>(L)</sup> = a<sup>(L)</sup> - y<sup>(t)</sup>

Where L is our total number of layers and a<sup>(L)</sup> is the vector of
outputs of the activation units for the last layer. So our "error
values" for the last layer are simply the differences of our actual
results in the last layer and the correct outputs in y. To get the
delta values of the layers before the last layer, we can use an
equation that steps us back from right to left:

1.  Compute &delta;<sup>(L-1)</sup>, &delta;<sup>(L-2)</sup>,&hellip;,&delta;<sup>(2)</sup> using \[ \delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)}) \]

The delta values of layer l are calculated by multiplying the delta
values in the next layer with the theta matrix of layer l. We then
element-wise multiply that with a function called g', or g-prime,
which is the derivative of the activation function g evaluated with
the input values given by z<sup>(l)</sup>.

The g-prime derivative terms can also be written out as:

g'(z<sup>(l)</sup>) = a<sup>(l)</sup> .\* (1 - a<sup>(l)</sup>)

1.  Δ<sub>i,j</sub><sup>(l)</sup> := Δ<sub>i,j</sub><sup>(l)</sup> + a<sub>j</sub><sup>(l)</sup>δ<sub>i</sub><sup>(l+1)</sup> or with vectorization,

&Delta;<sup>(l)</sup> := &Delta;<sup>(l)</sup> + &delta;<sup>(l+1)</sup>(a<sup>(l)</sup>)<sup>T</sup>

Hence we update our new &Delta; matrix.

-   \[ D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right) \]

-   \[ D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j} \ \ \ \text{if j = 0}\]

The capital-delta matrix D is used as an "accumulator" to add up our
values as we go along and eventually compute our partial
derivative. Thus we get \[ \frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta)\]


<a id="org93db040"></a>

### backpropagation intuition

Recall that the cost function for a neural network is:

\[ J(\Theta) = - \frac{1}{m} \sum_{t=1}^m\sum_{k=1}^K \left[ y^{(t)}_k \ \log (h_\Theta (x^{(t)}))_k + (1 - y^{(t)}_k)\ \log (1 - h_\Theta(x^{(t)})_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l+1} ( \Theta_{j,i}^{(l)})^2 \]

If we consider simple non-multiclass classification (k = 1) and
disregard regularization, the cost is computed with:

\[ cost(t) =y^{(t)} \ \log (h_\Theta (x^{(t)})) + (1 - y^{(t)})\ \log (1 - h_\Theta(x^{(t)})) \]

Intuitively, &delta;<sub>j</sub><sup>(l)</sup> is the "error" for a<sup>(l)</sup><sub>j</sub> (unit j in layer l). More
formally, the delta values are actually the derivative of the cost
function:

\[ \delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t) \]

Recall that our derivative is the slope of a line tangent to the cost
function, so the steeper the slope the more incorrect we are. Let us
consider the following neural network below and see how we could
calculate some &delta;<sub>j</sub><sup>(l)</sup>:

![img](quinta_semana/qc309rdcEea4MxKdJPaTxA_324034f1a3c3a3be8e7c6cfca90d3445_fixx_2019-08-15_22-19-20.png)

In the image above, to calculate &delta;<sub>2</sub><sup>(2)</sup>, we multiply the weights Θ<sup>(2)</sup><sub>12</sub>
and Θ<sup>(2)</sup><sub>22</sub> by their respective &delta; values found to the right of each
edge. So we get &delta;<sub>2</sub><sup>(2)</sup>​ = Θ<sup>(2)</sup><sub>12</sub>\*&delta;<sub>1</sub><sup>(3)</sup>​ + Θ<sup>(2)</sup><sub>22</sub>\*&delta;<sub>2</sub><sup>(3)</sup>. To calculate
every single possible &delta;<sub>j</sub><sup>(l)</sup>, we could start from the right of our
diagram. We can think of our edges as our Θ<sub>ij</sub>. Going from right to
left, to calculate the value of &delta;<sub>j</sub><sup>(l)</sup>, you can just take the
over all sum of each weight times the &delta; it is coming
from. Hence, another example would be &delta;<sub>2</sub><sup>(3)</sup> = Θ<sup>(3)</sup><sub>12</sub>\*&delta;<sub>1</sub><sup>(4)</sup>.


<a id="orgb4484cf"></a>

## backpropagation in practice


<a id="org7c5fe93"></a>

### implementation note: unrolling parameters

With neural networks, we are working with sets of matrices:

\[ \begin{align*} \Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}, \dots \\ D^{(1)}, D^{(2)}, D^{(3)}, \dots \end{align*} \]

In order to use optimizing functions such as "fminunc()", we will want
to "unroll" all the elements and put them into one long vector:

{% highlight octave %}
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
{% endhighlight %}

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is
1x11, then we can get back our original matrices from the "unrolled"
versions as follows:

{% highlight octave %}
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
{% endhighlight %}

To summarize:

![img](quinta_semana/kdK7ubT2EeajLxLfjQiSjg_d35545b8d6b6940e8577b5a8d75c8657_Screenshot-2016-11-27-15.09.24_2019-08-16_14-17-33.png)


<a id="orge38541f"></a>

### gradient checking

Gradient checking will assure that our backpropagation works as
intended. We can approximate the derivative of our cost function with:

\[ \dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon} \]

With multiple theta matrices, we can approximate the derivative with
respect to Θ<sub>j</sub> as follows:

\[ \dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon} \]

A small value for ϵ (epsilon) such as ϵ=10<sup>-4</sup>, guarantees that the math
works out properly. If the value for ϵ is too small, we can end up
with numerical problems.

Hence, we are only adding or subtracting epsilon to the Θ<sub>j</sub> matrix. In
octave we can do it as follows:

{% highlight octave %}
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
{% endhighlight %}

We previously saw how to calculate the deltaVector. So once we compute
our gradApprox vector, we can check that gradApprox ≈ deltaVector.

Once you have verified once that your backpropagation algorithm is
correct, you don't need to compute gradApprox again. The code to
compute gradApprox can be very slow.


<a id="org95413ab"></a>

### random initialization

Initializing all theta weights to zero does not work with neural
networks. When we backpropagate, all nodes will update to the same
value repeatedly. Instead we can randomly initialize our weights for
our Θ matrices using the following method:

![img](quinta_semana/y7gaS7pXEeaCrQqTpeD5ng_8868ccda2c387f5d481d0c54ab78a86e_Screen-Shot-2016-12-04-at-11.27.28-AM_2019-08-16_15-50-08.png)

Hence, we initialize each Θ<sup>(l)</sup><sub>ij</sub> to a random value
between[−ϵ,ϵ]. Using the above formula guarantees that we get the
desired bound. The same procedure applies to all the Θ's. Below is
some working code you could use to experiment.

{% highlight octave %}
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
{% endhighlight %}

rand(x,y) is just a function in octave that will initialize a matrix
of random real numbers between 0 and 1.

(Note: the epsilon used above is unrelated to the epsilon from
Gradient Checking)


<a id="org61fb001"></a>

### putting it together

First, pick a network architecture; choose the layout of your neural
network, including how many hidden units in each layer and how many
layers in total you want to have.

-   Number of input units = dimension of features x<sup>(i)</sup>

-   Number of output units = number of classes

-   Number of hidden units per layer = usually more the better (must
    balance with cost of computation as it increases with more hidden
    units)

-   Defaults: 1 hidden layer. If you have more than 1 hidden layer, then
    it is recommended that you have the same number of units in every
    hidden layer.

Training a Neural Network

1.  Randomly initialize the weights

2.Implement forward propagation to get h<sub>Θ</sub>(x<sup>(i)</sup>) for any x<sup>(i)</sup>

1.  Implement the cost function

2.  Implement backpropagation to compute partial derivatives

3.  Use gradient checking to confirm that your backpropagation
    works. Then disable gradient checking.

4.  Use gradient descent or a built-in optimization function to
    minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every
training example:

{% highlight octave %}
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
{% endhighlight %}

The following image gives us an intuition of what is happening as we
are implementing our neural network:

![img](quinta_semana/hGk18LsaEea7TQ6MHcgMPA_8de173808f362583eb39cdd0c89ef43e_Screen-Shot-2016-12-05-at-10.40.35-AM_2019-08-16_16-53-23.png)

Ideally, you want h<sub>Θ</sub>(x(i)) ≈ y<sup>(i)</sup> . This will minimize our cost
function. However, keep in mind that J(Θ) is not convex and thus we
can end up in a local minimum instead.


<a id="org0dedd65"></a>

# sexta semana


<a id="orgeaa1584"></a>

## evaluating a learning algorithm


<a id="orgb911536"></a>

### evaluating a hypothesis

Once we have done some trouble shooting for errors in our predictions by:

Getting more training examples
Trying smaller sets of features
Trying additional features
Trying polynomial features
Increasing or decreasing λ
We can move on to evaluate our new hypothesis.

A hypothesis may have a low error for the training examples but still
be inaccurate (because of overfitting). Thus, to evaluate a
hypothesis, given a dataset of training examples, we can split up the
data into two sets: a training set and a test set. Typically, the
training set consists of 70 % of your data and the test set is the
remaining 30 %.

The new procedure using these two sets is then:

-   Learn Θ and minimize J<sub>train</sub>(Θ) using the training set

-   Compute the test set error J<sub>test</sub>(Θ)

The test set error

1.  For linear regression: \[ J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2 \]
2.  For classification ~ Misclassification error (aka 0/1 misclassification error):

\[ err(h_\Theta(x),y) = \begin{matrix} 1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline 0 & \mbox otherwise \end{matrix} \]

This gives us a binary 0 or 1 error result based on a
misclassification. The average test error for the test set is:

\[ \text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test}) \]

This gives us the proportion of the test data that was misclassified.


<a id="orgf4649e3"></a>

### model selection and train/validation/test sets

Just because a learning algorithm fits a training set well, that does
not mean it is a good hypothesis. It could over fit and as a result
your predictions on the test set would be poor. The error of your
hypothesis as measured on the data set with which you trained the
parameters will be lower than the error on any other data set.

Given many models with different polynomial degrees, we can use a
systematic approach to identify the 'best' function. In order to
choose the model of your hypothesis, you can test each degree of
polynomial and look at the error result.

One way to break down our dataset into the three sets is:

Training set: 60%
Cross validation set: 20%
Test set: 20%

We can now calculate three separate error values for the three
different sets using the following method:

Optimize the parameters in Θ using the training set for each
polynomial degree.

Find the polynomial degree d with the least error using the cross
validation set.

Estimate the generalization error using the test set with J<sub>test</sub>(Θ<sup>(d)</sup>),
(d = theta from polynomial with lower error);

This way, the degree of the polynomial d has not been trained using
the test set.


<a id="orgb62f429"></a>

## bias vs variance


<a id="org7eb0b95"></a>

### diagnosing bias vs variance

In this section we examine the relationship between the degree of the
polynomial d and the underfitting or overfitting of our hypothesis.

We need to distinguish whether bias or variance is the problem
contributing to bad predictions.  High bias is underfitting and high
variance is overfitting. Ideally, we need to find a golden mean
between these two.  The training error will tend to decrease as we
increase the degree d of the polynomial.

At the same time, the cross validation error will tend to decrease as
we increase d up to a point, and then it will increase as d is
increased, forming a convex curve.

High bias (underfitting): both J<sub>train</sub><sup>(Θ)</sup> and J<sub>CV</sub><sup>(Θ)</sup> will be high. Also, J<sub>CV</sub><sup>(Θ)</sup>≈J<sub>train</sub><sup>(Θ)</sup>.

High variance (overfitting): J<sub>train</sub><sup>(Θ)</sup> will be low and J<sub>CV</sub><sup>(Θ)</sup> will be much greater than J<sub>train</sub><sup>(Θ)</sup>.

The is summarized in the figure below:

![img](sexta_semana/I4dRkz_pEeeHpAqQsW8qwg_bed7efdd48c13e8f75624c817fb39684_fixed_2019-08-24_00-06-26.png)


<a id="orgb4fa3a3"></a>

### regulatization and bias/variance

Note: [The regularization term below and through out the video should
be \[ \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 \] and NOT \[\frac{\lambda}{2m} \sum_{j=1}^m \theta_j^2\] ]

![img](sexta_semana/3XyCytntEeataRJ74fuL6g_3b6c06d065d24e0bf8d557e59027e87a_Screenshot-2017-01-13-16.09.36_2019-08-24_00-26-19.png)

In the figure above, we see that as λ increases, our fit becomes more
rigid. On the other hand, as λ approaches 0, we tend to over
overfit the data. So how do we choose our parameter λ to get it
'just right' ? In order to choose the model and the regularization
term λ, we need to:

Create a list of lambdas
(i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});

Create a set of models with different degrees or any other variants.

Iterate through the λs and for each λ go through all the
models to learn some Θ.

Compute the cross validation error using the learned Θ (computed with
λ) on the JCV(Θ) without regularization or λ = 0.

Select the best combo that produces the lowest error on the cross
validation set.

Using the best combo Θ and λ, apply it on Jtest(Θ) to see if it has a
good generalization of the problem.


<a id="org16e36b9"></a>

### learning curves

Training an algorithm on a very few number of data points (such as 1,
2 or 3) will easily have 0 errors because we can always find a
quadratic curve that touches exactly those number of points. Hence:

As the training set gets larger, the error for a quadratic function increases.

The error value will plateau out after a certain m, or training set size.

Experiencing high bias:

Low training set size: causes J<sub>train</sub>(Θ) to be low and J<sub>CV</sub>(Θ) to be
high.

Large training set size: causes both J<sub>train</sub>(Θ) and J<sub>CV</sub>(Θ) to be high
with J<sub>train</sub>(Θ)≈J<sub>CV</sub>(Θ).

If a learning algorithm is suffering from high bias, getting more
training data will not (by itself) help much.

![img](sexta_semana/bpAOvt9uEeaQlg5FcsXQDA_ecad653e01ee824b231ff8b5df7208d9_2-am_2019-08-24_00-43-34.png)

Experiencing high variance:

Low training set size: J<sub>train</sub>(Θ) will be low and J<sub>CV</sub>(Θ) will be high.

Large training set size: J<sub>train</sub>(Θ) increases with training set size
and J<sub>CV</sub>(Θ) continues to decrease without leveling off. Also, J<sub>train</sub>(Θ)
< J<sub>CV</sub>(Θ) but the difference between them remains significant.

If a learning algorithm is suffering from high variance, getting more
training data is likely to help.

![img](sexta_semana/vqlG7t9uEeaizBK307J26A_3e3e9f42b5e3ce9e3466a0416c4368ee_ITu3antfEeam4BLcQYZr8Q_37fe6be97e7b0740d1871ba99d4c2ed9_300px-Learning1_2019-08-24_00-44-25.png)


<a id="org1957621"></a>

### deciding what to do next revisited

Our decision process can be broken down as follows:

Getting more training examples: Fixes high variance

Trying smaller sets of features: Fixes high variance

Adding features: Fixes high bias

Adding polynomial features: Fixes high bias

Decreasing λ: Fixes high bias

Increasing λ: Fixes high variance.

Diagnosing Neural Networks

-   A neural network with fewer parameters is prone to underfitting. It is

also computationally cheaper.

-   A large neural network with more parameters is prone to

overfitting. It is also computationally expensive. In this case you
can use regularization (increase λ) to address the overfitting.

Using a single hidden layer is a good starting default. You can train
your neural network on a number of hidden layers using your cross
validation set. You can then select the one that performs best.

Model Complexity Effects:

-   Lower-order polynomials (low model complexity) have high bias and
    low variance. In this case, the model fits poorly consistently.

-   Higher-order polynomials (high model complexity) fit the training
    data extremely well and the test data extremely poorly. These have
    low bias on the training data, but very high variance.

-   In reality, we would want to choose a model somewhere in between,
    that can generalize well but also fits the data reasonably well.


<a id="org8652b13"></a>

## building a spam classifier


<a id="orgc5747af"></a>

### prioritizing what to work on

System Design Example:

Given a data set of emails, we could construct a vector for each
email. Each entry in this vector represents a word. The vector
normally contains 10,000 to 50,000 entries gathered by finding the
most frequently used words in our data set. If a word is to be found
in the email, we would assign its respective entry a 1, else if it is
not found, that entry would be a 0. Once we have all our x vectors
ready, we train our algorithm and finally, we could use it to classify
if an email is a spam or not.

![img](sexta_semana/Ys5NKOLJEeaPWBJZo44gSg_aba93cf4ce4507175d7e47ab5f9b7ce4_Screenshot-2017-01-24-22.29.45_2019-08-24_18-46-11.png)

So how could you spend your time to improve the accuracy of this classifier?

-   Collect lots of data (for example "honeypot" project but doesn't
    always work)

-   Develop sophisticated features (for example: using email header data
    in spam emails)

-   Develop algorithms to process your input in different ways
    (recognizing misspellings in spam).

It is difficult to tell which of the options will be most helpful.


<a id="org569bda7"></a>

### error analysis

The recommended approach to solving machine learning problems is to:

-   Start with a simple algorithm, implement it quickly, and test it
    early on your cross validation data.

-   Plot learning curves to decide if more data, more features, etc. are
    likely to help.

-   Manually examine the errors on examples in the cross validation set
    and try to spot a trend where most of the errors were made.

For example, assume that we have 500 emails and our algorithm
misclassifies a 100 of them. We could manually analyze the 100 emails
and categorize them based on what type of emails they are. We could
then try to come up with new cues and features that would help us
classify these 100 emails correctly. Hence, if most of our
misclassified emails are those which try to steal passwords, then we
could find some features that are particular to those emails and add
them to our model. We could also see how classifying each word
according to its root changes our error rate:

![img](sexta_semana/kky-ouM6EeacbA6ydECl3A_01b1fa64fcc9a7eb5da8e946f6a12636_Screenshot-2017-01-25-12.08.23_2019-08-24_19-03-22.png)

It is very important to get error results as a single, numerical
value. Otherwise it is difficult to assess your algorithm's
performance. For example if we use stemming, which is the process of
treating the same word with different forms (fail/failing/failed) as
one word (fail), and get a 3% error rate instead of 5%, then we should
definitely add it to our model. However, if we try to distinguish
between upper case and lower case letters and end up getting a 3.2%
error rate instead of 3%, then we should avoid using this new
feature. Hence, we should try new things, get a numerical value for
our error rate, and based on our result decide whether we want to keep
the new feature or not.


<a id="orgbb245c7"></a>

# sétima semana


<a id="org562890a"></a>

## optimization objective

The Support Vector Machine (SVM) is yet another type of supervised
machine learning algorithm. It is sometimes cleaner and more powerful.

Recall that in logistic regression, we use the following rules:

if y=1, then h<sub>&theta;</sub>(x) &asymp; 1 and \[ \Theta^Tx \gg 0 \]

if y=0, then h<sub>&theta;</sub>(x) &asymp; 0 and \[ \Theta^Tx \ll 0 \]

Recall the cost function for (unregularized) logistic regression:

\[ \begin{align*}J(\theta) & = \frac{1}{m}\sum_{i=1}^m -y^{(i)} \log(h_\theta(x^{(i)})) -
(1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))\\ & = \frac{1}{m}\sum_{i=1}^m -y^{(i)}
\log\Big(\dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big) - (1 -
y^{(i)})\log\Big(1 - \dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big)\end{align*} \]

To make a support vector machine, we will modify the first term of the
cost function \[ - \log(h_\theta(x)) = - \log
\Big(\frac{1}{1+e^{-\theta^T x}}\Big) \] so that when θ<sup>T</sup> x (from now on,
we shall refer to this as z) is greater than 1, it
outputs 0. Furthermore, for values of z less than 1, we shall use a
straight decreasing line instead of the sigmoid curve.(In the
literature, this is called a hinge loss
(<https://en.wikipedia.org/wiki/Hinge_loss>) function.)

![img](s%C3%A9tima_semana/67xwSHtkEeam4BLcQYZr8Q_1877395fcce3436991415c70ed819461_Svm_hing_2019-09-01_08-28-51.png)

Similarly, we modify the second term of the cost function
 \[ -
\log(1 - h_\theta(x)) = - \log\Big(1 - \frac{1}{1+e^{-\theta^T x}}\Big) \]
 so that when z is less than -1, it outputs 0. We also modify it so
that for values of z greater than -1, we use a straight increasing
line instead of the sigmoid curve.

![img](s%C3%A9tima_semana/KKNB3HtlEeaNlA6zo4Pi2Q_135a50fd32c5eb6f4bd89b22c476c45f_Svm_hinge_negative_class_2019-09-01_08-30-56.png)

We shall denote these as cost<sub>1</sub>(z) and cost<sub>0</sub>(z) (respectively, note
that cost<sub>1</sub>(z) is the cost for classifying when y=1, and cost<sub>0</sub>(z) is
the cost for classifying when y=0), and we may define them as follows
(where k is an arbitrary constant defining the magnitude of the slope
of the line):

z = θ<sup>T</sup> x

cost0(z) = max(0,k(1+z))
cost1(z) = max(0,k(1−z))

Recall the full cost function from (regularized) logistic regression:

\[ J(\theta) = \frac{1}{m} \sum_{i=1}^m y^{(i)}(-\log(h_\theta(x^{(i)}))) + (1 -
y^{(i)})(-\log(1 - h_\theta(x^{(i)}))) + \dfrac{\lambda}{2m}\sum_{j=1}^n \Theta^2_j
\]

Note that the negative sign has been distributed into the sum in the
above equation.

We may transform this into the cost function for support vector
machines by substituting cost<sub>0</sub>(z) and cost<sub>1</sub>(z):

\[ J(\theta) = \frac{1}{m} \sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 -
y^{(i)}) \ \text{cost}_0(\theta^Tx^{(i)}) + \dfrac{\lambda}{2m}\sum_{j=1}^n
\Theta^2_j \]

We can optimize this a bit by multiplying this by m (thus removing the
m factor in the denominators). Note that this does not affect our
optimization, since we're simply multiplying our cost function by a
positive constant (for example, minimizing (u-5)<sup>2</sup> + 1 gives us 5;
multiplying it by 10 to make it 10(u-5)<sup>2</sup> + 10 still gives us 5 when
minimized).

\[ J(\theta) = \sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \
\text{cost}_0(\theta^Tx^{(i)}) + \dfrac{\lambda}{2}\sum_{j=1}^n \Theta^2_j \]

This is equivalent to multiplying the equation by \[ C = \dfrac{1}{\lambda}
\], and thus results in the same values when optimized. Now, when we
wish to regularize more (that is, reduce overfitting), we decrease C,
and when we wish to regularize less (that is, reduce underfitting), we
increase C.

Finally, note that the hypothesis of the Support Vector Machine is not
interpreted as the probability of y being 1 or 0 (as it is for the
hypothesis of logistic regression). Instead, it outputs either 1
or 0. (In technical terms, it is a discriminant function.)

\[ h_\theta(x) =\begin{cases}    1 & \text{if} \ \Theta^Tx \geq 0
\\    0 & \text{otherwise}\end{cases} \] 


<a id="orgb46d5cd"></a>

## large margin intuition

A useful way to think about Support Vector Machines is to think of
them as Large Margin Classifiers.

If y=1, we want Θ<sup>T</sup> x ≥ 1 (not just ≥0)

If y=0, we want Θ<sup>T</sup> x ≤ −1 (not just <0)

Now when we set our constant C to a very large value (e.g. 100,000),
our optimizing function will constrain Θ such that the equation A (the
summation of the cost of each example) equals 0. We impose the
following constraints on Θ:

Θ<sup>T</sup> x ≥ 1 if y=1 and Θ<sup>T</sup> x ≤ −1 if y=0.

If C is very large, we must choose Θ parameters such that:

\[ \sum_{i=1}^m y^{(i)}\text{cost}_1(\Theta^Tx) + (1 - y^{(i)})\text{cost}_0(\Theta^Tx)
= 0 \]

This reduces our cost function to:

\[ \begin{align*}
J(\theta) = C \cdot 0 + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j \\
= \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j
\end{align*} \]

Recall the decision boundary from logistic regression (the line
separating the positive and negative examples). In SVMs, the decision
boundary has the special property that it is as far away as possible
from both the positive and the negative examples.

The distance of the decision boundary to the nearest example is called
the margin. Since SVMs maximize this margin, it is often called a
Large Margin Classifier.

The SVM will separate the negative and positive examples by a large
margin.

This large margin is only achieved when C is very large.

Data is linearly separable when a straight line can separate the
positive and negative examples.

If we have outlier examples that we don't want to affect the decision
boundary, then we can reduce C.

Increasing and decreasing C is similar to respectively decreasing and
increasing λ, and can simplify our decision boundary.


<a id="orge4c34c7"></a>

## Mathematics Behind Large Margin Classification (Optional)

Vector Inner Product

Say we have two vectors, u and v:

\[ \begin{align*} u = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} &\  v
= \begin{bmatrix}v_1 \\ v_2 \end{bmatrix} \end{align*} \]

The length of vector v is denoted \[ \lVert v \rVert \], and it describes the line on
a graph from origin (0,0) to (v<sub>1</sub>, v<sub>2</sub>).

The length of vector v can be calculated with \[ \sqrt{v_1^2 + v_2^2} \]
​by the Pythagorean theorem.

The projection of vector v onto vector u is found by taking a right
angle from u to the end of v, creating a right triangle.

-   p = length of projection of v onto the vector u.

-   \[ u^T v = p \cdot \lVert u \rVert \]

Note that \[ u^T v = \lVert u\rVert \cdot \lVert v \rVert \cos \theta \] where θ is the angle
between u and v. Also, \[ p = \lVert v \rVert \cos \theta \]. If you substitute p
for \[ \lVert{v}\rVert \cos \theta \], you get \[ u^T v = p \cdot \lVert u \rVert\].

So the product u<sup>T</sup> v is equal to the length of the projection times
the length of vector u.

In our example, since u and v are vectors of the same length, \[u^T v =
v^T u \].

\[ u^T v = v^T u = p \cdot \lVert u \rVert = u_1 v_1 + u_2 v_2 \]

If the angle between the lines for v and u is greater than 90 degrees,
then the projection p will be negative.

\[ \begin{align*}&\min_\Theta \dfrac{1}{2}\sum_{j=1}^n \Theta_j^2 \\&=
\dfrac{1}{2}(\Theta_1^2 + \Theta_2^2 + \dots + \Theta_n^2) \\&= \dfrac{1}{2}(\sqrt{\Theta_1^2 +
\Theta_2^2 + \dots + \Theta_n^2})^2 \\&= \dfrac{1}{2} \lVert \Theta \rVert ^2 \\\end{align*} \]

We can use the same rules to rewrite \[ \Theta^Tx^{(i)} \]:

\[ \Theta^Tx^{(i)} = p^{(i)} \cdot \lVert \Theta \rVert = \Theta_1x_1^{(i)} + \Theta_2x_2^{(i)} +
\dots + \Theta_n x_n^{(i)} \] 

So we now have a new optimization objective by substituting \[ p^{(i)}
\cdot \lVert \Theta \rVert \] in for \[ \Theta^Tx^{(i)} \]:

If y=1, we want \[p^{(i)}⋅\lVert \theta  \rVert\geq1\]

If y=0, we want \[p^{(i)}⋅\lVert \theta \rVert \leq−1\]

The reason this causes a "large margin" is because: the vector for Θ
is perpendicular to the decision boundary. In order for our
optimization objective (above) to hold true, we need the absolute
value of our projections p<sup>(i)</sup> to be as large as possible.

If Θ<sub>0</sub>=0, then all our decision boundaries will intersect (0,0). If
Θ<sub>0</sub>≠0, the support vector machine will still find a large margin for
the decision boundary.


<a id="org0ab8d6a"></a>

## kernels I

Kernels allow us to make complex, non-linear classifiers using Support Vector Machines.

Given x, compute new feature depending on proximity to landmarks l<sup>(1)</sup>, l<sup>(2)</sup>, l<sup>(3)</sup>.

To do this, we find the "similarity" of x and some landmark l<sup>(i)</sup>:

\[ f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{\lVert x - l^{(i)} \rVert^2}{2\sigma^2}) \]

This "similarity" function is called a Gaussian Kernel. It is a
specific example of a kernel.

The similarity function can also be written as follows:

\[ f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{\sum^n_{j=1}(x_j-l_j^{(i)})^2}{2\sigma^2}) \]

There are a couple properties of the similarity function:

If x ≈ l<sup>(i)</sup>, then \[ f_i = \exp(-\dfrac{\approx 0^2}{2\sigma^2}) \approx 1 \]

If x is far from l<sup>(i)</sup>, then \[ f_i = \exp(-\dfrac{(large\
number)^2}{2\sigma^2}) \approx 0 \]

In other words, if x and the landmark are close, then the similarity
will be close to 1, and if x and the landmark are far away from each
other, the similarity will be close to 0.

Each landmark gives us the features in our hypothesis:

\[ \begin{align*}l^{(1)} \rightarrow f_1 \\ l^{(2)} \rightarrow f_2
\\ l^{(3)} \rightarrow f_3 \\\dots \\ h_\Theta(x) =
\Theta_1f_1 + \Theta_2f_2 + \Theta_3f_3 + \dots\end{align*} \]

σ<sup>2</sup> is a parameter of the Gaussian Kernel, and it can be modified to
increase or decrease the drop-off of our feature f<sub>i</sub>. Combined with
looking at the values inside Θ, we can choose these landmarks to get
the general shape of the decision boundary.


<a id="org41c2fd9"></a>

## kernels II

One way to get the landmarks is to put them in the exact same
locations as all the training examples. This gives us m landmarks,
with one landmark per training example.

Given example x:

f<sub>1</sub> = similarity(x,l<sup>(1)</sup>), f<sub>2</sub> = similarity(x,l<sup>(2)</sup>), f<sub>3</sub> = similarity(x,l<sup>(3)</sup>), and so on.

This gives us a "feature vector," f<sub>(i)</sub> of all our features for
example x<sub>(i)</sub>. We may also set f<sub>0</sub> = 1 to correspond with Θ<sub>0</sub>. Thus given
training example x<sub>(i)</sub>:

\[ x^{(i)} \rightarrow \begin{bmatrix}f_1^{(i)} = similarity(x^{(i)}, l^{(1)})
\\ f_2^{(i)} = similarity(x^{(i)}, l^{(2)}) \\\vdots \\ f_m^{(i)}
= similarity(x^{(i)}, l^{(m)}) \\\end{bmatrix} \]

Now to get the parameters Θ we can use the SVM minimization algorithm
but with f<sup>(i)</sup> substituted in for x<sup>(i)</sup>:

\[ \min_{\Theta} C \sum_{i=1}^m y^{(i)}\text{cost}_1(\Theta^Tf^{(i)}) + (1 -
y^{(i)})\text{cost}_0(\theta^Tf^{(i)}) + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j \]

Using kernels to generate f(i) is not exclusive to SVMs and may also
be applied to logistic regression. However, because of computational
optimizations on SVMs, kernels combined with SVMs is much faster than
with other algorithms, so kernels are almost always found combined
only with SVMs.


<a id="org34462cb"></a>

## choosing SVM parameters

Choosing C (recall that \[ C = \dfrac{1}{\lambda}\])
​	 

-   If C is large, then we get higher variance/lower bias
-   If C is small, then we get lower variance/higher bias

The other parameter we must choose is σ<sub>2</sub> from the Gaussian Kernel function:

With a large σ<sub>2</sub>, the features fi vary more smoothly, causing higher
bias and lower variance.

With a small σ<sub>2</sub>, the features fi vary less smoothly, causing lower
bias and higher variance.


<a id="orge43e75e"></a>

## Using An SVM

There are lots of good SVM libraries already written. A. Ng often uses
'liblinear' and 'libsvm'. In practical application, you should use one
of these libraries rather than rewrite the functions.

In practical application, the choices you do need to make are:

-   Choice of parameter C
-   Choice of kernel (similarity function)
-   No kernel ("linear" kernel) &#x2013; gives standard linear classifier
-   Choose when n is large and when m is small
-   Gaussian Kernel (above) &#x2013; need to choose σ2
-   Choose when n is small and m is large
-   The library may ask you to provide the kernel function.

Note: do perform feature scaling before using the Gaussian Kernel.

Note: not all similarity functions are valid kernels. They must
satisfy "Mercer's Theorem" which guarantees that the SVM package's
optimizations run correctly and do not diverge.

You want to train C and the parameters for the kernel function using
the training and cross-validation datasets.


<a id="org150493f"></a>

## multi-class classification

Many SVM libraries have multi-class classification built-in.

You can use the one-vs-all method just like we did for logistic
regression, where y∈1,2,3,…,K with Θ<sup>(1)</sup>,Θ<sup>(2)</sup>,…,Θ<sup>(K)</sup>. We pick class i
with the largest (Θ<sup>(i)</sup>)<sup>T</sup> x.


<a id="org9bd9a23"></a>

## logistic regression vs. SVMs

If n is large (relative to m), then use logistic regression, or SVM
without a kernel (the "linear kernel")

If n is small and m is intermediate, then use SVM with a Gaussian
Kernel

If n is small and m is large, then manually create/add more features,
then use logistic regression or SVM without a kernel.

In the first case, we don't have enough examples to need a complicated
polynomial hypothesis. In the second example, we have enough examples
that we may need a complex non-linear hypothesis. In the last case, we
want to increase our features so that logistic regression becomes
applicable.

Note: a neural network is likely to work well for any of these
situations, but may be slower to train.


<a id="org6bda59f"></a>

# oitava semana


<a id="org9701f23"></a>

## ML:Clustering


<a id="org5bdb782"></a>

### Unsupervised Learning: Introduction

Unsupervised learning is contrasted from supervised learning because
it uses an unlabeled training set rather than a labeled one.

In other words, we don't have the vector y of expected results, we
only have a dataset of features where we can find structure.

Clustering is good for:

-   Market segmentation
-   Social network analysis
-   Organizing computer clusters
-   Astronomical data analysis


<a id="org9f5d159"></a>

### k-means algorithm

The K-Means Algorithm is the most popular and widely used algorithm
for automatically grouping data into coherent subsets.

-   Randomly initialize two points in the dataset called the cluster
    centroids.
-   Cluster assignment: assign all examples into one of two groups based
    on which cluster centroid the example is closest to.
-   Move centroid: compute the averages for all the points inside each
    of the two cluster centroid groups, then move the cluster centroid
    points to those averages.
-   Re-run (2) and (3) until we have found our clusters.

Our main variables are:

-   K (number of clusters)
-   Training set x<sup>(1)</sup>, x<sup>(2)</sup>, &hellip;,x<sup>(m)</sup>
-   Where x<sup>(i)</sup>∈R<sup>n</sup>

Note that we will not use the x0=1 convention.

The algorithm:

{% highlight octave %}
Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K)
Repeat:
   for i = 1 to m:
      c(i):= index (from 1 to K) of cluster centroid closest to x(i)
   for k = 1 to K:
      mu(k):= average (mean) of points assigned to cluster k
{% endhighlight %}

The first for-loop is the 'Cluster Assignment' step. We make a vector
c where c(i) represents the centroid assigned to example x(i).

We can write the operation of the Cluster Assignment step more
mathematically as follows:

\[ c^{(i)} = argmin_k\ \lVert x^{(i)} - \mu_k \rVert^2 \]

That is, each c<sup>(i)</sup> contains the index of the centroid that has
minimal distance to x<sup>(i)</sup>.

By convention, we square the right-hand-side, which makes the function
we are trying to minimize more sharply increasing. It is mostly just a
convention. But a convention that helps reduce the computation load
because the Euclidean distance requires a square root but it is
canceled.

Without the square:

\[ \lVert x^{(i)} - \mu_k \rVert\ =\ lVert\ \sqrt{(x^1_1 - \mu_{1(k)})^2 + (x^i_2 - \mu_{2(k)})^2 + (x^i_3 -
\mu_{3(k)})^2 +...}\ \rVert \]

With the square:

\[ \lVert x^{(i)} - \mu_k \rVert\ =\ \lVert\ (x^1_1 - \mu_{1(k)})^2 + (x^i_2 - \mu_{2(k)})^2 + (x^i_3 -
\mu_{3(k)})^2 +...\ \rVert \]

..so the square convention serves two purposes, minimize more sharply
and less computation.

The second for-loop is the 'Move Centroid' step where we move each
centroid to the average of its group.

More formally, the equation for this loop is as follows:

\[ \mu_k = \dfrac{1}{n}[x^{(k_1)} + x^{(k_2)} + \dots + x^{(k_n)}] \in \mathbb{R}^n \]

Where each of x<sup>(k<sub>1</sub>)</sup>, x<sup>(k<sub>2</sub>)</sup>, &hellip;, x<sup>(k<sub>n</sub>)</sup> are the training examples
assigned to group mμ<sub>k</sub>.

If you have a cluster centroid with 0 points assigned to it, you can
randomly re-initialize that centroid to a new point. You can also
simply eliminate that cluster group.

After a number of iterations the algorithm will converge, where new
iterations do not affect the clusters.

Note on non-separated clusters: some datasets have no real inner
separation or natural structure. K-means can still evenly segment your
data into K subsets, so can still be useful in this case.


<a id="org611688e"></a>

### optimization objective

Recall some of the parameters we used in our algorithm:

-   c<sup>(i)</sup> = index of cluster (1,2,&#x2026;,K) to which example x(i) is

currently assigned 

-   &mu;<sub>k</sub> = cluster centroid k (μk∈ℝn)

-   &mu;<sub>c<sup>(i)</sup></sub> = cluster centroid of cluster to which example x(i) has
    been assigned

Using these variables we can define our cost function:

\[ J(c^{(i)},...,c^{(m)},\mu_1,..., \mu_K) = \frac{1}{m} \sum\limits^{m}_{i=1} \lVert x^{(i)} - \mu_{c^(i)}\rVert^2 \]

Our optimization objective is to minimize all our parameters using the
above cost function:

min<sub>c,μ</sub>J(c,μ)

That is, we are finding all the values in sets c, representing all our
clusters, and μ, representing all our centroids, that will minimize
the average of the distances of every training example to its
corresponding cluster centroid.

The above cost function is often called the distortion of the training
examples.

In the cluster assignment step, our goal is to:

Minimize J(…) with c<sup>(1)</sup>,&hellip;,c<sup>(m)</sup> (holding &mu;<sub>1,&hellip;</sub>, &mu;<sub>K</sub> fixed)

In the move centroid step, our goal is to:

Minimize J(…) with &mu;<sub>1</sub> ,&hellip;, &mu;<sub>K</sub> 

With k-means, it is not possible for the cost function to sometimes
increase. It should always descend.


<a id="orgaf27add"></a>

### random initialization

There's one particular recommended method for randomly initializing your cluster centroids.

-   Have K<m. That is, make sure the number of your clusters is less than

the number of your training examples.

-   Randomly pick K training examples. (Not mentioned in the lecture, but

also be sure the selected examples are unique).

-   Set &mu;<sub>1</sub> ,&hellip;, &mu;<sub>K</sub> equal to these K examples.

K-means can get stuck in local optima. To decrease the chance of this
happening, you can run the algorithm on many different random
initializations. In cases where K<10 it is strongly recommended to run
a loop of random initializations.

{% highlight octave %}
for i = 1 to 100:
   randomly initialize k-means
   run k-means to get 'c' and 'm'
   compute the cost function (distortion) J(c,m)
pick the clustering that gave us the lowest cost
{% endhighlight %}


<a id="org63dbe35"></a>

### choosing the number of clusters

Choosing K can be quite arbitrary and ambiguous.

The elbow method: plot the cost J and the number of clusters K. The
cost function should reduce as we increase the number of clusters, and
then flatten out. Choose K at the point where the cost function starts
to flatten out.

However, fairly often, the curve is very gradual, so there's no clear elbow.

Note: J will always decrease as K is increased. The one exception is
if k-means gets stuck at a bad local optimum.

Another way to choose K is to observe how well k-means performs on a
downstream purpose. In other words, you choose K that proves to be
most useful for some goal you're trying to achieve from using these
clusters.


<a id="orgf57ddfa"></a>

### bonus: discussion of the drawbacks of k-means

[stack overflow discussion](http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)

1.  first answer

    **Clustering non-clustered data**
    
    Run k-means on uniform data, and you will still get clusters! It
    doesn't tell you when the data just does not cluster, and can take
    your research into a dead end this way.
    
    ![img](oitava_semana/gn1iM_2019-09-01_23-51-27.png)
    
    **Sensitive to scale**
    
    Rescaling your datasets will completely change results. While this
    itself is not bad, not realizing that you have to spend extra
    attention to scaling your data is bad. Scaling factors are extra d
    hidden parameters in k-means that "default" to 1 and thus are easily
    overlooked, yet have a major impact (but of course this applies to
    many other algorithms, too).
    
    This is probably what you referred to as "all variables have the same
    variance". Except that ideally, you would also consider non-linear
    scaling when appropriate.
    
    Also be aware that it is only a heuristic to scale every axis to have
    unit variance. This doesn't ensure that k-means works. Scaling depends
    on the meaning of your data set. And if you have more than one
    cluster, you would want every cluster (independently) to have the same
    variance in every variable, too.
    
    Here is a classic counterexample of data sets that k-means cannot
    cluster. Both axes are i.i.d. in each cluster, so it would be
    sufficient to do this in 1 dimension. But the clusters have varying
    variances, and k-means thus splits them incorrectly.
    
    ![img](oitava_semana/tXGTo_2019-09-01_23-52-11.png)
    
    I  don't think this counterexample for k-means is covered by your points:
    
    -   All clusters are spherical (i.i.d. Gaussian).
    -   All axes have the same distribution and thus variance.
    -   Both clusters have 500 elements each.
    
    Yet, k-means still fails badly (and it gets worse if I increase the
    variance beyond 0.5 for the larger cluster) But: it is not the
    algorithm that failed. It's the assumptions, which don't hold. K-means
    is working perfectly, it's just optimizing the wrong criterion.
    
    **Even on perfect data sets, it can get stuck in a local minimum**
    
    Below is the best of 10 runs of k-means on the classic A3 data
    set. This is a synthetic data set, designed for k-means. 50 clusters,
    each of Gaussian shape, reasonably well separated. Yet, it only with
    k-means++ and 100 iterations I did get the expected result&#x2026; (below
    is 10 iterations of regular k-means, for illustration).
    
    ![img](oitava_semana/BILDt_2019-09-01_23-53-01.png)
    
    You'll quickly find many clusters in this data set, where k-means
    failed to find the correct structure. For example in the bottom right,
    a cluster was broken into three parts. But there is no way, k-means is
    going to move one of these centroids to an entirely different place of
    the data set - it's trapped in a local minimum (and this already was
    the best of 10 runs!)
    
    And there are many of such local minima in this data set. Very often
    when you get two samples from the same cluster, it will get stuck in a
    minimum where this cluster remains split, and two other clusters
    merged instead. Not always, but very often. So you need a lot of
    iterations to have a lucky pick. With 100 iterations of k-means, I
    still counted 6 errors, and with 1000 iterations I got this down to 4
    errors. K-means++ by the way it weights the random samples, works much
    better on this data set.
    
    **Means are continuous**
    
    While you can run k-means on binary data (or one-hot encoded
    categorical data) the results will not be binary anymore. So you do
    get a result out, but you may be unable to interpret it in the end,
    because it has a different data type than your original data.
    
    **Hidden assumption: SSE is worth minimizing**
    
    This is essentially already present in above answer, nicely
    demonstrated with linear regression. There are some use cases where
    k-means makes perfect sense. When Lloyd had to decode PCM signals, he
    did know the number of different tones, and least squared error
    minimizes the chance of decoding errors. And in color quantization of
    imaged, you do minimize color error when reducing the palette,
    too. But on your data, is the sum of squared deviations a meaningful
    criterion to minimize?
    
    In above counterexample, the variance is not worth minimizing, because
    it depends on the cluster. Instead, a Gaussian Mixture Model should be
    fit to the data, as in the figure below:
    
    ![img](oitava_semana/oSVXJ_2019-09-01_23-54-20.png)
    
    (But this is not the ultimate method either. It's just as easy to
    construct data that does not satisfy the "mixture of k Gaussian
    distributions" assumptions, e.g., by adding a lot of background noise)
    
    **Too easy to use badly**
    
    All in all, it's too easy to throw k-means on your data, and
    nevertheless get a result out (that is pretty much random, but you
    won't notice). I think it would be better to have a method which can
    fail if you haven't understood your data&#x2026;
    
    **K-means as quantization**
    
    If you want a theoretical model of what k-means does, consider it a
    quantization approach, not a clustering algorithm.
    
    The objective of k-means - minimizing the squared error - is a
    reasonable choice if you replace every object by its nearest
    centroid. (It makes a lot less sense if you inspect the groups
    original data IMHO.)
    
    There are very good use cases for this. The original PCM use case of
    Lloyd comes to mind, or e.g. color quanization (Wikipedia). If you
    want to reduce an image to k colors, you do want to replace every
    pixel with the nearest centroid. Minimizing the squared color
    deviation then does measure L2 optimality in image approximation using
    k colors only.
    
    This quantization is probably quite similar to the linear regression
    example. Linear regression finds the best linear model. And k-means
    finds (sometimes) the best reduction to k values of a multidimensional
    data set. Where "best" is the least squared error.
    
    IMHO, k-means is a good quantization algorithm (see the first image in
    this post - if you want to approximate the data set to two points,
    this is a reasonable choice!). If you want to do cluster analysis as
    in discover structure then k-means is IMHO not the best choice. It
    tends to cluster when there are not clusters, and it cannot recognize
    various structures you do see a lot in data.

2.  second answer

    What a great question- it's a chance to show how one would inspect the
    drawbacks and assumptions of any statistical method. Namely: make up
    some data and try the algorithm on it!
    
    We'll consider two of your assumptions, and we'll see what happens to
    the k-means algorithm when those assumptions are broken. We'll stick
    to 2-dimensional data since it's easy to visualize. (Thanks to the
    curse of dimensionality, adding additional dimensions is likely to
    make these problems more severe, not less). We'll work with the
    statistical programming language R: you can find the full code here
    (and the post in blog form here).
    
    Diversion: Anscombe's Quartet First, an analogy. Imagine someone
    argued the following:
    
    I read some material about the drawbacks of linear regression- that it
    expects a linear trend, that the residuals are normally distributed,
    and that there are no outliers. But all linear regression is doing is
    minimizing the sum of squared errors (SSE) from the predicted
    line. That's an optimization problem that can be solved no matter what
    the shape of the curve or the distribution of the residuals is. Thus,
    linear regression requires no assumptions to work.
    
    Well, yes, linear regression works by minimizing the sum of squared
    residuals. But that by itself is not the goal of a regression: what
    we're trying to do is draw a line that serves as a reliable, unbiased
    predictor of y based on x. The Gauss-Markov theorem tells us that
    minimizing the SSE accomplishes that goal- but that theorem rests on
    some very specific assumptions. If those assumptions are broken, you
    can still minimize the SSE, but it might not do anything. Imagine
    saying "You drive a car by pushing the pedal: driving is essentially a
    'pedal-pushing process.' The pedal can be pushed no matter how much
    gas in the tank. Therefore, even if the tank is empty, you can still
    push the pedal and drive the car."
    
    But talk is cheap. Let's look at the cold, hard, data. Or actually,
    made-up data.
    
    ![img](oitava_semana/1nOuj_2019-09-01_20-47-44.png) This is in fact my
    favorite made-up data: Anscombe's Quartet. Created in 1973 by
    statistician Francis Anscombe, this delightful concoction illustrates
    the folly of trusting statistical methods blindly. Each of the
    datasets has the same linear regression slope, intercept, p-value and
    R2- and yet at a glance we can see that only one of them, I, is
    appropriate for linear regression. In II it suggests the wrong shape,
    in III it is skewed by a single outlier- and in IV there is clearly no
    trend at all!
    
    One could say "Linear regression is still working in those cases,
    because it's minimizing the sum of squares of the residuals." But what
    a Pyrrhic victory! Linear regression will always draw a line, but if
    it's a meaningless line, who cares?
    
    So now we see that just because an optimization can be performed
    doesn't mean we're accomplishing our goal. And we see that making up
    data, and visualizing it, is a good way to inspect the assumptions of
    a model. Hang on to that intuition, we're going to need it in a
    minute.
    
    Broken Assumption: Non-Spherical Data You argue that the k-means
    algorithm will work fine on non-spherical clusters. Non-spherical
    clusters like&#x2026; these?
    
    ![img](oitava_semana/g5Jb8_2019-09-01_20-48-55.png)
    
    Maybe this isn't what you were expecting- but it's a perfectly
    reasonable way to construct clusters. Looking at this image, we humans
    immediately recognize two natural groups of points- there's no
    mistaking them. So let's see how k-means does: assignments are shown
    in color, imputed centers are shown as X's.
    
    ![img](oitava_semana/SlpL1_2019-09-01_20-49-21.png)
    
    Well, that's not right. K-means was trying to fit a square peg in a
    round hole- trying to find nice centers with neat spheres around them-
    and it failed. Yes, it's still minimizing the within-cluster sum of
    squares- but just like in Anscombe's Quartet above, it's a Pyrrhic
    victory!
    
    You might say "That's not a fair example&#x2026; no clustering method could
    correctly find clusters that are that weird." Not true! Try single
    linkage hierachical clustering:
    
    ![img](oitava_semana/vBuTf_2019-09-01_20-49-46.png)
    
    Nailed it! This is because single-linkage hierarchical clustering
    makes the right assumptions for this dataset. (There's a whole other
    class of situations where it fails).
    
    You might say "That's a single, extreme, pathological case." But it's
    not! For instance, you can make the outer group a semi-circle instead
    of a circle, and you'll see k-means still does terribly (and
    hierarchical clustering still does well). I could come up with other
    problematic situations easily, and that's just in two dimensions. When
    you're clustering 16-dimensional data, there's all kinds of
    pathologies that could arise.
    
    Lastly, I should note that k-means is still salvagable! If you start
    by transforming your data into polar coordinates, the clustering now
    works:
    
    ![img](oitava_semana/0sUph_2019-09-01_20-50-17.png)
    
    That's why understanding the assumptions underlying a method is
    essential: it doesn't just tell you when a method has drawbacks, it
    tells you how to fix them.
    
    Broken Assumption: Unevenly Sized Clusters What if the clusters have
    an uneven number of points- does that also break k-means clustering?
    Well, consider this set of clusters, of sizes 20, 100, 500. I've
    generated each from a multivariate Gaussian:
    
    ![img](oitava_semana/WiH4T_2019-09-01_20-50-43.png)
    
    This looks like k-means could probably find those clusters, right?
    Everything seems to be generated into neat and tidy groups. So let's
    try k-means:
    
    ![img](oitava_semana/zAI1g_2019-09-01_20-51-31.png)
    
    Ouch. What happened here is a bit subtler. In its quest to minimize
    the within-cluster sum of squares, the k-means algorithm gives more
    "weight" to larger clusters. In practice, that means it's happy to let
    that small cluster end up far away from any center, while it uses
    those centers to "split up" a much larger cluster.
    
    If you play with these examples a little (R code here!), you'll see
    that you can construct far more scenarios where k-means gets it
    embarrassingly wrong.
    
    **Conclusion: No Free Lunch**
    
    There's a charming construction in mathematical folklore, formalized
    by Wolpert and Macready, called the "No Free Lunch Theorem." It's
    probably my favorite theorem in machine learning philosophy, and I
    relish any chance to bring it up (did I mention I love this question?)
    The basic idea is stated (non-rigorously) as this: **"When averaged**
    **across all possible situations, every algorithm performs equally**
    **well."**
    
    Sound counterintuitive? Consider that for every case where an
    algorithm works, I could construct a situation where it fails
    terribly. Linear regression assumes your data falls along a line- but
    what if it follows a sinusoidal wave? A t-test assumes each sample
    comes from a normal distribution: what if you throw in an outlier? Any
    gradient ascent algorithm can get trapped in local maxima, and any
    supervised classification can be tricked into overfitting.
    
    What does this mean? It means that assumptions are where your power
    comes from! When Netflix recommends movies to you, it's assuming that
    if you like one movie, you'll like similar ones (and vice
    versa). Imagine a world where that wasn't true, and your tastes are
    perfectly random- scattered haphazardly across genres, actors and
    directors. Their recommendation algorithm would fail terribly. Would
    it make sense to say "Well, it's still minimizing some expected
    squared error, so the algorithm is still working"? You can't make a
    recommendation algorithm without making some assumptions about users'
    tastes- just like you can't make a clustering algorithm without making
    some assumptions about the nature of those clusters.
    
    So don't just accept these drawbacks. Know them, so they can inform
    your choice of algorithms. Understand them, so you can tweak your
    algorithm and transform your data to solve them. And love them,
    because if your model could never be wrong, that means it will never
    be right.


<a id="org361b7d7"></a>

## ML:Dimensionality Reduction

**Motivation I:** Data Compression

-   We may want to reduce the dimension of our features if we have a lot

of redundant data.  

-   To do this, we find two highly correlated features, plot them, and

make a new line that seems to describe both features accurately. We
place all the new features on this single line.

Doing dimensionality reduction will reduce the total data we
have to store in computer memory and will speed up our learning
algorithm.

Note: in dimensionality reduction, we are reducing our features rather
than our number of examples. Our variable m will stay the same size;
n, the number of features each example from x<sup>(1)</sup> to x<sup>(m)</sup> carries, will
be reduced.

**Motivation II:** Visualization 

It is not easy to visualize data that is more than three
dimensions. We can reduce the dimensions of our data to 3 or less in
order to plot it.

We need to find new features, z<sub>1</sub>, z<sub>2</sub> (and perhaps z<sub>3</sub>) that can
effectively summarize all the other features.

Example: hundreds of features related to a country's economic system
may all be combined into one feature that you call "Economic
Activity."


<a id="orgb09de29"></a>

### principal component analysis problem formulation

The most popular dimensionality reduction algorithm is Principal
Component Analysis (PCA)

Problem formulation

Given two features, x<sub>1</sub> and x<sub>2</sub>, we want to find a single line that
effectively describes both features at once. We then map our old
features onto this new line to get a new single feature.

The same can be done with three features, where we map them to a
plane.

The goal of PCA is to reduce the average of all the distances of every
feature to the projection line. This is the projection error.

Reduce from 2d to 1d: find a direction (a vector u<sup>(1)</sup>∈R<sup>n</sup>) onto which
to project the data so as to minimize the projection error.

The more general case is as follows:

Reduce from n-dimension to k-dimension: Find k vectors u<sup>(1)</sup>, u<sup>(2)</sup>,
&hellip;, u<sup>(k)</sup> onto which to project the data so as to minimize the
projection error.

If we are converting from 3d to 2d, we will project our data onto two
directions (a plane), so k will be 2.

PCA is not linear regression

In linear regression, we are minimizing the squared error from every
point to our predictor line. These are vertical distances.  In PCA, we
are minimizing the shortest distance, or shortest orthogonal
distances, to our data points.  More generally, in linear regression
we are taking all our examples in x and applying the parameters in Θ
to predict y.

In PCA, we are taking a number of features x<sub>1</sub>, x<sub>2</sub>, &hellip;, x<sub>n</sub>, and finding
a closest common dataset among them. We aren't trying to predict any
result and we aren't applying any theta weights to the features.


<a id="org3c2a996"></a>

### principal component analysis algorithm

Before we can apply PCA, there is a data pre-processing step we must
perform:

**Data preprocessing**

-   Given training set: x(1),x(2),…,x(m)
-   Preprocess (feature scaling/mean normalization):

\[ \mu_j = \frac{1}{m} \sum^{m}_{i=1} x^{(i)}_j \]

-   Replace each x<sub>j</sub><sup>(i)</sup> with x<sub>j</sub><sup>(i)</sup> - &mu;<sub>j</sub> ​

-   If different features on different scales (e.g., x<sub>1</sub> = size of

house, x<sub>2</sub> = number of bedrooms), scale features to have comparable
range of values.

Above, we first subtract the mean of each feature from the original
feature. Then we scale all the features \[ x_j^{(i)} = \dfrac{x_j^{(i)} -
\mu_j}{s_j} ​\]

We can define specifically what it means to reduce from 2d to 1d data
as follows:

\[ \Sigma = \dfrac{1}{m}\sum^m_{i=1}(x^{(i)})(x^{(i)})^T \]

The z values are all real numbers and are the projections of our features onto u<sup>(1)</sup>u 
(1)
 .

So, PCA has two tasks: figure out u<sup>(1)</sup>,&hellip;,u<sup>(k)</sup> and also to find z<sub>1</sub>,
z<sub>2</sub>, &hellip;, z<sub>m</sub>.

The mathematical proof for the following procedure is complicated and
beyond the scope of this course.

**1. Compute "covariance matrix"**

\[ \Sigma = \dfrac{1}{m}\sum^m_{i=1}(x^{(i)})(x^{(i)})^T \]

This can be vectorized in Octave as:

{% highlight octave %}
Sigma = (1/m) * X' * X;
{% endhighlight %}

We denote the covariance matrix with a capital sigma (which happens to
be the same symbol for summation, confusingly&#x2014;they represent
entirely different things).

Note that x<sup>(i)</sup> is an n×1 vector, (x<sup>(i)</sup>)<sup>T</sup> is an 1×n vector and X is a
m×n matrix (row-wise stored examples). The product of those will be an
n×n matrix, which are the dimensions of Σ.

**2. Compute "eigenvectors" of covariance matrix Σ**

{% highlight octave %}
[U,S,V] = svd(Sigma);
{% endhighlight %}

svd() is the 'singular value decomposition', a built-in Octave function.

What we actually want out of svd() is the 'U' matrix of the Sigma
covariance matrix: U∈R<sup>n×n</sup>. U contains u<sup>(1)</sup>,&hellip;,u<sup>(n)</sup>, which is
exactly what we want.

**3. Take the first k columns of the U matrix and compute z**

We'll assign the first k columns of U to a variable called
'Ureduce'. This will be an n×k matrix. We compute z with:

z<sup>(i)</sup> = Ureduce<sup>T</sup> &sdot; x<sup>(i)</sup>

UreduceZ<sup>T</sup> will have dimensions k×n while x(i) will have dimensions
n×1. The product Ureduce<sup>T</sup> &sdot; x<sup>(i)</sup> will have dimensions k×1.

To summarize, the whole algorithm in octave is roughly:

{% highlight octave %}
Sigma = (1/m) * X' * X; % compute the covariance matrix
[U,S,V] = svd(Sigma);   % compute our projected directions
Ureduce = U(:,1:k);     % take the first k directions
Z = X * Ureduce;        % compute the projected data points
{% endhighlight %}


<a id="org2e8c879"></a>

### reconstruction from compressed representation

If we use PCA to compress our data, how can we uncompress our data, or
go back to our original number of features?

To go from 1-dimension back to 2d we do: z∈R→x∈R<sup>2</sup>.

We can do this with the equation: x<sub>approx</sub><sup>(1)</sup> = U<sub>reduce</sub> &sdot; z<sup>(1)</sup>.

Note that we can only get approximations of our original data.

Note: It turns out that the U matrix has the special property that it
is a Unitary Matrix. One of the special properties of a Unitary Matrix
is:

U<sup>-1</sup> = U<sup>\*</sup> where the "\*" means "conjugate transpose".

Since we are dealing with real numbers here, this is equivalent to:

U<sup>-1</sup> = U<sup>T</sup> So we could compute the inverse and use that, but it would be
a waste of energy and compute cycles.


<a id="org185d683"></a>

### choosing the number of principal components

How do we choose k, also called the number of principal components?
Recall that k is the dimension we are reducing to.

One way to choose k is by using the following formula:

-   Given the average squared projection error: \[ \frac{1}{m} \sum^m_{i=1}
      \lVert x^{(i)}-x^{(i)}_{approx}\rVert^2\]

-   Also given the total variation in the data: \[ \frac{1}{m} \sum^{m}_{i=1} \lVertx^{(i)}\rVert^2 \]

-   Choose k to be the smallest value such that: \[ \frac{ \frac{1}{m}
      \sum^m_{i=1} \lVert x^{(i)}-x^{(i)}_{approx} \rVert^2}{\frac{1}{m}
      \sum^{m}_{i=1} \lVert x^{(i)} \rVert ^2} \leq 0.01 \]

In other words, the squared projection error divided by the total
variation should be less than one percent, so that 99% of the variance
is retained.

Algorithm for choosing k

-   Try PCA with k=1,2,…
-   Compute U<sub>reduce</sub>, z, x
-   Check the formula given above that 99% of the variance is retained. If not, go

to step one and increase k.  

This procedure would actually be horribly inefficient. In Octave, we
will call svd:

{% highlight octave %}
[U,S,V] = svd(Sigma)
{% endhighlight %}

Which gives us a matrix S. We can actually check for 99% of retained
variance using the S matrix as follows:

\[ \dfrac{\sum^k_{i=1}S _{ii}}{\sum^n_{i=1}S _{ii}} \geq 0.99 \]


<a id="orgf381aea"></a>

### advice for applying PCA

The most common use of PCA is to speed up supervised learning.

Given a training set with a large number of features
(e.g. x<sup>(1)</sup>,…,x<sup>(m)</sup>∈R<sup>10000</sup> ) we can use PCA to reduce the number of
features in each example of the training set (e.g. z<sup>(1)</sup>,…,z<sup>(m)</sup>∈R<sup>1000</sup>).

Note that we should define the PCA reduction from x<sup>(i)</sup> to z<sup>(i)</sup> only on
the training set and not on the cross-validation or test sets. You can
apply the mapping z(i) to your cross-validation and test sets after it
is defined on the training set.

Applications

-   Compressions

Reduce space of data

Speed up algorithm

-   Visualization of data

Choose k = 2 or k = 3

**Bad use of PCA:** trying to prevent overfitting. We might think that
reducing the features with PCA would be an effective way to address
overfitting. It might work, but is not recommended because it does not
consider the values of our results y. Using just regularization will
be at least as effective.

Don't assume you need to do PCA. **Try your full machine learning**
**algorithm without PCA first.** Then use PCA if you find that you need
it.


<a id="org4526617"></a>

# nona semana


<a id="orgbdd8465"></a>

## ML:Anomaly detection


<a id="org1f3a356"></a>

### problem motivation

Just like in other learning problems, we are given a dataset x<sup>(1)</sup>, x<sup>(2)</sup>,&hellip;,x<sup>(m)</sup>.

We are then given a new example, x<sub>test</sub>, and we want to know whether
this new example is abnormal/anomalous.

We define a "model" p(x) that tells us the probability the example is
not anomalous. We also use a threshold ϵ (epsilon) as a dividing line
so we can say which examples are anomalous and which are not.

A very common application of anomaly detection is detecting fraud:

-   x<sup>(i)</sup> = features of user i's activities
-   Model p(x) from the data.
-   Identify unusual users by checking which have p(x)<ϵ.

If our anomaly detector is flagging too many anomalous examples, then
we need to decrease our threshold ϵ


<a id="org5b60b87"></a>

### gaussian distribution

The Gaussian Distribution is a familiar bell-shaped curve that can be
described by a function \[ \mathcal{N}(\mu,\sigma^2) \]

Let x∈ℝ. If the probability distribution of x is Gaussian with mean μ,
variance &sigma;<sup>2</sup>, then:

\[ x \sim \mathcal{N}(\mu, \sigma^2)\]

The little ∼ or 'tilde' can be read as "distributed as."

The Gaussian Distribution is parameterized by a mean and a variance.

Mu, or μ, describes the center of the curve, called the mean. The
width of the curve is described by sigma, or σ, called the standard
deviation.

The full function is as follows:

\[ \large p(x;\mu,\sigma^2) = \dfrac{1}{\sigma\sqrt{(2\pi)}}e^{-\dfrac{1}{2}(\dfrac{x -
\mu}{\sigma})^2} \] 

We can estimate the parameter μ from a given dataset by simply taking
the average of all the examples:

\[ \mu = \frac{1}{m} \sum^m_{i=1} x^{(i)} \]

We can estimate the other parameter, &sigma;<sup>2</sup>, with our familiar squared
error formula:

\[ \sigma^2 = \frac{1}{m}\sum^m_{i=1}(x^{(i)} - \mu)^2 \] 


<a id="orgf83c08a"></a>

### algorithm

Given a training set of examples, \[ \lbrace x^{(1)},\dots,x^{(m)}\rbrace \]
where each example is a vector, x∈R<sup>n</sup>.

\[ p(x) = p(x_1;\mu_1,\sigma_1^2)p(x_2;\mu_2,\sigma^2_2)\cdots
p(x_n;\mu_n,\sigma^2_n) \]

In statistics, this is called an "independence assumption" on the
values of the features inside training example x.

More compactly, the above expression can be written as follows:

\[ = \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2) \]

**The algorithm**

Choose features x<sub>i</sub> that you think might be indicative of anomalous
examples.

Fit parameters &mu;<sub>1</sub> ,&hellip;,&mu;<sub>n</sub> ,&sigma;<sub>1</sub><sup>2</sup> ,&hellip;,&sigma;<sub>n</sub><sup>2</sup> 

Calculate \[ \mu_j = \frac{1}{m} \sum^m_{i=1}x^{(i)}_j \]

Calculate \[ \sigma^2 = \frac{1}{m}\sum^m_{i=1}(x^{(i)}_j - \mu_j)^2 \]

Given a new example x, compute p(x):

\[ p(x) = \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2) = \prod\limits^n_{j=1}
\dfrac{1}{\sqrt{2\pi}\sigma_j}exp(-\dfrac{(x_j - \mu_j)^2}{2\sigma^2_j}) \]

Anomaly if p(x)< &epsilon;

A vectorized version of the calculation for μ is \[ \mu = \frac{1}{m}
\sum^m_{i=1}x^{(i)} \]. 
You can vectorize &sigma;<sup>2</sup> similarly.


<a id="orge3a0f44"></a>

### Developing and Evaluating an Anomaly Detection System

To evaluate our learning algorithm, we take some labeled data,
categorized into anomalous and non-anomalous examples ( y = 0 if
normal, y = 1 if anomalous).

Among that data, take a large proportion of good, non-anomalous data
for the training set on which to train p(x).

Then, take a smaller proportion of mixed anomalous and non-anomalous
examples (you will usually have many more non-anomalous examples) for
your cross-validation and test sets.

For example, we may have a set where 0.2% of the data is anomalous. We
take 60% of those examples, all of which are good (y=0) for the
training set. We then take 20% of the examples for the
cross-validation set (with 0.1% of the anomalous examples) and another
20% from the test set (with another 0.1% of the anomalous).

In other words, we split the data 60/20/20 training/CV/test and then
split the anomalous examples 50/50 between the CV and test sets.

**Algorithm evaluation:**

Fit model p(x) on training set \[\lbrace x^{(1)},\dots,x^{(m)} \rbrace\]

On a cross validation/test example x, predict:

If p(x) < &epsilon; (**anomaly**), then y=1

If p(x) ≥ &epsilon; (**normal**), then y=0

Possible evaluation metrics (see "Machine Learning System Design" section):

-   True positive, false positive, false negative, true negative.
-   Precision/recall
-   F<sub>1</sub> score

Note that we use the cross-validation set to choose parameter &epsilon;


<a id="orgf1cd9f4"></a>

### anomaly detection vs. supervised learning

When do we use anomaly detection and when do we use supervised learning?

Use anomaly detection when&#x2026;

-   We have a very small number of positive examples (y=1 &#x2026; 0-20
    examples is common) and a large number of negative (y=0) examples.
-   We have many different "types" of anomalies and it is hard for any
    algorithm to learn from positive examples what the anomalies look
    like; future anomalies may look nothing like any of the anomalous
    examples we've seen so far.

Use supervised learning when&#x2026;

-   We have a large number of both positive and negative examples. In
    other words, the training set is more evenly divided into classes.
-   We have enough positive examples for the algorithm to get a sense of
    what new positives examples look like. The future positive examples
    are likely similar to the ones in the training set.


<a id="org1120872"></a>

### choosing what features to use

The features will greatly affect how well your anomaly detection
algorithm works.

We can check that our features are gaussian by plotting a histogram of
our data and checking for the bell-shaped curve.

Some transforms we can try on an example feature x that does not have
the bell-shaped curve are:

-   \[log(x)\]
-   \[log(x+1)\]
-   \[log(x+c)\ \text{for some constant}\]
-   \[\sqrt{x} \]
-   \[x^{1/3}\]

We can play with each of these to try and achieve the gaussian shape
in our data.

There is an error analysis procedure for anomaly detection that is
very similar to the one in supervised learning.

Our goal is for p(x) to be large for normal examples and small for
anomalous examples.

One common problem is when p(x) is similar for both types of
examples. In this case, you need to examine the anomalous examples
that are giving high probability in detail and try to figure out new
features that will better distinguish the data.

In general, choose features that might take on unusually large or
small values in the event of an anomaly.


<a id="orgd911187"></a>

### Multivariate Gaussian Distribution (Optional)

The multivariate gaussian distribution is an extension of anomaly
detection and may (or may not) catch more anomalies.

Instead of modeling p(x<sub>1</sub>),p(x<sub>2</sub>),… separately, we will model p(x) all
in one go. Our parameters will be: &mu; &isin; R<sup>n</sup> and &Sigma; &isin; R<sup>n×n</sup>

\[ p(x;\mu,\Sigma) = \dfrac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
exp(-1/2(x-\mu)^T\Sigma^{-1}(x-\mu)) \]

The important effect is that we can model oblong gaussian contours,
allowing us to better fit data that might not fit into the normal
circular contours.

Varying Σ changes the shape, width, and orientation of the
contours. Changing μ will move the center of the distribution.

Check also:

The Multivariate Gaussian Distribution
<http://cs229.stanford.edu/section/gaussians.pdf> Chuong B. Do, October
10, 2008.


<a id="org376aafa"></a>

### Anomaly Detection using the Multivariate Gaussian Distribution (Optional)

When doing anomaly detection with multivariate gaussian distribution,
we compute μ and Σ normally. We then compute p(x) using the new
formula in the previous section and flag an anomaly if p(x) < &epsilon;.

The original model for p(x) corresponds to a multivariate Gaussian
where the contours of p(x;&mu;,&Sigma;) are axis-aligned.

The multivariate Gaussian model can automatically capture correlations
between different features of x.

However, the original model maintains some advantages: it is
computationally cheaper (no matrix to invert, which is costly for
large number of features) and it performs well even with small
training set size (in multivariate Gaussian model, it should be
greater than the number of features for &Sigma; to be invertible).


<a id="orgbcfeb10"></a>

## ML: Recommender Systems


<a id="org749267a"></a>

### problem formulation

Recommendation is currently a very popular application of machine learning.

Say we are trying to recommend movies to customers. We can use the following definitions

-   n<sub>u</sub> = number of users

-   n<sub>m</sub> = number of movies

-   r(i,j) = 1 if user j has rated movie i

-   y(i,j) = rating given by user j to movie i (defined only if
    r(i,j)=1)


<a id="org8fb2ecc"></a>

### content based recommendations

We can introduce two features, x<sub>1</sub> and x<sub>2</sub> which represents how much
romance or how much action a movie may have (on a scale of 0−1).

One approach is that we could do linear regression for every single
  user. For each user j, learn a parameter θ<sup>(j)</sup> &isin; R<sup>3</sup>. Predict user j as
  rating movie i with (&theta;<sup>(j)</sup>)<sup>T</sup> x<sup>(i) </sup>stars.

-   &theta;<sup>(j)</sup> = parameter vector for user j

-   x<sup>(i)</sup> = feature vector for movie i

For user j, movie i, predicted rating: (&theta;<sup>(j)</sup>)<sup>T</sup>(x<sup>(i)</sup>)

-   m<sup>(j)</sup> = number of movies rated by user j

To learn &theta;<sup>(j)</sup>, we do the following

\[ min_{\theta^{(j)}} = \dfrac{1}{2}\displaystyle \sum_{i:r(i,j)=1}
((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum_{k=1}^n(\theta_k^{(j)})^2 \]

This is our familiar linear regression. The base of the first
summation is choosing all i such that r(i,j) = 1.

To get the parameters for all our users, we do the following:

\[min_{\theta^{(1)},\dots,\theta^{(n_u)}} = \dfrac{1}{2}\displaystyle
\sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)})^2 +
\dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2 \]

We can apply our linear regression gradient descent update using the above cost function.

The only real difference is that we eliminate the constant \[\dfrac{1}{m}\].


<a id="orgae5bd3e"></a>

### collaborative filtering algorithm

To speed things up, we can simultaneously minimize our features and our parameters:

\[ J(x,\theta) = \dfrac{1}{2} \displaystyle
\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 +
\dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u}
\sum_{k=1}^{n} (\theta_k^{(j)})^2 \]

It looks very complicated, but we've only combined the cost function
for theta and the cost function for x.

Because the algorithm can learn them itself, the bias units where x0=1
have been removed, therefore x∈ℝ<sup>n</sup> and θ∈ℝ<sup>n</sup>.

These are the steps in the algorithm:

1.  Initialize \[ x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)} \] to small
    random values. This serves to break symmetry and ensures that the
    algorithm learns features \[ x^{(i)},...,x^{(n_m) \]} that are different from
    each other.

2.  Minimize \[ J(x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}) \] using gradient descent (or
    an advanced optimization algorithm). E.g. for every

\[ j=1,...,n_u ,i=1,...n_m := x_k^{(i)} -
 \alpha\left (\displaystyle \sum_{j:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)})
 \theta_k^{(j)}} + \lambda x_k^{(i)} \right) ​ \]

\[\theta_k^{(j)} := \theta_k^{(j)} - \alpha\left (\displaystyle \sum_{i:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) x_k^{(i)}} + \lambda
 \theta_k^{(j)} \right)\]

1.  For a user with parameters θ and a movie with (learned) features x,
    predict a star rating of \[ \theta^T x\].


<a id="org67a6b80"></a>

### vectorization: low rank matrix factorization

Given matrices X (each row containing features of a particular movie)
and Θ (each row containing the weights for those features for a given
user), then the full matrix Y of all predicted ratings of all movies
by all users is given simply by: \[Y=X\Theta^T\].

Predicting how similar two movies i and j are can be done using the
distance between their respective feature vectors x. Specifically, we
are looking for a small value of \[ \lVert x^{(i)} - x^{(j)} \rVert \].


<a id="orgdab2af8"></a>

### implementation detail: mean normalization

If the ranking system for movies is used from the previous lectures,
then new users (who have watched no movies), will be assigned new
movies incorrectly. Specifically, they will be assigned θ with all
components equal to zero due to the minimization of the regularization
term. That is, we assume that the new user will rank all movies 0,
which does not seem intuitively correct.

We rectify this problem by normalizing the data relative to the
mean. First, we use a matrix Y to store the data from previous
ratings, where the ith row of Y is the ratings for the ith movie and
the jth column corresponds to the ratings for the jth user.

We can now define a vector

\[ \mu = [\mu_1, \mu_2, \dots , \mu_{n_m}] \]

such that

\[ \mu_i = \frac{\sum_{j:r(i,j)=1}{Y_{i,j}}}{\sum_{j}{r(i,j)}} \]

Which is effectively the mean of the previous ratings for the ith
movie (where only movies that have been watched by users are
counted). We now can normalize the data by subtracting u, the mean
rating, from the actual ratings for each user (column in matrix Y):

As an example, consider the following matrix Y and mean ratings μ:
​	
\[ Y = 
\begin{bmatrix}
    5 & 5 & 0 & 0  \\
    4 & ? & ? & 0  \\
    0 & 0 & 5 & 4 \\
    0 & 0 & 5 & 0 \\
\end{bmatrix}, \quad
 \mu = 
\begin{bmatrix}
    2.5 \\
    2  \\
    2.25 \\
    1.25 \\
\end{bmatrix} \]

The resulting Y′ vector is:

\[ Y' =
\begin{bmatrix}
  2.5    & 2.5   & -2.5 & -2.5 \\
  2      & ?     & ?    & -2 \\
  -.2.25 & -2.25 & 3.75 & 1.25 \\
  -1.25  & -1.25 & 3.75 & -1.25
\end{bmatrix} \]

Now we must slightly modify the linear regression prediction to
include the mean normalization term:

Now we must slightly modify the linear regression prediction to
include the mean normalization term:

\[ (\theta^{(j)})^T x^{(i)} + \mu_i \] 

Now, for a new user, the initial predicted values will be equal to the
μ term instead of simply being initialized to zero, which is more
accurate.


<a id="org073dd44"></a>

# décima semana


<a id="orgc6989a5"></a>

## learning with large datasets

We mainly benefit from a very large dataset when our algorithm has
high variance when m is small. Recall that if our algorithm has high
bias, more data will not have any benefit.

Datasets can often approach such sizes as m = 100,000,000. In this
case, our gradient descent step will have to make a summation over all
one hundred million examples. We will want to try to avoid this &#x2013; the
approaches for doing so are described below.


<a id="org7a69b39"></a>

## stochastic gradient descent

Stochastic gradient descent is an alternative to classic (or batch)
gradient descent and is more efficient and scalable to large data
sets.

Stochastic gradient descent is written out in a different but similar way:

\[ cost(\theta,(x^{(i)}, y^{(i)})) = \dfrac{1}{2}(h_{\theta}(x^{(i)}) - y^{(i)})^2 \]

The only difference in the above cost function is the elimination of
the m constant within \[\dfrac{1}{2} \].

\[ J_{train}(\theta) = \dfrac{1}{m} \displaystyle \sum_{i=1}^m
cost(\theta,(x^{(i)}, y^{(i)})) \]

J<sub>train</sub> is now just the average of the cost applied to all of our training examples.

The algorithm is as follows:

1.  Randomly 'shuffle' the dataset
2.  For i = 1&hellip; m

\[ \Theta_j := \Theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)}).x^{(i)}_j \]

This algorithm will only try to fit one training example at a
time. This way we can make progress in gradient descent without having
to scan all m training examples first. Stochastic gradient descent
will be unlikely to converge at the global minimum and will instead
wander around it randomly, but usually yields a result that is close
enough. Stochastic gradient descent will usually take 1-10 passes
through your data set to get near the global minimum.


<a id="orgf5a0b26"></a>

## mini-batch gradient descent

Mini-batch gradient descent can sometimes be even faster than
stochastic gradient descent. Instead of using all m examples as in
batch gradient descent, and instead of using only 1 example as in
stochastic gradient descent, we will use some in-between number of
examples b.

Typical values for b range from 2-100 or so.

For example, with b=10 and m=1000:

Repeat:

For i = 1,11,21,31,&hellip;,991

\[ \theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) -
y^{(k)})x_j^{(k)} \]

We're simply summing over ten examples at a time. The advantage of
computing more than one example at a time is that we can use
vectorized implementations over the b examples.


<a id="orga981d14"></a>

## stochastic gradient descent convergence

How do we choose the learning rate α for stochastic gradient descent?
Also, how do we debug stochastic gradient descent to make sure it is
getting as close as possible to the global optimum?

One strategy is to plot the average cost of the hypothesis applied to
every 1000 or so training examples. We can compute and save these
costs during the gradient descent iterations.

With a smaller learning rate, it is possible that you may get a
slightly better solution with stochastic gradient descent. That is
because stochastic gradient descent will oscillate and jump around the
global minimum, and it will make smaller random jumps with a smaller
learning rate.

If you increase the number of examples you average over to plot the
performance of your algorithm, the plot's line will become smoother.

With a very small number of examples for the average, the line will be
too noisy and it will be difficult to find the trend.

One strategy for trying to actually converge at the global minimum is
to slowly decrease α over time. For example \[\alpha =
\dfrac{const1}{iterationNumber + const2} \]

However, this is not often done because people don't want to have to
fiddle with even more parameters.


<a id="orgf520bab"></a>

## online learning

With a continuous stream of users to a website, we can run an endless
loop that gets (x,y), where we collect some user actions for the
features in x to predict some behavior y.

You can update θ for each individual (x,y) pair as you collect
them. This way, you can adapt to new pools of users, since you are
continuously updating theta.


<a id="org2c95ad7"></a>

## map reduce and data parallelism

We can divide up batch gradient descent and dispatch the cost function
for a subset of the data to many different machines so that we can
train our algorithm in parallel.

You can split your training set into z subsets corresponding to the
number of machines you have. On each of those machines calculate \[
\displaystyle \sum_{i=p}^{q}(h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \],
where we've split the data starting at p and ending at q.

MapReduce will take all these dispatched (or 'mapped') jobs and
'reduce' them by calculating:

\[ \Theta_j := \Theta_j - (\alpha/z)(temp^{(1)}_j+temp^{(2)}_j+\dots+temp^{(z)}_j) \]
\[ \text{For all}\ j = 0, \dots, n. \]

This is simply taking the computed cost from all the machines,
calculating their average, multiplying by the learning rate, and
updating theta.

Your learning algorithm is MapReduceable if it can be expressed as
computing sums of functions over the training set. Linear regression
and logistic regression are easily parallelizable.

For neural networks, you can compute forward propagation and back
propagation on subsets of your data on many machines. Those machines
can report their derivatives back to a 'master' server that will
combine them.
