## Introduction
This is a package that offers the feature of automatic differentiation. 

Automatic differentiation is useful in many fields, including but not limit to:
- Calculation of derivatives when using some iterative methods to solve linear systems
- Calculation of the gradient of an objective function in optimization
- Calculation of derivatives/gradients which are parts of some numerical methods to solve differential equation systems

The functions and features in this package can evaluate derivatives of specified functions and free users from manual calculation.


## Background

For a function, even a complicated one, the computer is able to compute its derivatives by breaking it down into smaller parts, applying chain rule to the elementary operations, and calculate intermerdiate results at each step. 

In the graph structure of such calculation, each node is an intermediate result, and each arrow is an elementary operation. An elementary operation are such as addition, subtraction, multiplication, division, or taking exponential, log, sine, cosine, etc. An example is provided below.

<img src="https://latex.codecogs.com/svg.latex?f(x,y)&space;=&space;sin(x)&space;-&space;y^2,&space;\quad&space;v_{-1}&space;=&space;x,&space;\quad&space;v_0&space;=&space;y" title="f(x,y) = sin(x) - y^2, \quad v_{-1} = x, \quad v_0 = y" /></a>

<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;&v_1&space;=&space;sin(v_{-1})&space;=&space;sin(x),&space;\\&space;&v_2&space;=&space;v_0^2&space;=&space;y^2,&space;\quad&space;v_3&space;=&space;-v_2&space;=&space;-y^2,&space;\quad&space;v4&space;=&space;v_1\&space;&plus;&space;\&space;v_3&space;=&space;sin(x)&space;-&space;y^2&space;=&space;f(x,&space;y)&space;\end{aligned}" title="\begin{aligned} &v_1 = sin(v_{-1}) = sin(x), \\ &v_2 = v_0^2 = y^2, \quad v_3 = -v_2 = -y^2, \quad v4 = v_1\ + \ v_3 = sin(x) - y^2 = f(x, y) \end{aligned}" /></a>

![AD_example.png](AD_example.png)


## How to use

## Software Organizatoin

## Implementation

## License

