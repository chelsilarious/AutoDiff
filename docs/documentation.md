
# Documentation

## Introduction

This is a package that offers the feature of automatic differentiation.

Automatic differentiation is useful in many fields, including but not limit to:

- Calculation of derivatives when using some iterative methods to solve linear systems
- Calculation of the gradient of an objective function in optimization
- Calculation of derivatives/gradients which are parts of some numerical methods to solve differential equation systems

Automatic differentiation is better than other differencing methods like finite-difference because it is much cheaper. Finite differences are expensive, since you need to do a forward pass for each derivative. Automatic differentiation is both efficient (linear in the cost of computing the value) and numerically stable. Traditional methods of differentiation such as symbolic differentiation do not scale well to vector functions with multiple variable inputs, which are widely used to solve real world problems.

The functions and features in this package can evaluate derivatives/gradients of specified expressions and free users from manual calculation.


## Background

For a function, even a complicated one, the computer is able to compute its derivatives by breaking it down into smaller parts, applying chain rule to the elementary operations, and calculate intermerdiate results at each step.

In the graph structure of such calculation, each node is an intermediate result, and each arrow is an elementary operation. An elementary operation are such as addition, subtraction, multiplication, division, or taking exponential, log, sine, cosine, etc. In short, AD represent a function as a composition of elementary functions through elemtary operations by a sequence of intermediate values.

An example is provided below.

\begin{aligned}
&f(x,y) = \sin(x) - y^2, \quad v_{-1} = x, \quad v_0 = y \\
&v_1 = \sin(v_{-1}) = \sin(x), \quad v_2 = v_0^2 = y^2, v_3 = -v_2 = -y^2, \quad v_4 = v_1 + v_3 = \sin(x) - y^2 = f(x,y)
\end{aligned}

![AD_example](images/AD_example.png)

In forward mode, AD starts from the inputs and work towards the outputs, evaluating the value of each intermediate value along with its derivative with respect to a fixed input variable using the chain rule.

<img src="https://latex.codecogs.com/svg.latex?\dot{v}_k&space;=&space;\frac{\partial{v_k}}{\partial{x_i}}&space;=&space;\sum_{v_m&space;\in&space;\text{parent}(v_k)}&space;\frac{\partial{v_k}}{\partial{v_m}}&space;\frac{\partial{v_m}}{\partial{x_i}}" title="\dot{v}_k = \frac{\partial{v_k}}{\partial{x_i}} = \sum_{v_m \in \text{parent}(v_k)} \frac{\partial{v_k}}{\partial{v_m}} \frac{\partial{v_m}}{\partial{x_i}}" /></a>

In the example above, a trace table for forward AD would look like the following to compute and store intermediate values and derivatives:

![foward_tracetable_example.png](images/foward_tracetable_example.png)

In reverse mode, AD starts from the inputs to do a forward pass to calculate all the intermediate values, and then starts from the outputs to do a reverse pass to compute the derivatives of the function with respect to the intermediate values backwards using the chain rule.

<img src="https://latex.codecogs.com/svg.latex?\bar{v}_k&space;=&space;\frac{\partial{f}}{\partial{v_k}}&space;=&space;\sum_{v_n&space;\in&space;\text{child}(v_k)}&space;\frac{\partial{f}}{\partial{v_n}}&space;\frac{\partial{v_n}}{\partial{v_k}}" title="\bar{v}_k = \frac{\partial{f}}{\partial{v_k}} = \sum_{v_n \in \text{child}(v_k)} \frac{\partial{f}}{\partial{v_n}} \frac{\partial{v_n}}{\partial{v_k}}" /></a>

![reverse_tracetable_example](images/reverse_tracetable_example.png)


## How to use

### Installation

TODO

You are recommended to use the package under Python version 3.6.2 or later. 

### Demo

TODO

## Software Organization

### Directory Structure

```
cs107project/
├── LICENSE
├── README.md
└── AutoDiff/
    ├── __init__.py
    ├── ad.py
    ├── fowardNode.py
    ├── reverseNode.py
    └── utils.py    
└── docs/
    ├── README.md
    ├── milestone1.md
    ├── milestone2_progress.md
    ├── milestone2.ipynb
    ├── documentation.md
    └── images/
        └── ...
└── tests/
    ├── __init__.py
    ├── test_forwardNode.py
    └── test_reverseNode.py
├── .travis.yml
└── .circleci/
    └── config.yml
└── requirements.txt
```

### Included Modules and their Basic Functionality
We are using NumPy, UnitTest and PyTest. We use NumPy to create numpy arrays for easier vectorized calculations in the overloaded elementary functions, and UnitTest and PyTest to run tests on our code.

### Test Suite
Our test suite will live in the /tests directory and it will be tested by CircleCI.

### Package Distribution
We will distribute our package by uploading it to PyPI so everyone can use it.

### Notes
We will not be packing out software. The code will be on GitHub and PyPI so it will be accessible by everyone.

As of right now we are still working on this project, so we could potentially make changes to the software later.


## Implementation

### Core Data Structures

The core data structure is a `Node` structure that is able to represent all the intermediate function expressions. Every instance of a `Node` stores the actual value of the variable and has an attribute storing derivatives.

The exact structures of `Node` for forward mode and for reverse mode are different. More details can be found below.

### Classes

1. `class ForwardNode`: This is the most generic base class to accomodate for the different nodes in the AD structure in Forward Mode. 

2. `class ReverseNode`: This is the most generic base class to accomodate for the different nodes in the AD structure in Reverse Mode. 


### Methods and Name Attributes

The `ForwardNode` class has 3 attributes:
- `self.value`: the actual value of the function expression <img src="https://latex.codecogs.com/svg.latex?v_k" title="v_k" /></a> represented by a ForwardNode instance
- `self.trace`: a numpy array of the gradients <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;v_k}{\partial&space;x_i}" title="\frac{\partial v_k}{\partial x_i}" /></a> of this intermediate function expression with respect to the target input variables <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" /></a>
- `self.var`: a list of the names of the variable names <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_i" title="x_i" /></a>

The `ReverseNode` class has 3 attributes:
- `self.value`: the actual value of the function expression $v_k$ represented by a ReverseNode instance
- `self.adjoint`: a value of gradient <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;f_i}{\partial&space;v_k}" title="\frac{\partial f_i}{\partial v_k}" /></a> of the ultimate function expression with respect to the intermediate variable <img src="https://latex.codecogs.com/svg.latex?v_k" title="v_k" /></a>
- `self.children`: a list of tuples, with each of the tuple storing <img src="https://latex.codecogs.com/svg.latex?v_k" title="v_k" /></a>'s children <img src="https://latex.codecogs.com/svg.latex?v_n" title="v_n" /></a> and the derivatives <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;v_n}{\partial&space;v_k}" title="\frac{\partial v_n}{\partial v_k}" /></a> in the form (<img src="https://latex.codecogs.com/svg.latex?v_n" title="v_n" /></a>, <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;v_n}{\partial&space;v_k}" title="\frac{\partial v_n}{\partial v_k}" /></a>)

### External Dependencies

The implenentation is based heavily on numpy in the overloaded functions to do vectorized operations, and also in the wrap-up functions for easy calculation of gradients.

### Elementary Functions

The elementary operations are overloaded in this class. Doing any one of the operation would return a new `ForwardNode` instance that represents the new intermediate function expression, and it would contain the attributes mentioned above.

Below, `variables` is a list of names of the variables all the functions are derived with respect to. <a href="https://www.codecogs.com/eqnedit.php?latex=v_1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?v_1" title="v_1" /></a> is an instance of ForwardNode, representing a variable (function) having value `value1` and its derivatives with respect to the variables whose names are in `variables` are in `trace1`. <img src="https://latex.codecogs.com/svg.latex?v_2" title="v_2" /></a> is an instance of ForwardNode, representing a variable (function) having value `value2` and its derivatives with respect to the variables whose names are in `variables` are in `trace2`. 

```python
variables = ...
v1 = ForwardNode(value1, trace1, variables)
v2 = ForwardNode(value2, trace2, variables)
```

#### Binary elementary functions

- Addition: <img src="https://latex.codecogs.com/svg.latex?v_k&space;=&space;v_1&space;&plus;&space;v_2&space;\&space;\Rightarrow&space;\&space;\dot{v}_k&space;=&space;1&space;\cdot&space;\dot{v}_1&space;&plus;&space;1&space;\cdot&space;\dot{v}_2" title="v_k = v_1 + v_2 \ \Rightarrow \ \dot{v}_k = 1 \cdot \dot{v}_1 + 1 \cdot \dot{v}_2" /></a>

```python
valuek, tracek = value1+value2, trace1+trace2
vk = ForwardNode(valuek, tracek, variables)
```
