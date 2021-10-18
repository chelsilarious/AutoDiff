## Introduction
This is a package that offers the feature of automatic differentiation. 

Automatic differentiation is useful in many fields, including but not limit to:
- Calculation of derivatives when using some iterative methods to solve linear systems
- Calculation of the gradient of an objective function in optimization
- Calculation of derivatives/gradients which are parts of some numerical methods to solve differential equation systems

The functions and features in this package can evaluate derivatives/gradients of specified expressions and free users from manual calculation.


## Background

For a function, even a complicated one, the computer is able to compute its derivatives by breaking it down into smaller parts, applying chain rule to the elementary operations, and calculate intermerdiate results at each step. 

In the graph structure of such calculation, each node is an intermediate result, and each arrow is an elementary operation. An elementary operation are such as addition, subtraction, multiplication, division, or taking exponential, log, sine, cosine, etc. 

An example is provided below.

<img src="https://latex.codecogs.com/svg.latex?f(x,y)&space;=&space;sin(x)&space;-&space;y^2,&space;\quad&space;v_{-1}&space;=&space;x,&space;\quad&space;v_0&space;=&space;y" title="f(x,y) = sin(x) - y^2, \quad v_{-1} = x, \quad v_0 = y" /></a>

<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;&v_1&space;=&space;sin(v_{-1})&space;=&space;sin(x),&space;\\&space;&v_2&space;=&space;v_0^2&space;=&space;y^2,&space;\quad&space;v_3&space;=&space;-v_2&space;=&space;-y^2,&space;\quad&space;v4&space;=&space;v_1\&space;&plus;&space;\&space;v_3&space;=&space;sin(x)&space;-&space;y^2&space;=&space;f(x,&space;y)&space;\end{aligned}" title="\begin{aligned} &v_1 = sin(v_{-1}) = sin(x), \\ &v_2 = v_0^2 = y^2, \quad v_3 = -v_2 = -y^2, \quad v4 = v_1\ + \ v_3 = sin(x) - y^2 = f(x, y) \end{aligned}" /></a>

![AD_example.png](AD_example.png)


## How to use

### Installation

```
python -m pip install -i https://test.pypi.org/simple/cs107_ADpackage
```

You are recommended to use the package under Python version 3.6.2 or later. 

###  Demo

Import package

```python
import cs107_ADpackage as ad
```

Specify problem and draw the graph structure of the automatic differentiation

```python
f = lambda x, y: np.sin(x) - y**2

func = ad.objective(formula = f)
func.comp_graph()
```

Get the first derivatives of the function using backward propagation

```python
inputs = {x: 5, y: 6}
drvt = func.backward(degree=1, input=inputs)
print(drvt)
```


## Software Organizatoin

#### What will the directory structure look like?
cs107project/

src/

#### What modules do you plan on including? What is their basic functionality?
We plan on using NumPy, Matplotlib, PyTest and PyTorch. We intend to use NumPy to create matrices and perform elementary calculations, Matplotlib to properly portray our results on graphs, PyTest to run tests, and PyTorch to perform benchmarks on these tests.

#### Where will your test suite live? Will you use TravisCI? CodeCov?
Our test suite will live in TravisCI which will be in the /tests directory.

#### How will you distribute your package (e.g. PyPI)?
We will distribute our package by uploading it to PyPI so everyone can use it.

#### How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
We will not be packing out software. The code will be on GitHub and PyPI so it will be accessible by everyone.

#### Other considerations?
As of right now we are still working on this project, so we could potentially make changes to the software later.

## Implementation

#### What are the core data structures?

#### What classes will you implement?

#### What method and name attributes will your classes have?

#### What external dependencies will you rely on?

#### How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?


## License

The license that we decided to choose is the MIT License. We chose this license because our research showed that this license is usually the one that developers choose if they want their software to be easily accessible and quickly distributed to other developers and others in the community. We ultimately settled for this license because we believe in allowing other developers to freely use the software written for their desired purposes.
