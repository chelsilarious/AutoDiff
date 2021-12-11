# Group 24, cs107-FinalProject [![Build Status](https://app.travis-ci.com/cs107-runtimeterror/cs107-FinalProject.svg?token=stMPL4xedtyEMYyN72oW&branch=milestone1b-dev)](https://app.travis-ci.com/cs107-runtimeterror/cs107-FinalProject) [![codecov](https://codecov.io/gh/cs107-runtimeterror/cs107-FinalProject/branch/final/graph/badge.svg?token=FF27EQ75ID)](https://codecov.io/gh/cs107-runtimeterror/cs107-FinalProject) [![CircleCI](https://circleci.com/gh/cs107-runtimeterror/cs107-FinalProject/tree/final.svg?style=svg&circle-token=a541ffb380dd87b8b5e70a86f8ac3a5f5857e6c8)](https://circleci.com/gh/cs107-runtimeterror/cs107-FinalProject/tree/final)

## Group Members: 
Chelsea (Zixi) Chen: zixichen@g.harvard.edu  
Kexin Yang: kexin_yang@hsph.harvard.edu  
Kazi Tasnim: kazitasnim@college.harvard.edu  
Andrew Sima: asima@college.harvard.edu  

## Introduction

This is a package that offers the feature of automatic differentiation.

Automatic differentiation is useful in many fields, including but not limit to:

Calculation of derivatives when using some iterative methods to solve linear systems
Calculation of the gradient of an objective function in optimization
Calculation of derivatives/gradients which are parts of some numerical methods to solve differential equation systems
Automatic differentiation is better than other differencing methods like finite-difference because it is much cheaper. Finite differences are expensive, since you need to do a forward pass for each derivative. Automatic differentiation is both efficient (linear in the cost of computing the value) and numerically stable. Traditional methods of differentiation such as symbolic differentiation do not scale well to vector functions with multiple variable inputs, which are widely used to solve real world problems.

The functions and features in this package can evaluate derivatives/gradients of specified expressions and free users from manual calculation.

## Documentation

The full documentation can be found [here](https://github.com/cs107-runtimeterror/cs107-FinalProject/blob/final/docs/documentation.md).

## Quick Start on How to use

Install the package using `pip` command like below:

```
pip install AutoDiff-RunTimeTerror
```

```python
import AutoDiff-RunTimeTerror as ad
```

### Scalar Function with Scalar Input:

```python
f = lambda x: sin(x) + cos(x)  # or f = "sin(x) + cos(x)"
var = {"x": np.pi}
der = ad.auto_diff(functions=f, var_dict=var)
```
```
Functions: ['sin(x) + cos(x)']
Variables: {'x': 3.141592653589793}
------------------------------
Derivative:
 [[-1.0000000000000002]]
```

```python
print(der)
```
```
[[-1.0000000000000002]]
```


### Scalar Function with Vector Input:

```python
f = lambda x1, x2, x3: in(x1) + cos(x2) - exp(x3)  # or f = "sin(x1) + cos(x2) - exp(x3)"
vars = {"x1": np.pi/2, "x2": 1, "x3": 0}
der2 = auto_diff(functions=f, var_dict=vars, target=["x2"], mode="reverse")
```
```
Functions: ['sin(x1) + cos(x2) - exp(x3)']
Variables: {'x1': 1.5707963267948966, 'x2': 1, 'x3': 0}
------------------------------
Partial Derivative with respect to x2: [-0.8414709848078965]
```

```python
print(der2)
```
```
[[-0.8414709848078965]]
```

```python
grad = auto_diff(functions=f, var_dict=vars, mode="reverse")
```
```
Functions: ['sin(x1) + cos(x2) - exp(x3)']
Variables: {'x1': 1.5707963267948966, 'x2': 1, 'x3': 0}
------------------------------
Derivative:
 [[6.123233995736766e-17, -0.8414709848078965, -1.0]]
```

```python
print(grad)
```
```
 [[6.123233995736766e-17, -0.8414709848078965, -1.0]]
```


### Vector Function with Scalar Input
```python
fs = lambda x1: [sec(x1), x1/cos(x1), sin(x1) + x1]  
# or fs = ["sec(x1)", "x1/cos(x1)", "sin(x1) + x1"]
var = {"x1": np.pi/3}
ders = auto_diff(functions=fs, var_dict=var, mode="forward")
```
```
Functions: ['sec(x1)', 'x1/cos(x1)', 'sin(x1) + x1']
Variables: {'x1': 1.0471975511965976}
------------------------------
Jacobian:
 [[3.46410162]
 [5.62759873]
 [1.5       ]]
```

```python
print(ders)
```
```
[[3.46410162]
 [5.62759873]
 [1.5       ]]
```


### Vector Function with Vector Input:
```python
fs = lambda x1, x2, x3: ["tanh(x1) + cosh(x2 * 3) - sec(x3)", "x1 / x2 * cos(x3)", "sin(x1 / 2) + x2 * x3"]
# or fs = ["tanh(x1) + cosh(x2 * 3) - sec(x3)", "x1 / x2 * cos(x3)", "sin(x1 / 2) + x2 * x3"]
vars = {"x1": np.pi/2, "x2": 1, "x3": 0}
grad_x1 = auto_diff(f, vars, ["x1"], mode="reverse")
```
```
Functions: ['tanh(x1) + cosh(x2 * 3) - sec(x3)', 'x1 / x2 * cos(x3)', 'sin(x1 / 2) + x2 * x3']
Variables: {'x1': 1.5707963267948966, 'x2': 1, 'x3': 0}
------------------------------
Gradient with respect to x1: [0.15883159318006335, 1.0, 0.3535533905932738]
```

```python
jcb = auto_diff(functions=fs, var_dict=vars, mode="reverse")
```
```
Functions: ['tanh(x1) + cosh(x2 * 3) - sec(x3)', 'x1 / x2 * cos(x3)', 'sin(x1 / 2) + x2 * x3']
Variables: {'x1': 1.5707963267948966, 'x2': 1, 'x3': 0}
------------------------------
Jacobian:
 [[ 0.15883159 30.05362478  0.        ]
 [ 1.         -1.57079633  0.        ]
 [ 0.35355339  0.          1.        ]]
```

```python
print(jcb)
```
```
[[ 0.15883159 30.05362478  0.        ]
 [ 1.         -1.57079633  0.        ]
 [ 0.35355339  0.          1.        ]]
```



#### Inclusivity Statement
Over the past few years, people have put in an increased effort to bridge the gap in STEM between underrepresented groups and inclusivity. However, even with this increased effort there is much more that can and should be done to fill this gap. While creating our software, we kept in mind that people from different backgrounds and experience levels would access this. Therefore, we tried to add docstrings and the proper documentation in order to make this software as accessible as possible. However, we do understand that there is more work that needs to be done to make our software more accessible and user-friendly. Currently, our software is targeted towards those who are familiar with English mathematical terms and symbols. Our software is catered towards the average English speaker. Moving forward, to make our software more inclusive, we would try to make it more accessible for those who are not as familiar with the English language.

Furthermore, Harvard's diversity statement says, "[their] commitment to diversity in all forms is rooted in [the] fundamental belief that engaging with unfamiliar ideas, perspectives, cultures, and people creates the conditions for dramatic and meaningful growth." Our team believes that by engaging with the software we have created, we are sharing our ideas and perspectives on a certain way to solve a problem. However, we are open to suggestions and any feedback our users have. We are constantly seeking to improve the way we implemented our software.

#### Broader Impact
We understand that our software has both positive and negative implications. However, we believe the positive implications outweigh the negative ones. Our team has simply found one way to tackle the problem using Automatic Differentiation and believe that we are adding to the diversity of technology in the community by contributing our software. Furthermore, by using this software, we hope that users will be able to solve different real-world problems. However, we do understand that there is more that can be done to make this software inclusive as mentioned above.


