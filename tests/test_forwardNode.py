import unittest
import numpy as np

import sys
sys.path.append('../AutoDiff')  # directory for the package
sys.path.append('AutoDiff')

from forwardNode import ForwardNode
from utils import *
from ad import init_trace, create_node, gradientF, forward_auto_diff, auto_diff


class ForwardNodeTests(unittest.TestCase):

  # Checking to see if the variables are initialized correctly
  def test_init(self):
    value = 5.0
    trace = 1.0
    var = "x1"
    func = ForwardNode(value, trace=trace, var=var)
    assert func.value == value and func.trace == np.array([trace]) and func.var == [var]

  def test_init_vec(self):
    value = 5.0
    trace = [1.0, 0.0]
    var = ["x1", "x2"]
    func = ForwardNode(value, trace=trace, var=var)
    assert func.value == value and all(func.trace == np.array(trace)) and func.var == var

  def test_add(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x + 2.0
    assert func.value == 7.0 and all(func.trace == np.array(trace)) and func.var == var

  # Testing Addition with reverse order
  def test_radd(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2.0 + x
    assert func.value == 7.0 and all(func.trace == np.array(trace)) and func.var == var

  # Subtraction
  def test_sub(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x - 2.0
    assert func.value == 3.0 and all(func.trace == np.array(trace)) and func.var == var

  # Testing Subtraction with reverse order
  def test_rsub(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2.0 - x
    assert func.value == -3.0 and all(func.trace == -1 * np.array(trace)) and func.var == var

  # Multiplication
  def test_mul(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x * 2.0
    assert func.value == 10.0 and all(func.trace == 2 * np.array(trace)) and func.var == var

  # Reverse Multiplication
  def test_rmul(self):
    value = 5.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2.0 * x
    assert func.value == 10.0 and all(func.trace == 2 * np.array(trace)) and func.var == var

  # Division
  def test_truediv(self):
    value = 6.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x / 2.0
    assert func.value == 3.0 and all(func.trace == np.array(trace) / 2) and func.var == var

  # Reverse Division
  def test_rtruediv(self):
    value = 2.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2.0 / x
    assert func.value == 1.0 and all(func.trace == np.array([-0.5, -0.5, 0.0])) and func.var == var

  # Power
  def test_pow(self):
    value = 2.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x ** 2.0
    assert func.value == 4.0 and all(func.trace == np.array([4.0, 4.0, 0.0])) and func.var == var

  # Constant
  def test_constant(self):
    value = 2.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = x
    assert func.value == 2.0 and all(func.trace == np.array(trace)) and func.var == var

  # Sine Function
  def test_sin(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * sin(x)
    assert func.value == 0 and all(func.trace == np.array([2.0, 2.0, 0.0])) and func.var == var

  # Cosine Function
  def test_cos(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * cos(x)
    assert func.value == 2.0 and all(func.trace == np.array([0.0, 0.0, 0.0])) and func.var == var

  # Logarithmic Function
  def test_log(self):
    value = 1
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * log(x)
    assert func.value == 0.0 and all(func.trace == np.array([2.0, 2.0, 0.0])) and func.var == var

  # Exponential Function
  def test_exp(self):
    value = np.log(2.0)
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * exp(x)
    assert func.value == 4.0 and all(func.trace == np.array([4.0, 4.0, 0.0])) and func.var == var

  # Square root
  def test_sqrt(self):
    value = 4
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * sqrt(x)
    assert func.value == 4.0 and all(func.trace == np.array([0.5, 0.5, 0.0])) and func.var == var

  # Tan Function
  def test_tan(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * tan(x)
    assert func.value == 0 and all(func.trace == np.array([2.0, 2.0, 0.0])) and func.var == var

  # Arctan Function
  def test_arctan(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = arctan(x)
    assert func.value == 0.0 and all(func.trace == np.array([1.0, 1.0, 0.0])) and func.var == var

  # Arcsin Function
  def test_arcsin(self):
    value = 0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * arcsin(x)
    assert func.value == 0 and all(func.trace == np.array([2.0, 2.0, 0.0])) and func.var == var

  # Arccos Function
  def test_arccos(self):
    value = 0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = 2 * arccos(x)
    assert func.value == np.pi and all(func.trace == np.array([-2.0, -2.0, 0.0])) and func.var == var

  # Tanh Function
  def test_tanh(self):
    value = 0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = tanh(x)
    assert func.value == 0 and all(func.trace == np.array([1.0, 1.0, 0.0])) and func.var == var

  # Sinh Function
  def test_sinh(self):
    value = 0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = sinh(x)
    assert func.value == 0 and all(func.trace == np.array([1.0, 1.0, 0.0])) and func.var == var

  # Cosh Function
  def test_cosh(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = cosh(x)
    assert func.value == 1.0 and all(func.trace == np.array([0.0, 0.0, 0.0])) and func.var == var

  # Sec function
  def test_sec(self):
    value = 0.0
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = sec(x) * 2
    assert func.value == 2.0 and all(func.trace == np.array([0.0, 0.0, 0.0])) and func.var == var

  # Csc function
  def test_csc(self):
    value = np.pi / 2
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = csc(x) * 2
    assert func.value == 2.0 and all([round(x, 18) for x in func.trace] == np.array([-1.22e-16, -1.22e-16, 0.0])) and func.var == var

  # Cot function
  def test_cot(self):
    value = np.pi / 2
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = cot(x) * 2
    assert round(func.value, 18) == 1.22e-16 and all(func.trace == np.array([-2.0, -2.0, 0.0])) and func.var == var

  def test_log_base(self):
    value = 10 ** 5
    trace = [1.0, 1.0, 0.0]
    var = ["x1", "x2", "x3"]
    x = ForwardNode(value, trace=trace, var=var)
    func = log_base(x, base=10)
    assert func.value == 5.0 and all([round(x, 8) for x in func.trace] == np.array([4.34e-06, 4.34e-06, 0.0])) and func.var == var

  def test_comparison(self):
    x = ForwardNode(value=3.0, trace=[1.0, 0.0], var=["x1", "x2"])
    y = ForwardNode(value=5.0, trace=[1.0, 0.0], var=["x1", "x2"])
    z = ForwardNode(value=3.5, trace=[1.0, 0.0], var=["x1", "x2"])
    assert x < y
    assert x <= y
    assert y > z
    assert y >= z
    assert z == x + 0.5
    assert x != z

  def test_negation(self):
    value = 3.0
    trace = [1.0, 0.0]
    var = ["x1", "x2"]
    x = ForwardNode(value=3.0, trace=[1.0, 0.0], var=["x1", "x2"])
    func = -x
    assert func.value == -1 * value and all(func.trace == -1 * np.array(trace)) and func.var == var

  def test_math(self):
    x = ForwardNode(value=1, trace=1, var="x1")
    y1 = sin(2 * x) / cos(x / 7) + x ** 5
    y2 = exp(x) + sqrt(tan(x) * 100) - 30

    assert round(y1.value, 4) == 1.9187 and round(y1.trace[0], 4) == 4.1780 and y1.var == ["x1"]
    assert round(y2.value, 4) == -14.8021 and round(y2.trace[0], 4) == 16.4427 and y2.var == ["x1"]
    assert y1 > y2

  def test_init_trace(self):
    variables = ["x1", "x2", "x3"]
    x1_trace = init_trace("x1", variables)
    x2_trace = init_trace("x2", variables)
    x3_trace = init_trace("x3", variables)
    assert all(x1_trace == [1.0, 0.0, 0.0])
    assert all(x2_trace == [0.0, 1.0, 0.0])
    assert all(x3_trace == [0.0, 0.0, 1.0])

  def test_create_node(self):
    variables = ["x1", "x2", "x3"]
    x1 = create_node(var='x1', value=2, variables=variables)
    x2 = create_node(var='x2', value=3, variables=variables)
    x3 = create_node(var='x3', value=4, variables=variables)
    assert x1.value == 2 and all(x1.trace == np.array([1.0, 0.0, 0.0])) and x1.var == variables
    assert x2.value == 3 and all(x2.trace == np.array([0.0, 1.0, 0.0])) and x2.var == variables
    assert x3.value == 4 and all(x3.trace == np.array([0.0, 0.0, 1.0])) and x3.var == variables

  def test_derivative(self):
    variables = ['x1']

    x1 = create_node(var='x1', value=2, variables=variables)

    y = 3 * x1 + x1 ** 2 - exp(x1)
    der = gradientF(y, variables, target='x1')
    assert round(der, 4) == -0.3891

  def test_partial_derivative(self):
    variables = ['x1', 'x2', 'x3']

    x1 = create_node(var='x1', value=2, variables=variables)
    x2 = create_node(var='x2', value=3, variables=variables)
    x3 = create_node(var='x3', value=4, variables=variables)

    y = 3 * x1 + x2 ** 2 - exp(x3)

    der1 = gradientF(y, variables, target='x1')
    der2 = gradientF(y, variables, target='x2')
    der3 = gradientF(y, variables, target='x3')

    assert der1 == 3
    assert der2 == 6
    assert round(der3, 4) == -54.5982

  def test_gradient(self):
    variables = ['x1']

    x1 = create_node(var='x1', value=2, variables=variables)

    y1 = 3 * x1 + x1 ** 2 - exp(x1)
    y2 = log(x1) / sin(x1) + cos(x1) ** 2
    y = np.array([y1, y2])
    grad = gradientF(y, variables, target='x1')

    assert [round(g, 4) for g in grad] == [-0.3891, 1.6555]

  def test_jacobian(self):
    variables = ['x1', 'x2', 'x3']

    x1 = create_node(var='x1', value=2, variables=variables)
    x2 = create_node(var='x2', value=3, variables=variables)
    x3 = create_node(var='x3', value=4, variables=variables)

    y1 = 3 * x1 + x2 ** 2 - exp(x3)
    y2 = log(x1) / sin(x2) + cos(x3) ** 2
    y = np.array([y1, y2])
    jcb = gradientF(y, variables)

    assert [[round(r, 4) for r in row] for row in jcb] == [[3, 6, -54.5982], [3.5431, 34.4572, -0.9894]]

  def test_forward_auto_diff(self):
    functions = ["x1 + sin(x2) * 5", "exp(x1) - log(2 * x2)"]
    var_dict = {"x1": 5, "x2": 2}
    target = ["x1"]

    res = forward_auto_diff(functions, var_dict, target)

    assert [[round(r, 4) for r in row] for row in res] == [[1.0, 148.4132]]

  def test_auto_diff_derivative(self):
    functions = ["x1 + sin(x1) * 5"]
    var_dict = {"x1": 5}
    target = ["x1"]

    res = auto_diff(functions, var_dict, target, mode="forward")

    assert [[round(r, 4) for r in row] for row in res] == [[2.4183]]

  def test_auto_diff_partial_derivative(self):
    functions = ["4.5 * x1 + sin(x2) * 5"]
    var_dict = {"x1": 5, "x2": 2}

    res1 = auto_diff(functions, var_dict, ["x1"], mode="forward")
    res2 = auto_diff(functions, var_dict, ["x2"], mode="forward")

    assert res1 == [[4.5]]
    assert [[round(r, 4) for r in row] for row in res2] == [[-2.0807]]

  def test_auto_diff_gradient(self):
    functions = ["4.5 * x1 + sin(x2) * 5", "log(x1) + x2 ** 4 / exp(x2)"]
    var_dict = {"x1": 5, "x2": 2}

    res1 = auto_diff(functions, var_dict, ["x1"], mode="forward")
    res2 = auto_diff(functions, var_dict, ["x2"], mode="forward")

    assert res1 == [[4.5, 0.2]]
    assert [[round(r, 4) for r in row] for row in res2] == [[-2.0807, 2.1654]]

  def test_auto_diff_jacobian(self):
    functions = ["tanh(x1) + cosh(x2 * 3) - sec(x3)", "x1 / x2 * cos(x3)", "sin(x1 / 2) + x2 * x3"]
    var_dict = {"x1": np.pi / 2, "x2": 1, "x3": 0}

    res = auto_diff(functions, var_dict, ["x1", "x2", "x3"], mode="forward")

    assert [[round(r, 4) for r in row] for row in res] == [[ 0.1588, 30.0536, 0.0], [ 1.0, -1.5708, 0.0], [ 0.3536, 0.0, 1.0]]

  def test_repr(self):
    x = ForwardNode(value=5, trace=1, var="x1")
    assert repr(x) == "ForwardNode Variable: ['x1'],  Value: 5, Trace: [1]"

  def test_str(self):
    x = ForwardNode(value=5, trace=1, var="x1")
    assert str(x) == "ForwardNode Variable: ['x1'],  Value: 5, Trace: [1]"


if __name__ == "__main__":
  unittest.main()
