import unittest
import numpy as np

import sys
sys.path.append('../AutoDiff')  # directory for the package
sys.path.append('AutoDiff')

from reverseNode import ReverseNode
from utils import *
from ad import gradientR, reverse_auto_diff, auto_diff

class ReverseNodeTests(unittest.TestCase):
  
  def test_init(self):
    value = 0.1
    func = ReverseNode(value)
    assert func.value == 0.1
    assert func.adjoint == 1
    assert func.children == []

  def test_add(self):
    value = 0.1
    x = ReverseNode(value)
    func = x + 2
    assert func.value == 2.1
    assert x.gradient() == 1

  def test_sub(self):
    value = 0.1
    x = ReverseNode(value)
    func = x - 2
    assert func.value == -1.9
    assert x.gradient() == 1

  def test_mul(self):
    value = 0.1
    x = ReverseNode(value)
    func = x * 2
    assert func.value == 0.2
    assert x.gradient() == 2

  def test_pow(self):
    value = 2
    x = ReverseNode(value)
    func = x ** 2
    assert func.value == 4
    assert x.gradient() == 4.0
    
  # Constant
  def test_constant_val(self):
    value = 2.0
    x = ReverseNode(value)
    func = x
    assert func.value == 2.0
    assert x.gradient() == 1.0

  # Sine Function
  def test_sin_val(self):
    value = 0
    x = ReverseNode(value)
    func = 2 * sin(x)
    assert func.value == 0
    assert x.gradient() == 2.0

  # Cosine Function
  def test_cos_val(self):
    value = 0
    x = ReverseNode(value)
    func = 2 * cos(x)
    assert func.value == 2.0
    assert x.gradient() == 0

  def test_log_val(self):
    value = 2.0
    x = ReverseNode(value)
    func = 2 * log(x)
    assert round(func.value, 4) == 1.3863
    assert x.gradient() == 1.0

  # Exponential Function
  def test_exp_val(self):
    value = np.log(2.0)
    x = ReverseNode(value)
    func = 2 * exp(x)
    assert func.value == 4.0
    assert x.gradient() == 4.0

  # Square root
  def test_sqrt_val(self):
    value = 4
    x = ReverseNode(value)
    func = 2 * sqrt(x)
    assert func.value == 4.0
    assert x.gradient() == 0.5

  # Tan Function
  def test_tan_val(self):
    value = 0
    x = ReverseNode(value)
    func = 2 * tan(x)
    assert func.value == 0
    assert x.gradient() == 2

  # Arctan Function
  def test_arctan_val(self):
    value = 0
    x = ReverseNode(value)
    func = arctan(x)
    assert func.value == 0
    assert x.gradient() == 1.0

  # Arcsin Function
  def test_arcsin_val(self):
    value = 0
    x = ReverseNode(value)
    func = 2 * arcsin(x)
    assert func.value == 0
    assert x.gradient() == 2.0

  # Arccos Function
  def test_arccos_val(self):
    value = 0
    x = ReverseNode(value)
    func = 2 * arccos(x)
    assert func.value == np.pi
    assert x.gradient() == -2

  # Tanh Function
  def test_tanh_val(self):
    value = 0
    x = ReverseNode(value)
    func = tanh(x)
    assert func.value == 0
    assert x.gradient() == 1

  # Sinh Function
  def test_sinh_val(self):
    value = 0
    x = ReverseNode(value)
    func = sinh(x)
    assert func.value == 0.0
    assert x.gradient() == 1.0

  # Cosh Function
  def test_cosh(self):
    value = 0
    x = ReverseNode(value)
    func = cosh(x)
    assert func.value == 1.0
    assert x.gradient() == 0.0

  # Sec function
  def test_sec(self):
    value = 0.0
    x = ReverseNode(value)
    func = sec(x) * 2
    assert func.value == 2.0
    assert x.gradient() == 0.0

  # Csc function
  def test_csc(self):
    value = np.pi / 2
    x = ReverseNode(value)
    func = csc(x) * 2
    assert func.value == 2.0
    assert round(x.gradient(), 18) == -1.22e-16

  # Cot function
  def test_cot(self):
    value = np.pi / 2
    x = ReverseNode(value)
    func = cot(x) * 2
    assert round(func.value, 18) == 1.22e-16
    assert x.gradient() == -2.0

  def test_log_base(self):
    value = 10 ** 5
    x = ReverseNode(value)
    func = log_base(x, base=10)
    assert func.value == 5.0
    assert round(x.gradient(), 8) == 4.34e-06

  def test_comparison(self):
    x = ReverseNode(2.5)
    y = ReverseNode(4)
    z = ReverseNode(np.pi)
    assert x < y
    assert x <= y
    assert y > z
    assert y >= z
    assert y == x + 1.5
    assert x != z

  def test_negation(self):
    value = 3.0
    x = ReverseNode(value)
    func = -x
    assert func.value == -3.0
    assert x.gradient() == -1.0

  def test_math(self):
    x = ReverseNode(1.0)
    y1 = sin(2 * x) / cos(x / 7) + x ** 5

    assert round(y1.value, 4) == 1.9187
    assert round(x.gradient(), 4) == 4.1780

    x.gradient_reset()
    y2 = exp(x) + sqrt(tan(x) * 100) - 30

    assert round(y2.value, 4) == -14.8021
    assert round(x.gradient(), 4) == 16.4427
    assert y1 > y2

  def test_derivative(self):
    var_dict = {"x1": 2}
    functions = "3 * x1 + x1 ** 2 - exp(x1)"
    res = gradientR(functions, var_dict, target=["x1"])
    assert [[round(r, 4) for r in row] for row in res] == [[-0.3891]]

  def test_partial_derivative(self):
    var_dict = {'x1': 2, 'x2': 3, 'x3': 4}
    functions = "3 * x1 + x2 ** 2 - exp(x3)"

    res1 = gradientR(functions, var_dict, target=["x1"])
    res2 = gradientR(functions, var_dict, target=["x2"])
    res3 = gradientR(functions, var_dict, target=["x3"])

    assert res1 == [[3.0]]
    assert res2 == [[6.0]]
    assert [[round(r, 4) for r in row] for row in res3] == [[-54.5982]]

  def test_gradient(self):
    var_dict = {'x1': 2}
    functions = ["3 * x1 + x1 ** 2 - exp(x1)", "log(x1) / sin(x1) + cos(x1) ** 2"]
    res = gradientR(functions, var_dict, target=["x1"])

    assert [[round(r, 4) for r in row] for row in res] == [[-0.3891], [1.6555]]

  def test_jacobian(self):
    var_dict = {'x1': 2, 'x2': 3, 'x3': 4}
    functions = ["3 * x1 + x2 ** 2 - exp(x3)", "log(x1) / sin(x2) + cos(x3) ** 2"]
    res = gradientR(functions, var_dict, target=["x1", "x2", "x3"])

    assert [[round(r, 4) for r in row] for row in res] == [[3, 6, -54.5982], [3.5431, 34.4572, -0.9894]]

  def test_reverse_auto_diff(self):
    functions = ["x1 + sin(x2) * 5", "exp(x1) - log(2 * x2)"]
    var_dict = {"x1": 5, "x2": 2}
    target = ["x1"]

    res = reverse_auto_diff(functions, var_dict, target)

    assert [[round(r, 4) for r in row] for row in res] == [[1.0, 148.4132]]

  def test_auto_diff_derivative(self):
    functions = ["x1 + sin(x1) * 5"]
    var_dict = {"x1": 5}
    target = ["x1"]

    res = auto_diff(functions, var_dict, target, mode="reverse")

    assert [[round(r, 4) for r in row] for row in res] == [[2.4183]]

  def test_auto_diff_partial_derivative(self):
    functions = ["4.5 * x1 + sin(x2) * 5"]
    var_dict = {"x1": 5, "x2": 2}

    res1 = auto_diff(functions, var_dict, ["x1"], mode="reverse")
    res2 = auto_diff(functions, var_dict, ["x2"], mode="reverse")

    assert res1 == [[4.5]]
    assert [[round(r, 4) for r in row] for row in res2] == [[-2.0807]]

  def test_auto_diff_graidnet(self):
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

    assert [[round(r, 4) for r in row] for row in res] == [[0.1588, 30.0536, 0.0], [1.0, -1.5708, 0.0], [0.3536, 0.0, 1.0]]

  def test_repr(self):
    x = ReverseNode(value=5)
    assert repr(x) == "ReverseNode Variable Value: 5, Adjoint: 1.0, Chidren: []"

  def test_str(self):
    x = ReverseNode(value=5)
    assert str(x) == "ReverseNode Variable Value: 5, Adjoint: 1.0, Chidren: []"


if __name__ == "__main__":
  unittest.main()
