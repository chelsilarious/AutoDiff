import unittest
from cs107_package.src.reverseNode import ReverseNode
from cs107_package.src.utils import *

class ReverseNodeTests(unittest.TestCase):
  
  def test_init(self):
    value = 0.1
    func = ReverseNode(value)
    self.assertEqual(func.value, 0.1)
    self.assertEqual(func.adjoint, 1)
    self.assertEqual(func.children, [])

  def test_gradient(self):
    value = 0.1
    x = ReverseNode(value)
    self.assertEqual(x.gradient(), 1)
    func = x * ReverseNode(2.0)
    func2 = x + ReverseNode(1.0)
    self.assertNotEqual(x.children, [])
    self.assertEqual(x.gradient(), 3)

  def test_add_(self):
    value = 0.1
    x = ReverseNode(value)
    func = x + ReverseNode(2.0)
    self.assertEqual(func.value, 2.1)

  def test_sub_(self):
    value = 0.1
    x = ReverseNode(value)
    func = x - ReverseNode(2.0)
    self.assertEqual(func.value, -1.9)

  def test_mul_(self):
    value = 0.1
    x = ReverseNode(value)
    func = x * ReverseNode(2.0)
    self.assertEqual(func.value, 0.2)

  def test_pow_(self):
    value = 2
    x = ReverseNode(value)
    func = x ** ReverseNode(2.0)
    self.assertEqual(func.value, 4)
    
  # Constant
  def test_constant_val(self):
    value = 2.0
    x = ReverseNode(value)
    func = x
    assert func.value == 2.0

  def test_constant_adjoint(self):
    value = 2.0
    x = ReverseNode(value)
    func = x
    assert x.gradient() == 1.0

  # Sine Function
  def test_sin_val(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * sin(x)
    assert func.value == 0

  def test_sin_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * sin(x)
    assert x.gradient() == 2.0

  # Cosine Function
  def test_cos_val(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * cos(x)
    assert func.value == 2.0

  def test_cos_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * cos(x)
    assert x.gradient() == 0

    # Logarithmic Function

  def test_log_val(self):
    value = 2.0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * log(x)
    assert func.value == 1.3862943611198906

  def test_log_adjoint(self):
    value = 2.0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * log(x)
    assert x.gradient() == 1.0

  # Exponential Function
  def test_exp_val(self):
    value = np.log(2.0)
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * exp(x)
    assert func.value == 4.0

  def test_exp_adjoint(self):
    value = np.log(2.0)
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * exp(x)
    assert x.gradient() == 4.0

  # Square root
  def test_sqrt_val(self):
    value = 4
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * sqrt(x)
    assert func.value == 4.0

  def test_sqrt_adjoint(self):
    value = 4
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * sqrt(x)
    assert x.gradient() == 0.5

  # Tan Function
  def test_tan_val(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * tan(x)
    assert func.value == 0

  def test_tan_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * tan(x)
    assert x.gradient() == 2

  # Arctan Function
  def test_arctan_val(self):
    value = 0
    x = ReverseNode(value)
    func = arctan(x)
    assert func.value == 0

  def test_arctan_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = arctan(x)
    assert x.gradient() == 1.0

  # Arcsin Function
  def test_arcsin_val(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * arcsin(x)
    assert func.value == 0

  def test_arcsin_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * arcsin(x)
    assert x.gradient() == 2.0

  # Arccos Function
  def test_arccos_val(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * arccos(x)
    assert func.value == np.pi

  def test_arccos_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = constant(2.0, 'reverse') * arccos(x)
    assert x.gradient() == -2

  # Tanh Function
  def test_tanh_val(self):
    value = 0
    x = ReverseNode(value)
    func = tanh(x)
    assert func.value == 0

  def test_tanh_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = tanh(x)
    assert x.gradient() == 1

  # Sinh Function
  def test_sinh_val(self):
    value = 0
    x = ReverseNode(value)
    func = sinh(x)
    self.assertEqual(func.value, 0)

  def test_sinh_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = sinh(x)
    self.assertEqual(x.gradient(), 1.0)

  # Cosh Function
  def test_cosh_val(self):
    value = 0
    x = ReverseNode(value)
    func = cosh(x)
    self.assertEqual(func.value, 1.0)

  def test_cosh_adjoint(self):
    value = 0
    x = ReverseNode(value)
    func = cosh(x)
    self.assertEqual(x.gradient(), 0)

if __name__ == "__main__":
  unittest.main()
