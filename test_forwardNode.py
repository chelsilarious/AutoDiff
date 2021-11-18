# need to import the ForwardNode Function
import unittest
import numpy as np
from src.forwardNode import ForwardNode
from src.utils import *

class ForwardNodeTests(unittest.TestCase):

  # Checking to see if the variables are initialized correctly
  def test_init_value(self):
    value = 5.0
    trace = 1.0
    func = ForwardNode(value, trace = trace)
    self.assertEqual(func.value, value) # Check to see if the function value is correct

  def test_init_trace(self):
    value = 5.0
    trace = 1.0
    func = ForwardNode(value, trace = trace)
    self.assertEqual(func.trace, trace) # Check to see if the derivative is correct

  # Testing Addition
  def test_add_value(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x + 2.0
    self.assertEqual(func.value, 7.0)

  def test_add_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x + 2.0
    self.assertEqual(func.trace, 1.0)

  # Testing Addition with reverse order
  def test_radd_val(self):
    value = 5.0 
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 + x
    self.assertEqual(func.value, 7.0)

  def test_radd_trace(self):
    value = 5.0 
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 + x
    self.assertEqual(func.trace, 1.0)

  # Subtraction
  def test_sub_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x - 2.0
    self.assertEqual(func.value, 3.0)

  def test_sub_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x - 2.0
    self.assertEqual(func.trace, 1.0)

  # Testing Subtraction with reverse order
  def test_rsub_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 - x
    self.assertEqual(func.value, -3.0)

  def test_rsub_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 - x
    self.assertEqual(func.trace, -1.0)

  # Multiplication 
  def test_mul_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x * 2.0
    self.assertEqual(func.value, 10.0)
  
  def test_mul_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x * 2.0
    self.assertEqual(func.trace, 2.0)

  # Reverse Multiplication
  def test_rmul_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 * x
    self.assertEqual(func.value, 10.0)
  
  def test_rmul_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 * x
    self.assertEqual(func.trace, 2.0)

  # Division
  def test_truediv_val(self):
    value = 6.0
    x = ForwardNode(value, trace = 1.0)
    func = x / 2.0
    self.assertEqual(func.value, 3.0)
  
  def test_truediv_trace(self):
    value = 6.0
    x = ForwardNode(value, trace = 1.0)
    func = x / 2.0
    self.assertEqual(func.trace, 0.5)

  # Reverse Division
  def test_rtruediv_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 / x
    self.assertEqual(func.value, 1.0)

  def test_rtruediv_trace(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 / x
    self.assertEqual(func.trace, 2.0)

  # Power
  def test_pow_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = x**2.0
    self.assertEqual(func.value, 4.0) ## FINISH THIS ONE

  # Constant
  def test_constant_val(self):
    value = 2.0
    x = constant(value, mode='forward')
    func = x
    self.assertEqual(func.value, 2.0)

  def test_constant_trace(self):
    value = 2.0
    x = constant(value, mode='forward')
    func = x
    self.assertEqual(func.trace, 0)

  # Sine Function
  def test_sin_val(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2*sin(x)
    self.assertEqual(func.value, 0)

  def test_sin_trace(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2*sin(x)
    self.assertEqual(func.trace, -2.0)
  
  # Cosine Function
  def test_cos_val(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2*cos(x)
    self.assertEqual(func.value, -2.0)

  def test_cos_trace(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2*cos(x)
    self.assertEqual(func.trace, 0)
  
  # Logarithmic Function
  def test_log_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2*log(x)
    self.assertEqual(func.value, 1.3862943611198906)
  
  def test_log_trace(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2*log(x)
    self.assertEqual(func.trace, 1.0)

  # Exponential Function
  def test_exp_val(self):
    value = np.log(2.0)
    x = ForwardNode(value, trace = 1.0)
    func = 2 * exp(ForwardNode(x))
    self.assertEqual(func.value, 8.0)

  def test_exp_trace(self):
    value = np.log(2.0)
    x = ForwardNode(value, trace = 1.0)
    func = 2 * exp(ForwardNode(x))
    self.assertEqual(func.trace, 8.0)

  # Square root 
  def test_sqrt_val(self):
    value = 4
    x = ForwardNode(value, trace = 1.0)
    func = 2 * sqrt(ForwardNode(x))
    self.assertEqual(func.value, 4.0)

  def test_sqrt_trace(self):
    value = 4
    x = ForwardNode(value, trace = 1.0)
    func = 2 * sqrt(ForwardNode(x))
    self.assertEqual(func.trace, 0.5)

  # Tan Function
  def test_tan_val(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2 * tan(ForwardNode(x))
    self.assertEqual(func.value, 0.0)

  def test_tan_trace(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2 * tan(ForwardNode(x))
    self.assertEqual(func.trace, 2)

  # Arctan Function
  def test_arctan_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = arctan(x)
    self.assertEqual(func.value, 0)

  def test_arctan_trace(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = arctan(x)
    self.assertEqual(func.trace, 1)

  # Arcsin Function
  def test_arcsin_val(self):
    value = 0
    x = ForwardNode(value, trace = 0)
    func = 2*arcsin(x)
    self.assertEqual(func.value, 0)

  def test_arcsin_trace(self):
    value = 0
    x = ForwardNode(value, trace = 0)
    func = 2*arcsin(x)
    self.assertEqual(func.trace, 2.0)

  # Arccos Function
  def test_arccos_val(self):
    value = 1.0
    x = ForwardNode(value, trace = 0)
    func = 2*arccos(x)
    self.assertEqual(func.value, 0)

  def test_arccos_trace(self):
    value = 1.0
    x = ForwardNode(value, trace = 0)
    func = 2*arccos(x)
    self.assertEqual(func.trace, 2.0)
    
    # Tanh Function
  def test_tanh_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = tanh(x)
    assert func.value == 0

  def test_tanh_trace(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = tanh(x)
    assert func.trace == 0

  # Sinh Function
  def test_sinh_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = sinh(x)
    self.assertEqual(func.value, 0)

  def test_sinh_trace(self):
    value = 0 
    x = ForwardNode(value, trace = 1.0)
    func = sinh(x)
    self.assertEqual(func.value, 1.0)

  # Cosh Function
  def test_cosh_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = sinh(x)
    self.assertEqual(func.value, 1.0)

  def test_cosh_trace(self):
    value = 0 
    x = ForwardNode(value, trace = 1.0)
    func = sinh(x)
    self.assertEqual(func.value, 0)
  
   # Log Base
  def test_log_val(self):
    value = 10
    x = ForwardNode(value, trace = 1.0)
    func = np.log(x)
    self.assertEqual(func.value, 1.0)

  def test_log_trace(self):
    value = 10
    x = ForwardNode(value, trace = 1.0)
    func = np.log(x)
    self.assertEqual(func.trace, 0.10)

  # Cotangent
  def test_cot_val(self):
    value = 3.14/2
    x = ForwardNode(value, trace = 1.0)
    func = 1/tan(x)
    self.assertEqual(func.value, 0)

  def test_cot_trace(self):
    value = 3.14/2
    x = ForwardNode(value, trace = 1.0)
    func = 1/tan(x)
    self.assertEqual(func.trace, -1.0)

  # Secant
  def test_sec_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = 1/tan(x)
    self.assertEqual(func.value, 1)

  def test_sec_trace(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = 1/tan(x)
    self.assertEqual(func.trace, 0)



