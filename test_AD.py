# need to import the ForwardNode Function
import pytest
import numpy as np

class Tests:

  # Checking to see if the variables are initialized correctly
  def test_init_value(self):
    value = 5.0
    trace = 1.0
    func = ForwardNode(value, trace = trace)
    assert func.val == value # Check to see if the function value is correct

  def test_init_trace(self):
    value = 5.0
    trace = 1.0
    func = ForwardNode(value, trace = trace)
    assert func.trace == trace # Check to see if the derivative is correct

  # Testing Addition
  def test_add_value(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x + 2.0
    assert func.value == 7.0

  def test_add_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x + 2.0
    assert func.trace == 1.0

  # Testing Addition with reverse order
  def test_radd_val(self):
    value = 5.0 
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 + x
    assert func.value == 1.0

  def test_radd_trace(self):
    value = 5.0 
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 + x
    assert func.trace == 1.0

  # Subtraction
  def test_sub_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x - 2.0
    assert func.value == 3.0

  def test_sub_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x - 2.0
    assert func.trace == 1.0

  # Testing Subtraction with reverse order
  def test_rsub_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 - x
    assert func.value == -3.0

  def test_rsub_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 - x
    assert func.trace == -1.0

  # Multiplication 
  def test_mul_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x * 2.0
    assert func.value == 10.0
  
  def test_mul_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = x * 2.0
    assert func.trace == 2.0

  # Reverse Multiplication
  def test_rmul_val(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 * x
    assert func.value == 10.0
  
  def test_rmul_trace(self):
    value = 5.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 * x
    assert func.trace == 2.0

  # Division
  def test_truediv_val(self):
    value = 6.0
    x = ForwardNode(value, trace = 1.0)
    func = x / 2.0
    assert func.value == 3.0
  
  def test_truediv_trace(self):
    value = 6.0
    x = ForwardNode(value, trace = 1.0)
    func = x / 2.0
    assert func.trace == 0.5

  # Reverse Division
  def test_rtruediv_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 / x
    assert func.value == 1.0

  def test_rtruediv_trace(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0 / x
    assert func.trace == 2.0

  # Power
  def test_pow_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = x**2.0
    assert func.value == 4.0 ## FINISH THIS ONE

  # Constant
  def test_constant_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0
    assert func.value == 2.0

  def test_constant_trace(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2.0
    assert func.trace == 0

  # Sine Function
  def test_sin_val(self):
    value = 3.14
    x = ForwardNode(value, trace = 1.0)
    func = 2*np.sin(x)
    assert func.value == 0

  def test_sin_trace(self):
    value = 3.14
    x = ForwardNode(value, trace = 1.0)
    func = 2*np.sin(x)
    assert func.trace == -2.0
  
  # Cosine Function
  def test_cos_val(self):
    value = 3.14
    x = ForwardNode(value, trace = 1.0)
    func = 2*np.cos(x)
    assert func.value == -2.0  

  def test_cos_trace(self):
    value = 3.14
    x = ForwardNode(value, trace = 1.0)
    func = 2*np.cos(x)
    assert func.trace == 0  
  
  # Logarithmic Function
  def test_log_val(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2*log(x)
    assert func.value == 1.3862943611198906
  
  def test_log_trace(self):
    value = 2.0
    x = ForwardNode(value, trace = 1.0)
    func = 2*log(x)
    assert func.trace == 1.0

  # Exponential Function
  def test_exp_val(self):
    value = np.log(2.0)
    x = ForwardNode(value, trace = 1.0)
    func = 2 * exp(ForwardNode(x))
    assert func.value == 8.0

  def test_exp_trace(self):
    value = np.log(2.0)
    x = ForwardNode(value, trace = 1.0)
    func = 2 * exp(ForwardNode(x))
    assert func.trace == 8.0

  # Square root 
  def test_sqrt_val(self):
    value = 4
    x = ForwardNode(value, trace = 1.0)
    func = 2 * np.sqrt(ForwardNode(x))
    assert func.value == 4.0

  def test_sqrt_trace(self):
    value = 4
    x = ForwardNode(value, trace = 1.0)
    func = 2 * np.sqrt(ForwardNode(x))
    assert func.trace == 0.5

  # Tan Function
  def test_tan_val(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2 * np.tan(ForwardNode(x))
    assert func.value == 0.0

  def test_tan_trace(self):
    value = np.pi
    x = ForwardNode(value, trace = 1.0)
    func = 2 * np.tan(ForwardNode(x))
    assert func.trace == 2

  # Arctan Function
  def test_arctan_val(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = np.arctan(x)
    assert func.value == 0

  def test_arctan_trace(self):
    value = 0
    x = ForwardNode(value, trace = 1.0)
    func = np.arctan(x)
    assert func.trace == 1

  #

  

