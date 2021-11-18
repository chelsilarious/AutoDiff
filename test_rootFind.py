import unittest
from src.forwardNode import ForwardNode

def root_finding_algo(f, inp_dim, x_init, num_iter):
  # make a note for documentation of input types later
  # initial guess, x_init
  # input dimension, inp_dim
  # specify num_iter to be number of iterations
  # specify some function f
  x_res = x_init.copy()
  curr_iter = 0
  while curr_iter < num_iter:
    f_node = ForwardNode(f)
    grad_f_node = f_node.gradient()
    # x_n+1 = x_n - f(x_n) / f'(x_n)
    x_res = x_res - f_node(x_res) / grad_f_node(x_res)
    iter += 1
  return x_res