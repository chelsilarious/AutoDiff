import numpy as np
from src.forwardNode import ForwardNode
from src.reverseNode import ReverseNode


def constant(val, mode='forward'):
    if mode == 'forward':
        new = ForwardNode(val, 1)
    elif mode == 'reverse':
        new = ReverseNode(val)
        new.adjoint = 0
    return new


def sin(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.sin(node.value), node.trace * np.cos(node.value))
        # new.depends.append((np.cos(node.value), node))  # sin(x) -> d/dx = cos(x)
    elif type(node) is ReverseNode:
        new = ReverseNode(np.sin(node.value))
        node.children.append((np.cos(node.value), new))  # sin(x) -> d/dx = cos(x)
    return new


def cos(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.cos(node.value), -1 * node.trace * np.sin(node.value))
        # new.depends.append((-np.sin(node.value), node))  # cos(x) -> d/dx = -sin(x)
    elif type(node) is ReverseNode:
        new = ReverseNode(np.cos(node.value))
        node.children.append((-np.sin(node.value), new))  # cos(x) -> d/dx = -sin(x)
    return new


def log(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.log(node.value), node.trace / node.value)
        # new.depends.append((1.0/node.value, node))  # log(x) -> d/dx = 1/x
    elif type(node) is ReverseNode:
        new = ReverseNode(np.log(node.value))
        node.children.append((1.0 / node.value, new))  # log(x) -> d/dx = 1/x
    return new


def exp(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.exp(node.value), node.trace * np.exp(node.value))
        # new.depends.append((np.exp(node.value), node))  # e^x -> d/dx = e^x
    elif type(node) is ReverseNode:
        new = ReverseNode(np.exp(node.value))
        node.children.append((np.exp(node.value), new))  # e^x -> d/dx = e^x
    return new


def sqrt(node):
    if node.value < 0:
        raise ValueError(f"Invalid value: cannot calculate the square root for {node.value}.")
    elif type(node) is ForwardNode:
        new = ForwardNode(node.value ** 0.5, node.trace * node.value ** -0.5)
        # new.depends.append((node.value ** -0.5, node))  # sqrt(x) -> d/dx = x ^ -1/2
    elif type(node) is ReverseNode:
        new = ReverseNode(node.value ** 0.5)
        node.children.append((node.value ** -0.5, new))  # sqrt(x) -> d/dx = x ^ -1/2
    return new


def tan(node):
    if node.value % (np.pi / 2) == 0 and node.value % np.pi > 0:
        raise ValueError(f"Invalid value: the tangent for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(np.tan(node.value), node.trace / np.cos(node.value) ** 2)
        # new.depends.append((1 / np.cos(node.value) ** 2, node))  # tan(x) -> d/dx = 1 / cos(x) ^ 2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.tan(node.value))
        node.children.append((1 / np.cos(node.value) ** 2, new))  # tan(x) -> d/dx = 1 / cos(x) ^ 2
    return new


def arctan(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.arctan(node.value), node.trace / (1 + node.value ** 2))
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.arctan(node.value))
        node.children.append((1 / (1 + node.value ** 2), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def arcsin(node):
    if np.abs(node.value) > 1:
        raise ValueError(f"Invalid value: arcsin for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(np.arcsin(node.value), node.trace * (1 / np.sqrt(1 - node.value ** 2)))
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.arcsin(node.value))
        node.children.append((1 / np.sqrt(1 - node.value ** 2)), new)  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def arccos(node):
    if np.abs(node.value) > 1:
        raise ValueError(f"Invalid value: arccos for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(np.arccos(node.value), -1 * node.trace * (1 / np.sqrt(1 - node.value ** 2)))
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.arccos(node.value))
        node.children.append((1 / np.sqrt(1 - node.value ** 2), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def tanh(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.tanh(node.value), (1 / np.cosh(node.value)) ** 2 * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.tanh(node.value))
        node.children.append(((1 / np.cosh(node.value)) ** 2, new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def sinh(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.sinh(node.value), np.cosh(node.value) * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.sinh(node.value))
        node.children.append((np.cosh(node.value), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def cosh(node):
    if type(node) is ForwardNode:
        new = ForwardNode(np.cosh(node.value), np.sinh(node.value) * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.cosh(node.value))
        node.children.append((np.sinh(node.value), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def log_base(node, base=10):
    if node.value < 0:
        raise ValueError(f"Invalid Value: the log operation for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(np.log(node.value) / np.log(base), 1 / (node.value * np.log(base)) * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.log(node.value))
        node.children.append((1 / (node.value * np.log(base)), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def log_base(node, base=10):
    if node.value < 0:
        raise ValueError(f"Invalid Value: the log operation for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(np.log(node.value) / np.log(base), 1 / (node.value * np.log(base)) * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(np.log(node.value))
        node.children.append((1 / (node.value * np.log(base)), new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def cot(node):
    if node.value % np.pi == 0:
        raise ValueError(f"Invalid Value: the cotangent operation for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = ForwardNode(1 / np.tan(node.value), (1 / np.sin(node.value)) ** 2 * node.trace)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(1 / np.tan(node.value))
        node.children.append((1 / np.sin(node.value) ** 2, new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new


def sec(node):
    if node.value % (np.pi / 2) == 0 and node.value % np.pi > 0:
        raise ValueError(f"Invalid Value: the secant operation for {node.value} doesn't exist.")
    elif type(node) is ForwardNode:
        new = 1 / cos(node)
        # new.depends.append((1 / (1 + node.value ** 2), node)) # arctan(x) -> d/dx = 1 / 1 + x^2
    elif type(node) is ReverseNode:
        new = ReverseNode(1 / np.tan(node.value))
        node.children.append((1 / np.sin(node.value) ** 2, new))  # arctan(x) -> d/dx = 1 / 1 + x^2
    return new
