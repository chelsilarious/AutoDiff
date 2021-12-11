import numpy as np
from AutoDiff.forwardNode import ForwardNode
from AutoDiff.reverseNode import ReverseNode

__all__ = ['sin', 'cos', 'log', 'exp', 'sqrt', 'tan', 'cot', 'sec', 'csc',
           'arctan', 'arcsin', 'arccos', 'tanh', 'sinh', 'cosh', 'log_base']


def exp(node):
    '''
    Compute the exponent of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after taking exponent

    Examples:
    >>> x = ForwardNode(3, trace=1, var=['x'])
    >>> exp(x)
    ForwardNode Variable: ['x'],  Value: 20.085536923187668, Trace: [20.08553692]

    '''
    if isinstance(node, (int, float)):
        return np.exp(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.exp(node.value), node.trace * np.exp(node.value), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.exp(node.value))
        node.children.append((np.exp(node.value), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def log(node):
    '''
    Compute the log of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after taking log

    Examples:
    >>> x = ForwardNode(3, trace=1, var=['x'])
    >>> log(x)
    ForwardNode Variable: ['x'],  Value: 1.0986122886681098, Trace: [0.33333333]

    '''
    if isinstance(node, (int, float)):
        if node <= 0:
            raise ValueError("Invalid inpput: cannot take log for value <= 0")
        return np.log(node)
    elif isinstance(node, ForwardNode):
        if node.value <= 0:
            raise ValueError("Invalid inpput: cannot take log for value <= 0")
        return ForwardNode(np.log(node.value), node.trace / node.value, node.var)
    elif isinstance(node, ReverseNode):
        if node.value <= 0:
            raise ValueError("Invalid inpput: cannot take log for value <= 0")
        new = ReverseNode(np.log(node.value))
        node.children.append((1/node.value, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def sqrt(node):
    '''
    Compute the square root of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after taking the square root

    Examples:
    >>> x = ForwardNode(4, trace=1, var=['x'])
    >>> sqrt(x)
    ForwardNode Variable: ['x'],  Value: 2.0, Trace: [0.25]

    '''
    if isinstance(node, (int, float)):
        if node < 0:
            raise ValueError(f"Invalid Value: cannot calculate square root of {node}.")
        else:
            return np.sqrt(node)
    elif isinstance(node, ForwardNode):
        if node.value < 0:
            raise ValueError(f"Invalid Value: cannot calculate square root of {node.value}.")
        else:
            return ForwardNode(node.value ** 0.5, node.trace * 0.5 * node.value ** (-0.5), node.var)
    elif isinstance(node, ReverseNode):
        if node.value < 0:
            raise ValueError(f"Invalid Value: cannot calculate square root of {node.value}.")
        else:
            new = ReverseNode(node.value ** 0.5)
            node.children.append((0.5 * node.value ** (-0.5), new))
            return new
    else:
        raise AttributeError("Invalid Input!")


def sin(node):
    '''
    Compute the sine of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after sin operation

    Examples:
    >>> x = ForwardNode(np.pi, trace=1, var=['x'])
    >>> sin(x)
    ForwardNode Variable: ['x'],  Value: 1.2246467991473532e-16, Trace: [-1.]

    '''
    if isinstance(node, (int, float)):
        return np.sin(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.sin(node.value), node.trace * np.cos(node.value), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.sin(node.value))
        node.children.append((np.cos(node.value), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def cos(node):
    '''
    Compute the cosine of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after cos operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> cos(x)
    ForwardNode Variable: ['x'],  Value: 1.0, Trace: [-0.]

    '''
    if isinstance(node, (int, float)):
        return np.cos(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.cos(node.value), -1.0 * node.trace * np.sin(node.value), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.cos(node.value))
        node.children.append((-np.sin(node.value), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def tan(node):
    '''
    Compute the tangent of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after tan operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> tan(x)
    ForwardNode Variable: ['x'],  Value: 0.0, Trace: [1.]

    '''
    if isinstance(node, (int, float)):
        if node % (np.pi / 2) == 0 and node % np.pi != 0:
            raise ValueError(f"Invalid input: derivative for tangent of {node} doesn't exist")
        return np.tan(node)
    elif isinstance(node, ForwardNode):
        if node.value % (np.pi / 2) == 0 and node.value % np.pi != 0:
            raise ValueError(f"Invalid input: derivative for tangent of {node.value} doesn't exist")
        return ForwardNode(np.tan(node.value), node.trace / np.cos(node.value) ** 2, node.var)
    elif isinstance(node, ReverseNode):
        if node.value % (np.pi / 2) == 0 and node.value % np.pi != 0:
            raise ValueError(f"Invalid input: derivative for tangent of {node.value} doesn't exist")
        new = ReverseNode(np.tan(node.value))
        node.children.append((1.0 / np.cos(node.value) ** 2, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def cot(node):
    '''
    Compute the cotangent of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after cot operation

    Examples:
    >>> x = ForwardNode(np.pi / 2, trace=1, var=['x'])
    >>> cot(x)
    ForwardNode Variable: ['x'],  Value: 6.123233995736766e-17, Trace: [-1.]

    '''
    if isinstance(node, (int, float)):
        if node % np.pi == 0:
            raise ValueError(f"Invalid Value: cotangent of {node} does not exist.")
        return 1 / np.tan(node)
    elif isinstance(node, ForwardNode):
        if node.value % np.pi == 0:
            raise ValueError(f"Invalid Value: cotangent of {node.value} does not exist.")
        return ForwardNode(1 / np.tan(node.value), node.trace * (-1.0) / np.sin(node.value) ** 2, node.var)
    elif isinstance(node, ReverseNode):
        if node.value % np.pi == 0:
            raise ValueError(f"Invalid Value: cotangent of {node.value} does not exist.")
        new = ReverseNode(1 / np.tan(node.value))
        node.children.append((-1.0 / np.sin(node.value) ** 2, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def sec(node):
    '''
    Compute the secant of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after sec operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> sec(x)
    ForwardNode Variable: ['x'],  Value: 1.0, Trace: [0.]

    '''
    if isinstance(node, (int, float)):
        if node % (np.pi / 2) == 0 and node % np.pi != 0:
            raise ValueError(f"Invalid Value: secant of {node} does not exist.")
        return 1 / np.cos(node)
    elif isinstance(node, ForwardNode):
        if node.value % (np.pi / 2) == 0 and node.value % np.pi != 0:
            raise ValueError(f"Invalid Value: secant of {node.value} does not exist.")
        return ForwardNode(1 / np.cos(node.value), node.trace * np.sin(node.value) / np.cos(node.value) ** 2, node.var)
    elif isinstance(node, ReverseNode):
        if node.value % (np.pi / 2) == 0 and node.value % np.pi != 0:
            raise ValueError(f"Invalid Value: secant of {node.value} does not exist.")
        new = ReverseNode(1 / np.cos(node.value))
        node.children.append((np.sin(node.value) / np.cos(node.value) ** 2, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def csc(node):
    '''
    Compute the cosecant of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after csc operation

    Examples:
    >>> x = ForwardNode(np.pi / 2, trace=1, var=['x'])
    >>> csc(x)
    ForwardNode Variable: ['x'],  Value: 1.0, Trace: [-6.123234e-17]

    '''
    if isinstance(node, (int, float)):
        if node % np.pi == 0:
            raise ValueError(f"Invalid Value: cosecant of {node} does not exist.")
        return 1 / np.sin(node)
    elif isinstance(node, ForwardNode):
        if node.value % np.pi == 0:
            raise ValueError(f"Invalid Value: cosecant of {node.value} does not exist.")
        return ForwardNode(1 / np.sin(node.value), node.trace * (-1.0) * np.cos(node.value) / np.sin(node.value) ** 2,
                           node.var)
    elif isinstance(node, ReverseNode):
        if node.value % np.pi == 0:
            raise ValueError(f"Invalid Value: cosecant of {node.value} does not exist.")
        new = ReverseNode(1 / np.sin(node.value))
        node.children.append((-1.0 * np.cos(node.value) / np.sin(node.value) ** 2, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def arcsin(node):
    '''
    Compute the arcsine of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after arcsin operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> arcsin(x)
    ForwardNode Variable: ['x'],  Value: 0.0, Trace: [1.]

    '''
    if isinstance(node, (int, float)):
        if np.abs(node) > 1:
            raise ValueError(f"Invalid Value: arcsin of {node} does not exist.")
        return np.arcsin(node)
    elif isinstance(node, ForwardNode):
        if np.abs(node.value) >= 1:
            raise ValueError(f"Invalid Value: derivative of arcsin of {node.value} does not exist.")
        return ForwardNode(np.arcsin(node.value), node.trace / np.sqrt(1 - node.value ** 2), node.var)
    elif isinstance(node, ReverseNode):
        if np.abs(node.value) >= 1:
            raise ValueError(f"Invalid Value: derivative of arcsin of {node.value} does not exist.")
        new = ReverseNode(np.arcsin(node.value))
        node.children.append((1.0 / np.sqrt(1 - node.value ** 2), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def arccos(node):
    '''
    Compute the arccosine of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after arccos operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> arccos(x)
    ForwardNode Variable: ['x'],  Value: 1.5707963267948966, Trace: [-1.]

    '''
    if isinstance(node, (int, float)):
        if np.abs(node) > 1:
            raise ValueError(f"Invalid Value: arccos of {node} does not exist.")
        return np.arccos(node)
    elif isinstance(node, ForwardNode):
        if np.abs(node.value) > 1:
            raise ValueError(f"Invalid Value: derivative of arccos of {node.value} does not exist.")
        return ForwardNode(np.arccos(node.value), node.trace * (-1.0) / np.sqrt(1 - node.value ** 2), node.var)
    elif isinstance(node, ReverseNode):
        if np.abs(node.value) > 1:
            raise ValueError(f"Invalid Value: derivative of arccos of {node.value} does not exist.")
        new = ReverseNode(np.arccos(node.value))
        node.children.append((-1.0 / np.sqrt(1 - node.value ** 2), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def arctan(node):
    '''
    Compute the arctangent of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after arctan operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> arctan(x)
    ForwardNode Variable: ['x'],  Value: 0.0, Trace: [1.]

    '''
    if isinstance(node, (int, float)):
        return np.arctan(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.arctan(node.value), node.trace / (1 + node.value ** 2), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.arctan(node.value))
        node.children.append((1.0 / (1 + node.value ** 2), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def sinh(node):
    '''
    Compute the sinh of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after sinh operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> sinh(x)
    ForwardNode Variable: ['x'],  Value: 0.0, Trace: [1.]

    '''
    if isinstance(node, (int, float)):
        return np.sinh(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.sinh(node.value), node.trace * np.cosh(node.value), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.sinh(node.value))
        node.children.append((np.cosh(node.value), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def cosh(node):
    '''
    Compute the cosh of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after cosh operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> cosh(x)
    ForwardNode Variable: ['x'],  Value: 1.0, Trace: [0.]

    '''
    if isinstance(node, (int, float)):
        return np.cosh(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.cosh(node.value), node.trace * np.sinh(node.value), node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.cosh(node.value))
        node.children.append((np.sinh(node.value), new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def tanh(node):
    '''
    Compute the tanh of the ForwardNode object

    Input:
    self - a ForwardNode variable

    Output:
    The value and trace of the ForwardNode object after tanh operation

    Examples:
    >>> x = ForwardNode(0, trace=1, var=['x'])
    >>> tanh(x)
    ForwardNode Variable: ['x'],  Value: 0.0, Trace: [1.]

    '''
    if isinstance(node, (int, float)):
        return np.tanh(node)
    elif isinstance(node, ForwardNode):
        return ForwardNode(np.tanh(node.value), node.trace / np.cosh(node.value) ** 2, node.var)
    elif isinstance(node, ReverseNode):
        new = ReverseNode(np.tanh(node.value))
        node.children.append((1 / np.cosh(node.value) ** 2, new))
        return new
    else:
        raise AttributeError("Invalid Input!")


def log_base(node, base=10):
    '''
    Compute the log of the ForwardNode object with base specified in the base parameter

    Input:
    self - a ForwardNode variable
    base - a number indicating the base

    Output:
    The value and trace of the ForwardNode object after the logarithm operation with a specific base

    Examples:
    >>> x = ForwardNode(np.exp(3), trace=1, var=['x'])
    >>> log_base(x, base = np.e)
    ForwardNode Variable: ['x1'],  Value: 3.0, Trace: [0.04978707]

    '''
    if (not isinstance(base, (int, float))) or base <= 0:
        raise ValueError("Invalid input: base must be a positive number")
    if isinstance(node, (int, float)):
        if node < 0:
            raise ValueError(f"Invalid input: base-{base} log of {node} does not exist.")
        return np.log(node) / np.log(base)
    elif isinstance(node, ForwardNode):
        if node.value < 0:
            raise ValueError(f"Invalid input: base-{base} log of {node.value} does not exist.")
        return ForwardNode(np.log(node.value) / np.log(base), node.trace / (node.value * np.log(base)), node.var)
    elif isinstance(node, ReverseNode):
        if node.value < 0:
            raise ValueError(f"Invalid input: base-{base} log of {node.value} does not exist.")
        new = ReverseNode(np.log(node.value) / np.log(base))
        node.children.append((1.0 / (node.value * np.log(base)), new))
        return new
    else:
        raise AttributeError("Invalid Input!")
