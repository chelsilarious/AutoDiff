import numpy as np
import inspect
from AutoDiff.forwardNode import ForwardNode
from AutoDiff.reverseNode import ReverseNode
from AutoDiff.utils import *


def init_trace(var, variables):
    '''
    Initialize the trace for ForwardNode objects given all variables in function

    Input:
    var - str, name of variable to initialize the trace
    variables - list, names for all variables in function

    Output:
    Initilized trace for ForwardNode object var

    Examples:
    >>> init_trace(var="x1", variables=["x1", "x2", "x3"])
    [1.0, 0.0, 0.0]
    '''
    trc = np.zeros(len(variables))
    trc[variables.index(var)] = 1
    return trc


def create_node(var, value, variables):
    '''
    Create a new ForwardNode object ForwardNode objects

    Input:
    var - str, name of the new ForwardNode variable
    value - int/float, value of the ForwardNode variable
    variables - list, names for all variables in function

    Output:
    A new ForwardNode variable

    Examples:
    >>> create_node(var="x1", value=5.0, variables=["x1", "x2", "x3"])
    ForwardNode Variable: ['x1', 'x2', 'x3'],  Value: 5.0, Trace: [1. 0. 0.]
    '''
    return ForwardNode(value, init_trace(var,variables), variables)


def gradientF(y, variables, target=None):
    '''
    Calculate the graident using forward mode methods

    Input:
    y - ForwardNode, the functions output we a{re caculating
    variables - list, names for all variables in function
    target - list, name of our target variable(s) to calculate the gradient

    Output:
    The derivative / partial derivative / gradient / jacobian of input function and target variable

    Examples:
    >>> variables = ['x1']
    >>> x1 = create_node(var='x1', value=2, variables=variables)
    >>> y = 3 * x1 + x1 ** 2 - exp(x1)
    >>> der = gradientF(y, variables, target='x1')
    >>> der
    -0.3890560989306504

    '''
    if isinstance(y, ForwardNode):
        if target:
            if target not in variables:
                raise AttributeError("This is not a variable used! No gradient available!")
            # derivative dy/dxi or partial derivatives dely/delxi
            return y.trace[variables.index(target)]
        else:
            # gradient [dely/delx1, ..., dely/delxn]
            return [y.trace[variables.index(var)] for var in variables]
    elif isinstance(y, (np.ndarray, list)) and all([isinstance(yi, ForwardNode) for yi in y]):
        if target:
            if target not in variables:
                raise AttributeError("This is not a variable used! No gradient available!")
            # derivatives [dy1/dxi, ..., dym/dxi]
            return [yi.trace[variables.index(target)] for yi in y]
        else:
            # jacobian [[dely1/delx1, ..., dely1/delxn], ..., [delym/delx1, ..., delym/delxn]]
            jcb = [[yi.trace[variables.index(var)] for yi in y] for var in variables]
            return np.array(jcb).T
    else:
        raise TypeError("Invalid Input!")


def gradientR(functions, var_dict, target=None):
    '''
    Calculate the graident using reverse mode methods

    Input:
    functions - str, the functions output we a{re caculating
    var_dict - dictionary, name and value pair of all variables in function
    target - list, name of our target variable(s) to calculate the gradient

    Output:
    The derivative / partial derivative / gradient / jacobian of input function and target variable

    Examples:
    >>> var_dict = {"x1": 2}
    >>> functions = "3 * x1 + x1 ** 2 - exp(x1)"
    >>> res = gradientR(functions, var_dict, target=["x1"])
    >>> res
    [[-0.3890560989306504]]

    '''
    res = []
    variables = list(var_dict.keys())
    if not target:
        target = variables
    nodes = []
    functions = [functions] if isinstance(functions, str) else functions

    for var in variables:
        value = var_dict[var]
        exec(f'{var} = ReverseNode(value=value)')
        exec(f'nodes.append((var, {var}))')

    for f in functions:
        grads = []
        for name, node in nodes:
            node.gradient_reset()
        y = eval(f)
        for name, node in nodes:
            if name in target:
                g = node.gradient()
                grads.append(g)
        res.append(grads)

    if isinstance(target, str):
        res = [item for sublist in res for item in sublist]
    return np.array(res)


def forward_auto_diff(functions, var_dict, target=None):
    '''
    Perform forward mode automatic differentiation

    Input:
    functions - str, the functions output we a{re caculating
    var_dict - dictionary, name and value pair of all variables in function
    target - list, name of our target variable(s) to calculate the gradient

    Output:
    The derivative / partial derivative / gradient / jacobian of input function and target variable

    Examples:
    >>> functions = ["x1 + sin(x2) * 5", "exp(x1) - log(2 * x2)"]
    >>> var_dict = {"x1": 5, "x2": 2}
    >>> target = ["x1"]
    >>> res = forward_auto_diff(functions, var_dict, target)
    >>> res
    [[1.0, 148.4131591025766]]

    '''
    variables = list(var_dict.keys())
    if not target:
        target = variables
    funcs = []
    res = []
    functions = [functions] if isinstance(functions, str) else functions

    for var in variables:
        value = var_dict[var]
        exec(f'{var} = create_node(var=var, value=value, variables=variables)')

    for f in functions:
        y = eval(f)
        funcs.append(y)

    if isinstance(target, list) and len(variables) == len(target):
        res = gradientF(funcs, variables)
        name = "Jacobian" if len(funcs) > 1 else "Derivative"
        print(f"Functions: {functions}\nVariables: {var_dict}\n------------------------------\n{name}:\n {res}")
    else:
        name = "Gradient" if len(funcs) > 1 else "Partial derivative"
        s = ""
        for t in target:
            der = gradientF(funcs, variables, target=t)
            res.append(der)
            s += f"{name} with respect to {t}: {der}\n"
        print(f"Functions: {functions}\nVariables: {var_dict}\n------------------------------\n" + s)

    return res


def reverse_auto_diff(functions, var_dict, target=None):
    '''
    Perform reverse mode automatic differentiation

    Input:
    functions - str, the functions output we a{re caculating
    var_dict - dictionary, name and value pair of all variables in function
    target - list, name of our target variable(s) to calculate the gradient

    Output:
    The derivative / partial derivative / gradient / jacobian of input function and target variable

    Examples:
    >>> functions = ["x1 + sin(x2) * 5", "exp(x1) - log(2 * x2)"]
    >>> var_dict = {"x1": 5, "x2": 2}
    >>> target = ["x1"]
    >>> res = reverse_auto_diff(functions, var_dict, target)
    >>> res
    [[1.0, 148.4131591025766]]

    '''

    variables = list(var_dict.keys())
    if not target:
        target = variables
    res = []
    functions = [functions] if isinstance(functions, str) else functions

    if isinstance(target, list) and len(variables) == len(target):
        res = gradientR(functions, var_dict, variables)
        name = "Jacobian" if len(functions) > 1 else "Derivative"
        print(f"Functions: {functions}\nVariables: {var_dict}\n------------------------------\n{name}:\n {res}")
    else:
        name = "Gradient" if len(functions) > 1 else "Partial Derivative"
        s = ""
        for t in target:
            der = gradientR(functions, var_dict, target=t)
            res.append(der)
            s += f"{name} with respect to {t}: {der}\n"
        print(f"Functions: {functions}\nVariables: {var_dict}\n------------------------------\n" + s)

    return res


def translate(lambda_func):
    '''
    Translate lambda function input to string representation of function

    Input:
    lambda_func - function, lambda function(s) that the user want to perform automatic differentiation

    Output:
    String representation of function input

    Examples:
    >>> functions = lambda x1, x2: x1 + sin(x2)
    >>> translate(lambda_func=functions)
    ['x1 + sin(x2)', 'cos(x1) + 5 / exp(x2)']

    >>> functions = lambda x1, x2: [x1 + sin(x2), cos(x1) + 5 / exp(x2)]
    >>> translate(lambda_func=functions)
    ['exp(x1) + log(x2) - 5']

    '''
    functions = inspect.getsourcelines(lambda_func)[0][0].strip('\n').split(": ")[-1].strip('][').split(", ")
    return functions


def auto_diff(functions, var_dict, target=None, mode="forward"):
    '''
    Wrap function for automatic differentiation

    Input:
    functions - str, the functions output we are caculating
    var_dict - dictionary, name and value pair of all variables in function
    target - list, name of our target variable(s) to calculate the gradient
    mode - str, either forward or reverse model

    Output:
    The derivative / partial derivative / gradient / jacobian of input function and target variable

    Examples:
    >>> functions = ["tanh(x1) + cosh(x2 * 3) - sec(x3)", "x1 / x2 * cos(x3)", "sin(x1 / 2) + x2 * x3"]
    >>> var_dict = {"x1": np.pi / 2, "x2": 1, "x3": 0}
    >>> gradient = auto_diff(functions, var_dict, ["x1", "x2"], mode="forward")
    >>> gradient
    [[0.15883159318006335, 1.0, 0.3535533905932738], [30.053624782229708, -1.5707963267948966, 0.0]]

    >>> jacobian = auto_diff(functions, var_dict, ["x1", "x2", "x3"], mode="reverse")
    >>> jacobian
    [[0.15883159318006335, 30.053624782229708, 0.0], [1.0, -1.5707963267948966, 0.0], [0.3535533905932738, 0.0, 1.0]]

    '''
    if (not isinstance(functions, (str, list))) and functions.__name__ == "<lambda>":
        functions = translate(functions)
    if not (isinstance(functions, str) or all([isinstance(f, str) for f in functions])):
        raise TypeError('Invalid input type: each function should be a string or lambda function')
    if not isinstance(var_dict, dict):
        raise TypeError('Invalid input type: input variables should be dictionary')
    if target:
        if not (isinstance(target, str) or all([t in var_dict.keys() for t in target])):
            raise ValueError('Invalid target value: target must be in the variable dictionary')

    if mode == "forward":
        return forward_auto_diff(functions, var_dict, target)
    elif mode == "reverse":
        return reverse_auto_diff(functions, var_dict, target)
    else:
        raise ValueError("Invalid mode: please choose between forward and reverse mode")


def main():
    functions = lambda x1, x2: [exp(x1) + log(x2) - 5, sin(x1) + cos(x2)]
    var_dict = {"x1": 3, "x2": 5}
    auto_diff(functions, var_dict, ["x1", "x2"], "reverse")


if __name__ == "__main__":
    main()

