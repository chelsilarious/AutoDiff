import numpy as np
import re
from forwardNode import ForwardNode
from reverseNode import ReverseNode
from utils import *


def derive(func_expr, varname, value, mode='reverse'):
    if mode == 'forward':
        x = ForwardNode(value)
        y = eval(re.sub('(\d+\.*\d*)', f'constant(\\1, mode=\'{mode}\')', func_expr))
        val, grad = y.value, y.trace
    elif mode == 'reverse':
        x = ReverseNode(value)
        y = eval(re.sub('(\d+\.*\d*)', f'constant(\\1, mode=\'{mode}\')', func_expr))
        val, grad = y.value, x.gradient()

    return val, grad


def main():
    x = 3 * cos(ForwardNode(np.pi)) + 4  # 3cos(x)+4, x=pi
    print(x)


if __name__ == "__main__":
    main()