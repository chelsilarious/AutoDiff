import numpy as np


class ForwardNode():
    def __init__(self, value, trace=1.0, var='x1'):
        '''
        Constructor
        ===========
        Input:
        self - a ForwardNode variable
        value - int/flot, specifying the value of the current variable
        trace - int/float/np.array, derivative(s) of the current variable with respect to the input variable(s), default to be 1
        var - str, initialize the name of the ForwardNode variable, defaut as "x1"

        Output:
        a ForwardNode object, containing the value and trace of this variable

        Example:
        >>> x = ForwardNode(5, [0, 1], "x1, x2")
        ForwardNode Variable: ['x1, x2'],  Value: 5, Trace: [0 1]

        '''
        if isinstance(value, (int, float)):
            self.value = value
        else:
            raise TypeError("Invalid Input!")

        if isinstance(trace, (int, float)):
            self.trace = np.array([trace])
        elif isinstance(trace, list) and all([isinstance(num, (int, float)) for num in trace]):
            self.trace = np.array(trace)
        elif isinstance(trace, np.ndarray) and all([isinstance(num, (np.int64, np.float64)) for num in trace]):
            self.trace = trace
        else:
            raise TypeError("Invalid Input!")

        if isinstance(var, str):
            self.var = [var]
        elif isinstance(var, list) and all([isinstance(varname, str) for varname in var]):
            self.var = var
        else:
            raise TypeError("Invalid Input!")

    def __add__(self, other):
        '''
        Dunder method to add another ForwardNode variable, scalar and vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after addition

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = x + 3
        ForwardNode(6, 1, 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array([0,1]), var=['x1','x2'])
        >>> z = x1 + x2
        ForwardNode(7, [1,1], ['x1','x2'])

        '''
        if isinstance(other, (int, float)):
            # v = y + c; dv/dx1 = dy/dx1, dv/dx2 = dy/dx2, ...
            return ForwardNode(self.value + other, self.trace, self.var)
        elif isinstance(other, ForwardNode):
            # v = y + z; dv/dx1 = dy/dx1 + dz/dx1, dv/dx2 = dy/dx2 + dz/dx2, ...
            return ForwardNode(self.value + other.value, self.trace + other.trace, self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __radd__(self, other):
        '''
        Dunder method to add another ForwardNode variable, scalar and vector from the left

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after addition

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = 3 + x
        ForwardNode(6, 1 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x2 + x1
        ForwardNode(7, [1, 1], ['x1', 'x2'])

        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        Dunder method to subtract another ForwardNode variable, scalar and vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after subtraction

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = x - 2
        ForwardNode(1, 1 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x1 - x2
        ForwardNode(-1, [1, -1], ['x1', 'x2'])

        '''
        if isinstance(other, (int, float)):
            # v = y - c; dv/dx1 = dy/dx1, dv/dx2 = dy/dx2, ...
            return ForwardNode(self.value - other, self.trace, self.var)
        elif isinstance(other, ForwardNode):
            # v = y - z; dv/dx1 = dy/dx1 - dz/dx1, dv/dx2 = dy/dx2 - dz/dx2, ...
            return ForwardNode(self.value - other.value, self.trace - other.trace, self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __rsub__(self, other):
        '''
        Dunder method to subtract another ForwardNode variable, scalar and vector from the left

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after subtraction

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = 4 - x
        ForwardNode(1, 1 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x2 - x1
        ForwardNode(1, [-1, 1], ['x1', 'x2'])

        '''
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        '''
        Dunder method to multiply another ForwardNode variable, scalar and vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after multiplication

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = x * 2
        ForwardNode(6, 2 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x1 * x2
        ForwardNode(12, [4, 3], ['x1', 'x2'])

        '''
        if isinstance(other, (int, float)):
            # v = y * c; dv/dx1 = dy/dx1 * c, dv/dx2 = dy/dx2 * c, ...
            return ForwardNode(self.value * other, self.trace * other, self.var)
        elif isinstance(other, ForwardNode):
            # v = y * z; dv/dx1 = dy/dx1 * z + y * dz/dx1, dv/dx2 = dy/dx2 * z + y * dz/dx2, ...
            return ForwardNode(self.value * other.value, self.trace * other.value + self.value * other.trace, self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __rmul__(self, other):
        '''
        Dunder method to multiply another ForwardNode variable, scalar and vector from the left

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after multiplication

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = 2 * x
        ForwardNode(6, 2 'x')

        >>> x1 = ForwardNode(3, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x2 * x1
        ForwardNode(12, [4, 3], ['x1', 'x2'])

        '''
        return self.__mul__(other)

    def __truediv__(self, other):
        '''
        Dunder method to divide another ForwardNode variable, scalar and vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after division

        Examples:
        >>> x = ForwardNode(4, trace=1, var=['x'])
        >>> y = x / 2
        ForwardNode(2, 0.5 'x')

        >>> x1 = ForwardNode(12, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x1 / x2
        ForwardNode(3, [0.25, -0.75], ['x1', 'x2'])

        '''
        if isinstance(other, (int, float)):
            # v = y / c; dv/dx1 = dy/dx1 / c, dv/dx2 = dy/dx2 / c, ...
            return ForwardNode(self.value / other, self.trace / other, self.var)
        elif isinstance(other, ForwardNode):
            # v = y / z; dv/dx1 = (z * dy/dx1 - y * dz/dx1) / (z**2), dv/dx2 = (z * dy/dx2 - y * dz/dx2) / (z**2), ...
            return ForwardNode(self.value / other.value,
                               (other.value * self.trace - self.value * other.trace) / (other.value ** 2), self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __rtruediv__(self, other):
        '''
        Dunder method to divide another ForwardNode variable, scalar and vector from the left

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after division

        Examples:
        >>> x = ForwardNode(4, trace=1, var=['x'])
        >>> y = 8 / x
        ForwardNode(2, -0.5 'x')

        >>> x1 = ForwardNode(2, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x2 / x1
        ForwardNode(2, [-1, 0.5], ['x1', 'x2'])

        '''
        if isinstance(self, ForwardNode):
            if not isinstance(other, (int,float)):
                raise AttributeError("Invalid Input!")
            return ForwardNode(other / self.value, self.trace * (-1 * other) / (self.value ** 2), self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __pow__(self, other):
        '''
        Dunder method to compute the power of a ForwardNode variable subject to another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after taking the power

        Examples:
        >>> x = ForwardNode(4, trace=1, var=['x'])
        >>> y = x ** 2
        ForwardNode(16, 8, 'x')

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(2, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x1 ** x2
        ForwardNode(16, [8, 22.18070978], ['x1', 'x2'])

        '''
        if isinstance(other, (int, float)):
            if (self.value < 0) and abs(other) < 1:
                raise ValueError("Derivatives of variables with negative values to a power between -1 and 1 are not supported!")
            # v = y ** c; dv/dx1 = c * (y ** (c-1)) * dy/dx1, dv/dx2 = c * (y ** (c-1)) * dy/dx2, ...
            new_trace = other * (self.value ** (other - 1)) * self.trace
            return ForwardNode(self.value ** other, new_trace, self.var)
        elif isinstance(other, ForwardNode):
            # v = y ** z; dv/dx1 = z * (y ** (z-1)) * dy/dx1 + (y ** z) * log(y) * dz/dx1, ...
            new_trace = other.value * (self.value ** (other.value - 1)) * self.trace + (
                        self.value ** other.value) * np.log(self.value) * other.trace
            return ForwardNode(self.value ** other.value, new_trace, self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __rpow__(self, other):
        '''
        Dunder method to compute the power of a ForwardNode variable subject to another ForwardNode variable, scalar or vector from the left

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        a ForwardNode object, containing new value and trace after taking the power

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> y = 2 ** x
        ForwardNode(8, 36, 'x')

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(2, trace=np.array(([0,1])), var=['x1','x2'])
        >>> z = x2 ** x1
        ForwardNode(16, [11.09035489, 32], ['x1', 'x2'])

        '''
        if isinstance(self, ForwardNode):
            if not isinstance(other, (int,float)):
                raise AttributeError("Invalid Input!")
            if (self.value < 0) and abs(other) < 1:
                raise ValueError("Derivatives of negative values to a power variable between -1 and 1 are not supported!")
            new_trace = (other ** self.value) * np.log(other) * self.trace
            return ForwardNode(other ** self.value, new_trace, self.var)
        else:
            raise AttributeError("Invalid Input!")

    def __neg__(self):
        '''
        Dunder method to take the negation of a ForwardNode variable

        Input:
        self - a ForwardNode variable

        Output:
        The negation of the input ForwardNode variable

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> -x
        ForwardNode(-3, trace=-1, var=['x'])
        '''
        return ForwardNode(-1 * self.value, -1 * self.trace, self.var)

    def __lt__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is less than another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value < other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x < 2
        False

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(8, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 < x2
        True

        '''
        if isinstance(other, (int, float)):
            return self.value < other
        elif isinstance(other, ForwardNode):
            return self.value < other.value
        else:
            raise AttributeError("Invalid Input!")

    def __gt__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is greater than another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value > other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x > 2
        True

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(8, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 > x2
        False

        '''
        if isinstance(other, (int, float)):
            return self.value > other
        elif isinstance(other, ForwardNode):
            return other.__lt__(self)
        else:
            raise AttributeError("Invalid Input!")

    def __le__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is less than or equal to another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value <= other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x <= 3
        True

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(8, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 <= x2
        False

        '''
        # if isinstance(self, (int,float)):
        #    if isinstance(other, (int,float)):
        #        return self <= other
        #    elif isinstance(other, ForwardNode):
        #        return self <= other.value
        if isinstance(self, ForwardNode):
            if isinstance(other, (int, float)):
                return self.value <= other
            elif isinstance(other, ForwardNode):
                return self.value <= other.value
        elif isinstance(other, ForwardNode):
            if isinstance(self, (int, float)):
                return self <= other.value
        raise AttributeError("Invalid Input!")

    def __ge__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is greater than or equal to another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value >= other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x >= 3
        True

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(8, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 >= x2
        False

        '''
        return other.__le__(self)

    def __eq__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is equal to another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value == other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x == 2
        False

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 == x2
        True

        '''
        if isinstance(self, (int, float)):
            if isinstance(other, (int, float)):
                return self == other
            elif isinstance(other, ForwardNode):
                return self == other.value
        elif isinstance(self, ForwardNode):
            if isinstance(other, (int, float)):
                return self.value == other
            elif isinstance(other, ForwardNode):
                return self.value == other.value
        raise AttributeError("Invalid Input!")

    def __neq__(self, other):
        '''
        Dunder method to compare if the value of a ForwardNode variable is not equal to another ForwardNode variable, scalar or vector

        Input:
        self - a ForwardNode variable
        other - a constant of integers or decimals / a ForwardNode object representing a variable

        Output:
        True if self value != other value, False otherwise

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> x != 2
        True

        >>> x1 = ForwardNode(4, trace=np.array([1,0]), var=['x1','x2'])
        >>> x2 = ForwardNode(4, trace=np.array(([0,1])), var=['x1','x2'])
        >>> x1 != x2
        False

        '''
        return not self.__eq__(other)

    def __repr__(self):
        '''
        Dunder method to represent a ForwardNode objects as a string

        Input:
        self - a ForwardNode variable

        Output:
        The value and trace of the ForwardNode object represented as a string

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> repr(x)
        ForwardNode Variable: ['x'],  Value: 3, Trace: [1]

        '''
        return f'ForwardNode Variable: {self.var},  Value: {self.value}, Trace: {self.trace}'

    def __str__(self):
        '''
        Dunder method to represent a ForwardNode objects as a string

        Input:
        self - a ForwardNode variable

        Output:
        The value and trace of the ForwardNode object represented as a string

        Examples:
        >>> x = ForwardNode(3, trace=1, var=['x'])
        >>> print(x)
        ForwardNode Variable: ['x'],  Value: 3, Trace: [1]

        '''
        return f'ForwardNode Variable: {self.var},  Value: {self.value}, Trace: {self.trace}'
