import numpy as np


class ReverseNode():
    def __init__(self, value):
        '''
        Constructor
        ===========
        Input:
        self - a ReverseNode variable
        value - int/flot, specifying the value of the current variable

        Output:
        a ReverseNode object, containing the value and trace of this variable

        Example:
        '''
        if isinstance(value, (int, float)):
            self.value = value

        self.children = []
        self.adjoint = 1.0

    def gradient(self):
        if len(self.children) > 0:
            self.adjoint = sum(der * child.gradient() for der, child in self.children)
        return self.adjoint

    def gradient_reset(self, value=None):
        if value:
            self.value = value
        self.children = []
        self.adjoint = 1.0

    def __add__(self, other):
        '''
        Dunder method to add another ReverseNode variable, scalar and vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after addition

        Examples:
        >>> x = ReverseNode(3)
        >>> y = x + 3
        ReverseNode(6)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x1 + x2
        ReverseNode(7)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(self.value + other)
            self.children.append((1.0, new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value + other.value)
            self.children.append((1.0, new))
            other.children.append((1.0, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __radd__(self, other):
        '''
        Dunder method to add another ReverseNode variable, scalar and vector from the left

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after addition

        Examples:
        >>> x = ReverseNode(3)
        >>> y = 3 + x
        ReverseNode(6)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x2 + x1
        ReverseNode(7)

        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        Dunder method to subtract another ReverseNode variable, scalar and vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after subtraction

        Examples:
        >>> x = ReverseNode(3)
        >>> y = x - 2
        ReverseNode(1)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x1 - x2
        ReverseNode(-1)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(self.value - other)
            self.children.append((1.0, new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value - other.value)
            self.children.append((1.0, new))
            other.children.append((-1.0, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __rsub__(self, other):
        '''
        Dunder method to subtract another ReverseNode variable, scalar and vector from the left

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after subtraction

        Examples:
        >>> x = ReverseNode(3)
        >>> y = 5 - x
        ReverseNode(2)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x2 - x1
        ReverseNode(1)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(other - self.value)
            self.children.append((-1.0, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __mul__(self, other):
        '''
        Dunder method to multiply another ReverseNode variable, scalar and vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after multiplication

        Examples:
        >>> x = ReverseNode(3)
        >>> y = x * 3
        ReverseNode(9)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x1 * x2
        ReverseNode(12)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(self.value * other)
            self.children.append((other, new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value * other.value)
            self.children.append((other.value, new))
            other.children.append((self.value, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __rmul__(self, other):
        '''
        Dunder method to multiply another ReverseNode variable, scalar and vector from the left

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after multiplication

        Examples:
        >>> x = ReverseNode(3)
        >>> y = 3 * x
        ReverseNode(9)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x2 * x1
        ReverseNode(12)

        '''
        return self.__mul__(other)

    def __truediv__(self, other):
        '''
        Dunder method to divide another ReverseNode variable, scalar and vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after division

        Examples:
        >>> x = ReverseNode(3)
        >>> y = x / 3
        ReverseNode(1)

        >>> x1 = ReverseNode(3)
        >>> x2 = ReverseNode(4)
        >>> z = x1 / x2
        ReverseNode(0.75)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(self.value / other)
            self.children.append((1 / other, new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value / other.value)
            self.children.append((1 / other.value, new))
            other.children.append((-self.value / other.value ** 2, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __rtruediv__(self, other):
        '''
        Dunder method to divide another ReverseNode variable, scalar and vector from the left

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object containing new value after division

        Examples:
        >>> x = ReverseNode(3)
        >>> y = 3 / x
        ReverseNode(1)

        >>> x1 = ReverseNode(8)
        >>> x2 = ReverseNode(4)
        >>> z = x2 / x1
        ReverseNode(0.5)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(other / self.value)
            self.children.append((-other / self.value ** 2, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __pow__(self, other):
        '''
        Dunder method to compute the power of a ReverseNode variable subject to another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object, containing new value and trace after taking the power

        Examples:
        >>> x = ReverseNode(4)
        >>> y = x ** 2
        ReverseNode(16)

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(2)
        >>> z = x1 ** x2
        ReverseNode(16)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(self.value ** other)
            self.children.append((other * self.value ** (other - 1), new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value ** other.value)
            self.children.append((other.value * (self.value) ** (other.value - 1), new))
            other.children.append((np.log(self.value) * self.value ** other.value, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __rpow__(self, other):
        '''
        Dunder method to compute the power of a ReverseNode variable subject to another ReverseNode variable, scalar or vector from the left

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        a ReverseNode object, containing new value and trace after taking the power

        Examples:
        >>> x = ReverseNode(4)
        >>> y = 2 ** x
        ReverseNode(16)

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(2)
        >>> z = x2 ** x1
        ReverseNode(16)

        '''
        if isinstance(other, (int, float)):
            new = ReverseNode(other ** self.value)
            self.children.append((np.log(other) * other ** self.value, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __neg__(self):
        '''
        Dunder method to take the negation of a ReverseNode variable

        Input:
        self - a ReverseNode variable

        Output:
        The negation of the input ReverseNode variable

        Examples:
        >>> x = ReverseNode(3)
        >>> -x
        ReverseNode(-3)
        '''
        new = ReverseNode(-self.value)
        self.children.append((-1.0, new))
        return new
    
    def __lt__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is less than another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value < other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x < 2
        False

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(8)
        >>> x1 < x2
        True

        '''
        if isinstance(other, (int, float)):
            return self.value < other
        elif isinstance(other, ReverseNode):
            return self.value < other.value
        else:
            raise AttributeError("Invalid Input!")

    def __gt__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is greater than another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value > other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x > 2
        True

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(8)
        >>> x1 > x2
        False

        '''
        if isinstance(other, (int, float)):
            return self.value > other
        elif isinstance(other, ReverseNode):
            return other.__lt__(self)
        else:
            raise AttributeError("Invalid Input!")

    def __le__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is less than or equal to another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value <= other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x <= 3
        True

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(8)
        >>> x1 <= x2
        False

        '''
        # if isinstance(self, (int,float)):
        #    if isinstance(other, (int,float)):
        #        return self <= other
        #    elif isinstance(other, ReverseNode):
        #        return self <= other.value
        if isinstance(self, ReverseNode):
            if isinstance(other, (int, float)):
                return self.value <= other
            elif isinstance(other, ReverseNode):
                return self.value <= other.value
        elif isinstance(other, ReverseNode):
            if isinstance(self, (int, float)):
                return self <= other.value
        raise AttributeError("Invalid Input!")

    def __ge__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is greater than or equal to another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value >= other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x >= 3
        True

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(8)
        >>> x1 >= x2
        False

        '''
        return other.__le__(self)

    def __eq__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is equal to another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value == other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x == 2
        False

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(4)
        >>> x1 == x2
        True

        '''
        if isinstance(self, (int, float)):
            if isinstance(other, (int, float)):
                return self == other
            elif isinstance(other, ReverseNode):
                return self == other.value
        elif isinstance(self, ReverseNode):
            if isinstance(other, (int, float)):
                return self.value == other
            elif isinstance(other, ReverseNode):
                return self.value == other.value
        raise AttributeError("Invalid Input!")

    def __neq__(self, other):
        '''
        Dunder method to compare if the value of a ReverseNode variable is not equal to another ReverseNode variable, scalar or vector

        Input:
        self - a ReverseNode variable
        other - a constant of integers or decimals / a ReverseNode object representing a variable

        Output:
        True if self value != other value, False otherwise

        Examples:
        >>> x = ReverseNode(3)
        >>> x != 2
        True

        >>> x1 = ReverseNode(4)
        >>> x2 = ReverseNode(4)
        >>> x1 != x2
        False

        '''
        return not self.__eq__(other)
    
    def __repr__(self):
        '''
        Dunder method to represent a ReverseNode objects as a string

        Input:
        self - a ReverseNode variable

        Output:
        The value and trace of the ReverseNode object represented as a string

        Examples:
        >>> x = ReverseNode(3)
        >>> repr(x)
        ReverseNode Variable Value: 3, Adjoint: 1.0, Chidren: []

        '''
        return f'ReverseNode Variable Value: {self.value}, Adjoint: {self.adjoint}, Chidren: {self.children}'

    def __str__(self):
        '''
        Dunder method to represent a ReverseNode objects as a string

        Input:
        self - a ReverseNode variable

        Output:
        The value and trace of the ReverseNode object represented as a string

        Examples:
        >>> x = ReverseNode(3)
        >>> print(x)
        ReverseNode Variable Value: 3, Adjoint: 1.0, Chidren: []

        '''
        return f'ReverseNode Variable Value: {self.value}, Adjoint: {self.adjoint}, Chidren: {self.children}'

