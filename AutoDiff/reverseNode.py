
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
        if isinstance(value, (int,float)):
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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x1 + x2
        ForwardNode(7)

        '''
        if isinstance(other, (int,float)):
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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x2 + x1
        ForwardNode(7)

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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x1 - x2
        ForwardNode(-1)

        '''
        if isinstance(other, (int,float)):
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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x2 - x1
        ForwardNode(1)

        '''
        if isinstance(other, (int,float)):
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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x1 * x2
        ForwardNode(12)

        '''
        if isinstance(other, (int,float)):
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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x2 * x1
        ForwardNode(12)

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

        >>> x1 = ForwardNode(3)
        >>> x2 = ForwardNode(4)
        >>> z = x1 / x2
        ForwardNode(0.75)

        '''
        if isinstance(other, (int,float)):
            new = ReverseNode(self.value / other)
            self.children.append((1/other, new))
            return new
        elif isinstance(other, ReverseNode):
            new = ReverseNode(self.value / other.value)
            self.children.append((1 / other.value, new))
            other.children.append((-self.value / other.value**2, new))
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

        >>> x1 = ForwardNode(8)
        >>> x2 = ForwardNode(4)
        >>> z = x2 / x1
        ForwardNode(0.5)

        '''
        if isinstance(other, (int,float)):
            new = ReverseNode(other / self.value)
            self.children.append((-other / self.value**2, new))
            return new
        else:
            raise AttributeError("Invalid Input!")

    def __pow__(self, other):
        '''
        Dunder method to compute the power of a ReverseNode variable subject to another ForwardNode variable, scalar or vector

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
        if isinstance(other, (int,float)):
            new = ReverseNode(self.value ** other)
            self.children.append((other * self.value ** (other-1), new))
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
        Dunder method to compute the power of a ReverseNode variable subject to another ForwardNode variable, scalar or vector from the left

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
        if isinstance(other, (int,float)):
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

     
