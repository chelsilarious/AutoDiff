import numpy as np


class ForwardNode():
    def __init__(self, value, trace=1.0):
        self.value = value
        self.trace = trace

    def __add__(self, other):
        try:
            new = ForwardNode(self.value + other.value, self.trace + other.trace)
        except AttributeError:
            new = ForwardNode(self.value + other, self.trace)
        return new

    def __radd__(self, other):

        return self.__add__(other)

    def __sub__(self, other):
        
        return self.__add__(-1 * other)

    def __rsub__(self, other):

        return (-1 * self).__add__(other)
    
    def __mul__(self, other):
        try:
            new = ForwardNode(self.value * other.value, self.value * other.trace + self.trace + other.value)
        except AttributeError:
            new = ForwardNode(self.value * other, self.trace * other)
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            new = ForwardNode(self.value / other.value, (self.trace * other.value - self.value * other.trace) / (other.value ** 2))
        except AttributeError:
            new = ForwardNode(self.value / other, self.trace / other)
        return new

    def __rtruediv__(self, other):
        try:
            new = ForwardNode(other.value / self.value, (other.trace * self.value - other.value * self.trace)/(self.value ** 2))
        except AttributeError:
            new = ForwardNode(other / self.value, other * (-self.value**(-2)) * self.trace)
        return new

    def __pow__(self, other):
        try:
            value = self.value ** other.value
            trace = other.trace * (self.value ** (other.value - 1)) * self.trace + (self.value ** other.value) * np.log(self.value) * other.trace
            new = ForwardNode(value, trace)
        except AttributeError:
            new = ForwardNode(self.value ** other, self.trace * other * self.value ** (other - 1))
        return new

    def __rpow__(self, other):
        try:
            value = other.value ** self.value
            trace = self.trace * (other.value ** (self.value - 1)) * other.trace + (other.value ** self.value) * np.log(other.value) * self.trace
            new = ForwardNode(value, trace)
        except AttributeError:
            new = ForwardNode(other ** self.value, other ** self.value * np.log(other) * self.trace)
        return new

    def __repr__(self):
        return 'Value: ' + str(self.value) + ' , Derivative: ' + str(self.trace)

    def __str__(self):
        return 'Value: ' + str(self.value) + ' , Derivative: ' + str(self.trace)


