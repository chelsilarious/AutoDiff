import numpy as np

class ReverseNode():
    def __init__(self, value):
        self.value = value
        self.children = []
        self.adjoint = 1.0

    def gradient(self):
        if len(self.children) > 0:
            self.adjoint = sum(coef * child.gradient() for coef, child in self.children)
        return self.adjoint

    def __add__(self, other):
        new = ReverseNode(self.value + other.value)
        self.children.append((1.0, new))
        other.children.append((1.0, new))
        return new

    def __sub__(self, other):
        new = ReverseNode(self.value - other.value)
        self.children.append((1.0, new))
        other.children.append((-1.0, new))
        return new

    def __mul__(self, other):
        new = ReverseNode(self.value * other.value)
        self.children.append((other.value, new))
        other.children.append((self.value, new))
        return new

    def __pow__(self, other):
        new = ReverseNode((self.value) ** (other.value))
        self.children.append((other.value * ((self.value) ** (other.value - 1)), new))
        other.children.append((np.log(self.value) * (self.value) ** (other.value), new))
        return new
        
    def __eq__(self, other):
        raise NotImplementedError
