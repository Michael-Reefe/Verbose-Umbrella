# All from https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a
# and https://sidsite.com/posts/autodiff
import numpy as np
from collections import defaultdict


class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return add(self, neg(other))

    def __truediv__(self, other):
        return div(self, other)

    # def __pow__(self, other):
    #     return pow(self, other)


ONE = Variable(1.)
NEG_ONE = Variable(-1.)


def add(a, b):
    value = a.value + b.value
    local_gradients = (
        (a, lambda path_value: path_value),
        (b, lambda path_value: path_value)
    )
    return Variable(value, local_gradients)


def mul(a, b):
    value = a.value * b.value
    local_gradients = (
        (a, lambda path_value: path_value * b),
        (b, lambda path_value: path_value * a)
    )
    return Variable(value, local_gradients)


def div(a, b):
    value = a.value / b.value
    local_gradients = (
        (a, lambda path_value: path_value * ONE/b),
        (b, lambda path_value: path_value * NEG_ONE * a/(b*b))
    )
    return Variable(value, local_gradients)


# def pow(a, b):
#     value = np.power(a.value, b.value)
#     local_gradients = (
#         (a, lambda path_value: path_value * b * pow(a, b-ONE)),
#         (b, lambda path_value: path_value * log(a)*pow(a, b))
#     )
#     return Variable(value, local_gradients)


def neg(a):
    value = -1 * a.value
    local_gradients = (
        (a, lambda path_value: NEG_ONE * path_value),
    )
    return Variable(value, local_gradients)


def log(a):
    value = np.log(a.value)
    local_gradients = (
        (a, lambda path_value: path_value * ONE/a),
    )
    return Variable(value, local_gradients)


def get_gradients(variable):
    gradients = defaultdict(lambda: Variable(0))

    def compute_gradients(variable, path_value):
        for child, logcgrad in variable.local_gradients:
            path_child = logcgrad(path_value)
            gradients[child] += path_child
            compute_gradients(child, path_child)

    compute_gradients(variable, path_value=ONE)
    return gradients