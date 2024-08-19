import numpy as np
from functools import partial

def optimize(start, fun):

    delta = 10**(-3)
    eps = 10**(-3)
    last_val = start
    first_der_func = partial(derivative, fun=fun, eps=eps)
    first_der = first_der_func(last_val)
    second_der = derivative(last_val, first_der_func, eps)
    next_val = last_val - (first_der/second_der)

    while abs(next_val - last_val) >= delta:
        last_val = next_val
        first_der_func = partial(derivative, fun=fun, eps=eps)
        first_der = first_der_func(last_val)
        second_der = derivative(last_val, first_der_func, eps)
        next_val = last_val - (first_der/second_der)

    return (next_val, fun(next_val))


def derivative(x, fun, eps):
    der = (fun(x+eps) - fun(x))/eps
    return der
