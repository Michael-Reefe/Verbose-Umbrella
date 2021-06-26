from scipy.misc import derivative


def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)


def legendreP(x, n):
    """
    Calculate the legendre polynomial Pn of order n at the value x.

    :param n: int, order of the polynomial
    :param x: float, value to evaluate the function at.
    :return: float, Pn(x)
    """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*legendreP(x,n-1) - (n-1)*legendreP(x,n-2))/n


def assoc_legendreP(x, m, n):
    if m >= 0:
        dPl = derivative(legendreP, x, dx=1e-10, n=m, args=(n,))
        return (-1)**m*(1-x**2)**(m/2)*dPl
    else:
        return (-1)**m*factorial(n - m)/factorial(n + m)*legendreP(x, n)

