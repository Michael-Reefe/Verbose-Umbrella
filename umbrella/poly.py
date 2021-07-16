from umbrella.autodiff import *


def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)


def fibonacci(n):
    return 1/np.sqrt(5)*(((1 + np.sqrt(5))/2)**n - ((1 - np.sqrt(5))/2)**n)


def choose(n, k):
    return int(factorial(n)/(factorial(k)*factorial(n-k)))


def gen_choose(a, k):
    a0 = a
    for i in range(1, k):
        a *= a0 - i
    return a / factorial(k)


def _P(x, n):
    if isinstance(x, np.ndarray):
        raise ValueError("cannot evaluate entire array at once")
    return 1/2**n * np.sum([choose(n, k)**2 * (x-1)**(n-k)*(x+1)**k for k in range(n+1)])


def _autodiffP(x, n):
    if isinstance(x, np.ndarray):
        raise ValueError("cannot evaluate entire array at once")
    x = Variable(x)
    one = Variable(1.)
    c = 1/(2**n)
    s = Variable(0.)
    for k in range(n):
        x1 = Variable(1.)
        x2 = Variable(1.)
        for exp in range(n-k):
            x1 *= (x - one)
        for exp in range(k):
            x2 *= (x + one)
        s += Variable(c) * (Variable(choose(n, k))*Variable(choose(n, k)) * x1 * x2)
    return x, s


def legendreP(x, n, m=0):
    """
    Calculate the associated Legendre function Pnm of order n, m at the value x.

    :param n: int, order of the polynomial
    :param m: int, order of the associated function
    :param x: float, value to evaluate the function at.
    :return: float, Pnm(x)
    """
    if np.abs(m) > n:
        raise ValueError("abs(m) must be smaller than n")
    if m == 0:
        return _P(x, n)
    elif m < 0:
        m = np.abs(m)
        return (-1)**m * factorial(n - m)/factorial(n + m) * legendreP(x, n, m)
    return (-1)**m * 2**n * (1 - x**2)**(m/2) * np.sum([factorial(k)/factorial(k-m) * x**(k-m) * gen_choose(n, k) * gen_choose((n+k-1)/2, n) for k in range(m, n+1)])

