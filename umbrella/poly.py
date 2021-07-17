from umbrella.autodiff import *
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt
from scipy import ndimage


def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)


def fibonacci(n):
    return 1/np.sqrt(5)*(((1 + np.sqrt(5))/2)**n - ((1 - np.sqrt(5))/2)**n)


def choose(n, k):
    return np.prod([(n+1-i)/i for i in range(1, k+1)])


def _P(x, l):
    return sum([choose(l, k) * choose(-l-1, k) * np.power((1-x)/2, k) for k in range(l+1)])

def _autodifferentiableP(x, l):
    s = np.array([])
    for xi in x:
        x = Variable(xi)
        one = Variable(1.)
        c = 1/(2 ** l)
        si = Variable(0.)
        for k in range(l):
            x1 = Variable(1.)
            x2 = Variable(1.)
            for exp in range(l - k):
                x1 *= (x - one)
            for exp in range(k):
                x2 *= (x + one)
            si += Variable(c) * (Variable(choose(l, k)) * Variable(choose(l, k)) * x1 * x2)
        np.append(s, si)
    return x, s


def legendre(x, l, m=0):
    """
    Calculate the associated Legendre function Pnm of order n, m at the value x.

    :param l: int, order of the polynomial
    :param m: int, order of the associated function
    :param x: float, value to evaluate the function at.
    :return: float, Pnm(x)
    """
    if np.abs(m) > l:
        raise ValueError("abs(m) must be smaller than n")
    if m < 0:
        m = np.abs(m)
        return (-1) ** m * factorial(l - m) / factorial(l + m) * legendre(x, l, m)
    return (-1)**m * 2**l * (1 - x**2)**(m / 2) * sum([factorial(k)/factorial(k - m) * x**(k - m) * choose(l, k) * choose((l + k - 1) / 2, l) for k in range(m, l + 1)])


def hermite(x, n):
    return factorial(n) * sum([(-1)**m/(factorial(m)*factorial(n-2*m)) * (2*x)**(n-2*m) for m in range(n//2+1)])


def _L(x, q):
    return sum([(-1)**k/factorial(k) * choose(q, k) * x**k for k in range(q+1)])


def laguerre(x, q, p=0):
    return sum([(-1)**m * factorial(q+p)/(factorial(q-m)*factorial(p+m)*factorial(m)) * x**m for m in range(q+1)])


def plot_legendre(m=0, l=None, *args, **kwargs):
    assert type(m) is int
    if not l:
        l = np.arange(np.abs(m), np.abs(m)+5, 1)
    x = np.linspace(-1, 1, 1000)
    fig, ax = plt.subplots()
    y = []
    for li in l:
        yi = legendre(x, l, m)
        y.append(yi)
        ax.plot(x, yi, label='$P_{%d}^{%d}$' % (li, m), *args, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(l), fancybox=True)
    plt.show()
    return x, y


def polar_plot_legendre(m=0, l=None, *args, **kwargs):
    assert type(m) is int
    if not l:
        l = np.arange(np.abs(m), np.abs(m)+5, 1)
    theta = np.linspace(-np.pi, np.pi, 1000)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi/2)
    r = []
    for li in l:
        ri = np.abs(legendre(np.cos(theta), li, m))
        r.append(ri)
        ax.plot(theta, ri, label='$P_{%d}^{%d}$' % (li, m), *args, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(l), fancybox=True)
    plt.show()
    return theta, r


def plot_hermite(n=None, zoom=None, *args, **kwargs):
    if not n:
        n = np.arange(0, 5, 1)
    x = np.linspace(-3, 3, 1000)
    fig, ax = plt.subplots()
    y = []
    for ni in n:
        yi = hermite(x, ni)
        y.append(yi)
        ax.plot(x, yi, label='$H_{%d}$' % ni, *args, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(n), fancybox=True)
    if zoom:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    plt.show()
    return x, y


def plot_laguerre(p=0, q=None, zoom=None, *args, **kwargs):
    assert type(p) is int
    if not q:
        q = np.arange(np.abs(p), np.abs(p)+5, 1)
    x = np.linspace(-2, 10, 1000)
    fig, ax = plt.subplots()
    y = []
    for qi in q:
        yi = laguerre(x, qi, p)
        y.append(yi)
        ax.plot(x, yi, label='$L_{%d}^{%d}$' % (qi, p), *args, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(q), fancybox=True)
    if zoom:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    plt.show()
    return x, y


def _harmonicY(theta, phi, l, m):
    return np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m)) * legendre(np.cos(theta), l, m) * np.exp(m * phi * complex('j'))


def harmonicY(theta, phi, l, m):
    """
    Calculate the spherical harmonic function Ylm of order l,m at the value theta, phi.

    :param theta: float, radians, to evaulate function at
    :param phi: float, radians, to evaluate function at
    :param l: int, order of associated Legendre function
    :param m: int, order of associated Legendre function
    :return: float, Ylm(theta, phi)
    """
    if m < 0:
        return (-1) ** m * np.sqrt(2) * np.sqrt((2*l+1)/(4*np.pi)*factorial(l-np.abs(m))/factorial(l+np.abs(m))) \
               * legendre(np.cos(theta), l, np.abs(m)) * np.sin(np.abs(m) * phi)
    elif m == 0:
        return np.sqrt((2*l+1)/(4*np.pi)) * legendre(np.cos(theta), l, m)
    else:
        return (-1) ** m * np.sqrt(2) * np.sqrt((2*l+1)/(4*np.pi)*factorial(l-np.abs(m))/factorial(l+np.abs(m))) \
               * legendre(np.cos(theta), l, m) * np.cos(m * phi)


def plot_harmonics(m=0, l=None, resolution=1, *args, **kwargs):
    assert type(m) is int
    if not l:
        l = 0
    theta = np.linspace(0, np.pi, 100*resolution)
    phi = np.linspace(-np.pi, np.pi, 100*resolution)
    THETA, PHI = np.meshgrid(theta, phi)
    R = np.abs(harmonicY(THETA, PHI, l, m))
    X = R * np.cos(PHI) * np.sin(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(THETA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(X, Y, Z, label='$Y_{%d}^{%d}$' % (l, m), rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
                    linewidth=0, antialiased=False, alpha=0.5, *args, **kwargs)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_xlim(-.5,.5)
    ax.set_ylim(-.5,.5)
    ax.set_zlim(-.5,.5)
    ax.view_init(30, 45)
    plt.show()
    return X, Y, Z


def hydrogen(r, theta, phi, n, l, m):
    # Bohr radius
    a = 0.529e-10
    return np.sqrt((2/(n*a))**3 * factorial(n-l-1)/(2*n*factorial(n+l))) \
        * np.exp(-r/(n*a)) * (2*r/(n*a))**l * laguerre(2*r/(n*a), n-l-1, 2*l+1) * _harmonicY(theta, phi, l, m)


def hydrogen_cartesian(x, y, z, n, l, m):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    phi = np.arctan2(y, x)
    return hydrogen(r, theta, phi, n, l, m)


def plot_hydrogen(n=1, l=0, m=0, resolution=1, *args, **kwargs):
    # Bohr radius
    a = 0.529e-10
    assert type(n) is int
    assert type(l) is int
    assert type(m) is int
    x = np.linspace(-30*a, 30*a, 100*resolution)
    y = 0
    z = np.linspace(-30*a, 30*a, 100*resolution)
    X, Z = np.meshgrid(x, z)
    density = np.abs(hydrogen_cartesian(X, y, Z, n, l, m))**2
    fig, ax = plt.subplots()
    ax.pcolor(X/a, Z/a, density, cmap='gnuplot', shading='nearest')
    plt.show()
    return X, y, Z, density
