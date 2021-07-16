from umbrella.autodiff import *
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.pyplot as plt


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


def legendreP(x, l, m=0):
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
        return (-1) ** m * factorial(l - m) / factorial(l + m) * legendreP(x, l, m)
    return (-1)**m * 2**l * (1 - x**2)**(m / 2) * sum([factorial(k)/factorial(k - m) * x**(k - m) * choose(l, k) * choose((l + k - 1) / 2, l) for k in range(m, l + 1)])


def plot_legendre(m=0, l=None):
    assert type(m) is int
    if not l:
        l = np.arange(np.abs(m), np.abs(m)+5, 1)
    x = np.linspace(-1, 1, 1000)
    fig, ax = plt.subplots()
    y = []
    for li in l:
        yi = [legendreP(xi, li, m) for xi in x]
        y.append(yi)
        ax.plot(x, yi, label='$P_{%d}^{%d}$' % (li, m))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(l), fancybox=True)
    plt.show()
    return x, y


def polar_plot_legendre(m=0, l=None):
    assert type(m) is int
    if not l:
        l = np.arange(np.abs(m), np.abs(m)+5, 1)
    theta = np.linspace(-np.pi, np.pi, 1000)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi/2)
    r = []
    for li in l:
        ri = np.abs(legendreP(np.cos(theta), li, m))
        r.append(ri)
        ax.plot(theta, ri, label='$P_{%d}^{%d}$' % (li, m))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(l), fancybox=True)
    plt.show()
    return theta, r


def _harmonicY(theta, phi, l, m):
    return np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m)) * legendreP(np.cos(theta), l, m) * np.exp(m*phi*complex('j'))


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
        return (-1)**m * np.sqrt(2) * np.sqrt((2*l+1)/(4*np.pi)*factorial(l-np.abs(m))/factorial(l+np.abs(m))) \
               * legendreP(np.cos(theta), l, np.abs(m)) * np.sin(np.abs(m)*phi)
    elif m == 0:
        return np.sqrt((2*l+1)/(4*np.pi)) * legendreP(np.cos(theta), l, m)
    else:
        return (-1)**m * np.sqrt(2) * np.sqrt((2*l+1)/(4*np.pi)*factorial(l-np.abs(m))/factorial(l+np.abs(m))) \
               * legendreP(np.cos(theta), l, m) * np.cos(m * phi)


def plot_harmonics(m=0, l=None):
    assert type(m) is int
    if not l:
        l = 0
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(-np.pi, np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)
    R = np.abs(harmonicY(THETA, PHI, l, m))
    X = R * np.cos(PHI) * np.sin(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(THETA)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(X, Y, Z, label='$Y_{%d}^{%d}$' % (l, m), rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
                    linewidth=0, antialiased=False, alpha=0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_xlim(-.5,.5)
    ax.set_ylim(-.5,.5)
    ax.set_zlim(-.5,.5)
    ax.view_init(30, 45)
    plt.show()
    return X, Y, Z