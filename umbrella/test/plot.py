import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import accumulate

# Adjustable parameters
a = 1
b = 2
iterations = 30


def T_2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    global a, b
    n = 0
    last = 0
    while True:
        n += 1
        k = n*np.pi/b
        current = (-1)**(n+1)*100*b/(n*np.pi) * (np.exp(k*x) - np.exp(k*(2*a - x))) / (1 - np.exp(2*k*a)) * np.sin(k*y)
        yield last + current
        last += current


def T_3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    global a, b
    n = 0
    last = 0
    while True:
        n += 1
        k = n*np.pi/b
        part_1 = (-1)**(n+1)*100*b/(n*np.pi) * (np.exp(k*x) - np.exp(k*(2*a - x))) / (1 - np.exp(2*k*a)) * np.sin(k*y)
        part_2 = (n % 2)*(400/(n*np.pi)) * np.sinh(n*np.pi*y/a)/np.sinh(n*np.pi*b/a) * np.sin(n*np.pi*x/a)
        current = part_1 + part_2
        yield last + current
        last += current


probe_x = np.linspace(0, a, 100)
probe_y = np.linspace(0, b, 100)
AX_X, AX_Y = np.meshgrid(probe_x, probe_y)

T2_generator = T_2(AX_X, AX_Y)
AX_T2 = np.swapaxes(np.array([next(T2_generator) for i in range(iterations)], dtype=float).transpose(), 0, 1)

T3_generator = T_3(AX_X, AX_Y)
AX_T3 = np.swapaxes(np.array([next(T3_generator) for i in range(iterations)], dtype=float).transpose(), 0, 1)


def plotter(Z, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X=AX_X, Y=AX_Y, Z=Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('$x$')
    ax.set_xticks(np.arange(0, 1.5, 0.5))
    ax.set_ylabel('$y$')
    ax.set_yticks(np.arange(0, 2.5, 0.5))
    ax.set_zlabel('$T(x,y)$')
    ax.set_zticks(np.arange(0, 120, 20))
    plt.savefig(name, dpi=300)
    plt.close()


# for i in range(iterations):
plotter(AX_T2[:, :, iterations - 1], 'out.T2_{}.png'.format(iterations))
plotter(AX_T3[:, :, iterations - 1], 'out.T3_{}.png'.format(iterations))