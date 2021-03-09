import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

hbar = 1.054e-34

# Adjustable parameters
b = 1
D = .5
T0 = 100
t = b**2/2
iterations = 100

a = 1
m = 9.109e-31
t2 = a**2*2*m / (hbar * np.pi**2)


def T(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    global b, D
    for ti in t:
        last = 0
        for n in range(iterations):
            theta = (np.pi/2 + np.pi*n)/b
            current = -(2*T0)/(theta*b)*(-1)**n * np.cos(theta*x) * np.exp(-D*theta**2*ti)
            last += current
            n += 1
        yield T0 + last


def psi(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    global a, m, hbar
    for ti in t:
        last = 0
        for n in range(1, iterations):
            En = (hbar**2)/(2*m) * (n*np.pi/a)**2
            current = -2*np.sqrt(2)/(n*np.pi*np.sqrt(a))*((-1)**n - np.cos(n*np.pi/2)) * np.sin(n*np.pi*x/a) * \
                      np.exp(-complex('j')*En*ti/hbar)
            last += current
            n += 1
        yield last


probe_x1 = np.linspace(0, b, 1000)
probe_x2 = np.linspace(0, a, 1000)
probe_t = np.array([0, t/2, t, 10*t])
# probe_t = np.linspace(0, 10*t, 30)
probe_t2 = np.array([0, t2/2, np.pi*t2, 10*t2])
# probe_t2 = np.linspace(0, np.pi*t2, 100)

T_generator = T(probe_x1, probe_t)
T_vals = np.array([next(T_generator) for i in range(len(probe_t))], dtype=float).transpose()
psi_generator = psi(probe_x2, probe_t2)
psi_vals = np.array([next(psi_generator) for i in range(len(probe_t2))], dtype=complex).transpose()


colors = 'rgbymc'
fig, ax = plt.subplots()
for i in range(len(probe_t)):
    ax.plot(probe_x1, T_vals[:, i], '{}-'.format(colors[i % 6]), label='$t={:.2f} s$'.format(probe_t[i]))
ax.set_xlabel('$x [m]$')
# ax.set_xticks(np.linspace(0, b, 5))
ax.set_ylabel('$u(x,t)$')
ax.set_ylim(-5, T0 + 5)
plt.grid()
plt.title('Temperature Diffusion through slab, diffusion constant $D = %.1f$' % D)
plt.legend()
plt.savefig('out.T.png', dpi=300)
plt.close()



for i in range(len(probe_t2)):
    fig, ax = plt.subplots()
    # ax.plot(probe_x2, psi_vals[:, i].real, 'r-', label='$Re\\ t={:.2e} s$'.format(probe_t2[i]))
    # ax.plot(probe_x2, psi_vals[:, i].imag, 'b-', label='$Im\\ t={:.2e} s$'.format(probe_t2[i]))
    ax.plot(probe_x2, np.abs(psi_vals[:, i])**2, '{}-'.format(colors[i % 6]), label='$t={:.2e} s$'.format(probe_t2[i]))
    ax.set_xlabel('$x [m]$')
    ax.set_xticks(np.linspace(0, a, 5))
    ax.set_ylabel('$|\Psi (x,t)|^2$')
    plt.grid()
    plt.title('Wavefunction of particle in infinite potential well')
    plt.legend()
    plt.savefig('out.psi_{}.png'.format(i + 1), dpi=300)
    plt.close()

