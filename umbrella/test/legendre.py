import numpy as np
from matplotlib import pyplot as plt
import time


def legendre(l: int, x: np.ndarray) -> np.ndarray:
    return np.full(np.size(x), 1, dtype=float) if l == 0 \
        else np.full(np.size(x), x, dtype=float) if l == 1 \
        else ((2*l-1)*x*legendre(l - 1, x) - (l-1)*legendre(l - 2, x)) / l


x = np.linspace(-1, 1, 2001)
styles = {0: 'r-', 1: 'b-', 2: 'g-', 3: 'y-', 4: 'c-', 5: 'm-'}
plt.figure()
plt.xlim(-1, 1)
plt.xlabel('$x$')
plt.ylabel('$P_{\mathcal{l}}(x)$')
plt.title('Legendre Polynomials')
plt.grid()

for i in range(4):
    y = legendre(i, x)
    plt.plot(x, y, styles[i % 6], label='$P_{%d}(x)$' % i, linewidth=0.7)

plt.legend()
plt.savefig('out.legendre.png', dpi=300)
