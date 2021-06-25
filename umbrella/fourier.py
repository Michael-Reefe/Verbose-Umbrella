import umbrella.waves as w
import os
import plotly.subplots as psp
import plotly.graph_objects as pgo
import numpy as np
from numba import jit, njit
import scipy.integrate as spint


class FourierSeries:

    __slots__ = ['wave_class', 'save_path', 'name']

    def __init__(self, wave_class, save_path=None, name='FourierSeries'):
        assert hasattr(wave_class, 'eval')
        assert hasattr(wave_class, 'period')
        self.wave_class = wave_class
        if not save_path:
            save_path = os.path.abspath(os.path.dirname(__file__)) + '/resources/'
        self.save_path = save_path
        self.name = name

    def run(self, timestamp=0, iterations=50, interval=(-np.pi, np.pi), resolution=1000):
        x = np.linspace(interval[0], interval[1], resolution)
        y = np.array([self.wave_class.eval(xi, timestamp) for xi in x])
        fig = psp.make_subplots(rows=1, cols=1)
        self._append_plot(fig, x, y, 'Function')
        coefficients = np.zeros((2, iterations))
        generator = self._find_coefficients()
        summation = np.zeros((len(x)))
        for i in range(iterations):
            ai, bi = next(generator)
            if i == 0:
                ai /= 2
            coefficients[:, i] = ai, bi
            term = ai*np.cos(2*np.pi*i*(x-self.wave_class.velocity*timestamp)/self.wave_class.period) + \
                   bi*np.sin(2*np.pi*i*(x-self.wave_class.velocity*timestamp)/self.wave_class.period)
            summation += term
            self._append_plot(fig, x, summation)
        # fig.write_html(os.path.join(self.save_path, 'fourier.html'))
        return fig

    def _find_coefficients(self):
        per = self.wave_class.period

        def an(p, n, fun):
            return fun(p)*np.cos(2*np.pi*n*p/per)

        def bn(p, n, fun):
            return fun(p)*np.sin(2*np.pi*n*p/per)

        n = 0
        while True:
            a = (2/per)*spint.quad(an, -per/2, per/2, args=(n, self.wave_class.eval,))[0]
            b = (2/per)*spint.quad(bn, -per/2, per/2, args=(n, self.wave_class.eval,))[0]
            yield a, b
            n += 1

    def _append_plot(self, fig, x, y, name=None):
        if name:
            fig.add_trace(pgo.Scatter(x=x, y=y, name=name, line=dict(color='black', width=2)))
        else:
            fig.add_trace(pgo.Scatter(x=x, y=y, line=dict(width=.6)))