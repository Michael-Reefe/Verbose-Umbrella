import umbrella.waves as w
import os
import plotly.subplots as psp
import plotly.graph_objects as pgo
import numpy as np
from numba import jit, njit
import scipy.integrate as spint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Arrow


class FourierSeries:

    __slots__ = ['wave_class', 'save_path', 'name']

    def __init__(self, wave_class, save_path=None, name='FourierSeries'):
        """
        A class for calculating the fourier series expansion of a repeating wave pattern.

        :param wave_class: umbrella.waves.AbstractWave, the wave to be approximated
        :param save_path: str, the path to output graphs
        :param name: str, unused
        """
        assert hasattr(wave_class, 'eval')
        assert hasattr(wave_class, 'period')
        self.wave_class = wave_class
        if not save_path:
            save_path = os.path.abspath(os.path.dirname(__file__)) + '/resources/'
        self.save_path = save_path
        self.name = name

    def __repr__(self):
        return "A Fourier Series approximation of a {} wave".format(self.wave_class.name)


    def integrate(self, timestamp=0, iterations=50, interval=(-np.pi, np.pi), resolution=1000,
                  return_cumsum=False):
        """
        Integrate a fourier series approximation of the given wave pattern.

        :param timestamp: float, time t to draw the wave and approximation
        :param iterations: int, number of fourier terms to include in the approximation
        :param interval: tuple (min, max), where to evaluate the function and fourier series
        :param resolution: int, number of evaluation points
        :param return_cumsum: bool, if true returns the entire cumulative summation at each integration point,
            if false, returns only the final iteration summation
        :return: x values, y values, fourier coefficients (an, bn), cumulative sum / final sum, plotly figure
        """
        x = np.linspace(interval[0], interval[1], resolution)
        y = np.array([self.wave_class.eval(xi, timestamp) for xi in x])
        fig = psp.make_subplots(rows=1, cols=1)
        self._append_plot(fig, x, y, 'Function')
        coefficients = np.zeros((2, iterations))
        generator = self.find_coefficients(self.wave_class.period, self.wave_class.eval)
        summation = np.zeros((len(x)))
        if return_cumsum:
            cumsum = np.zeros((len(x), iterations))
        for i in range(iterations):
            ai, bi = next(generator)
            if i == 0:
                ai /= 2
            coefficients[:, i] = ai, bi
            term = ai*np.cos(2*np.pi*i*(x-self.wave_class.velocity*timestamp)/self.wave_class.period) + \
                   bi*np.sin(2*np.pi*i*(x-self.wave_class.velocity*timestamp)/self.wave_class.period)
            summation += term
            if return_cumsum:
                cumsum[:, i] = summation
            self._append_plot(fig, x, summation)
        if return_cumsum:
            return x, y, coefficients, cumsum, fig
        else:
            return x, y, coefficients, summation, fig

    def complex_animate(self, timestamp=0, iterations=50, interval=(-np.pi, np.pi), resolution=1000):
        x = np.linspace(interval[0], interval[1], resolution)
        y = np.array([self.wave_class.eval(xi, timestamp) for xi in x])
        summation = np.zeros((len(x),), dtype=complex)
        cumsum = np.zeros((len(x), iterations), dtype=complex)
        coefficients = np.zeros((iterations,), dtype=complex)
        generator = self.find_coefficients_complex(self.wave_class.period, self.wave_class.eval)
        for i in range(iterations):
            cR, cI = next(generator)
            cn = cR + complex('j')*cI
            cN = cR.conjugate() + complex('j')*cI.conjugate()
            if i == 0:
                term = cn
            else:
                term = cn*np.exp(complex('j')*2*np.pi*i*x/self.wave_class.period) + \
                    cN*np.exp(-complex('j')*2*np.pi*i*x/self.wave_class.period)
            summation += term
            cumsum[:, i] = summation
        anim = self._create_animation(iterations, cumsum)
        return x, y, coefficients, cumsum, anim

    @staticmethod
    def find_coefficients(period, function):
        """
        A generator function for determining fourier coefficients using scipy.integrate.quad

        :return: an, bn
        """
        # Alias
        per = period

        def an(p, n, fun):
            return fun(p)*np.cos(2*np.pi*n*p/per)

        def bn(p, n, fun):
            return fun(p)*np.sin(2*np.pi*n*p/per)

        n = 0
        while True:
            a = (2/per)*spint.quad(an, -per/2, per/2, args=(n, function,))[0]
            b = (2/per)*spint.quad(bn, -per/2, per/2, args=(n, function,))[0]
            yield a, b
            n += 1

    @staticmethod
    def find_coefficients_complex(period, function):
        per = period

        def an(p, n, fun, r):
            return fun(p)[r]*np.cos(2*np.pi*n*p/per)

        def bn(p, n, fun, r):
            return fun(p)[r]*np.sin(2*np.pi*n*p/per)

        n = 0
        while True:
            aR = (2/per)*spint.quad(an, -per/2, per/2, args=(n, function, 0))[0]
            aI = (2/per)*spint.quad(an, -per/2, per/2, args=(n, function, 1))[0]
            bR = (2/per)*spint.quad(bn, -per/2, per/2, args=(n, function, 0))[0]
            bI = (2/per)*spint.quad(bn, -per/2, per/2, args=(n, function, 1))[0]
            cR = (aR - complex('j')*bR)/2
            cI = (aI - complex('j')*bI)/2
            yield cR, cI
            n += 1

    def _create_animation(self, i, outputarray):
        fig = plt.figure()
        ax = plt.axes(xlim=(np.nanmin(outputarray.real)-1, np.nanmax(outputarray.real)+1),
                      ylim=(np.nanmin(outputarray.imag)-1, np.nanmax(outputarray.imag)+1))
        # ax.plot(inputarray.real, inputarray.imag)
        function, = ax.plot([], [], lw=2)
        arrows = []

        def init():
            function.set_data([], [])
            for ii in range(i):
                arrows.append(Arrow(0, 0, 0, 0))
                ax.add_patch(arrows[ii])
            return function,

        def animate(n):
            function.set_data(outputarray.real[:n:2, -1], outputarray.imag[:n:2, -1])
            for ii in range(i):
                arrows[ii].remove()
            for ii in range(i):
                if ii == 0:
                    arrows[ii] = Arrow(0, 0, outputarray.real[n, 0], outputarray.imag[n, 1], color='k', lw=1)
                else:
                    arrows[ii] = Arrow(outputarray.real[n, ii-1], outputarray.imag[n, ii-1],
                                        outputarray.real[n, ii], outputarray.imag[n, ii], color='k', lw=1)
                ax.add_patch(arrows[ii])
            return function,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(outputarray), blit=True,
                                       interval=100)
        gifwriter = animation.PillowWriter(fps=60)
        anim.save(self.save_path + os.sep + 'fourier_series.gif', writer=gifwriter)
        return

    def _append_plot(self, fig, x, y, name=None):
        """
        Utility function for adding lines to a plotly chart.

        :param fig: plotly figure
        :param x: x data
        :param y: y data
        :param name: string, name of data
        :return: None
        """
        if name:
            fig.add_trace(pgo.Scatter(x=x, y=y, name=name, line=dict(color='black', width=2)))
        else:
            fig.add_trace(pgo.Scatter(x=x, y=y, line=dict(width=.6)))


# class FourierTransformPair:
#
#     def __init__(self, xf=None, xF=None, f=None, F=None):
#         self.xf = xf
#         self.xF = xF
#         self.f = f
#         self.F = F
#         if xf is not None:
#             self.samplerate = xf[1] - xf[0]
#         elif xF is not None:
#             self.samplerate = 1/(xF[1] - xF[0])
#         if f is None and F is None:
#             raise AttributeError("Must define either f or F")
#         if xf is None and xF is None:
#             raise AttributeError("Must define either xf or xF")
#         if (xf is not None and f is None) or (xF is not None and F is None):
#             raise AttributeError("Mistmatch between defined parameters")
#
#     def fft(self):
#         assert self.f is not None
#         assert self.xf is not None
#         n = np.arange(0, len(self.f), 1)
        # self.F = np.fft.fft(self.f)
        # self.F = np.abs(self.F)**2
        # val = 1/(len(self.xf)*1/self.samplerate)
        # N = (len(self.xf)-1)//2 + 1
        # self.xF = np.empty(len(self.xf), np.int)
        # p1 = np.arange(0, N, dtype=np.int)
        # p2 = np.arange(-(len(self.xf)//2), 0, dtype=np.int)
        # self.xF[:N] = p1
        # self.xF[N:] = p2
        # self.xF = self.xF * val
        # return self.xF, self.F
    #
    # def ifft(self):
    #     assert self.F is not None
    #     assert self.xF is not None
    #     k = np.arange(0, len(self.f), 1)
    #     self.f = np.fft.ifft(self.F)
    #     self.f = np.abs(self.f)**2
    #     val = 1/(len(self.xF)*1/self.samplerate)
    #     N = (len(self.xF)-1)//2 + 1
    #     self.xf = np.empty(len(self.xF), np.int)
    #     p1 = np.arange(0, N, dtype=np.int)
    #     p2 = np.arange(-(len(self.xF)//2), 0, dtype=np.int)
    #     self.xf[:N] = p1
    #     self.xf[N:] = p2
    #     self.xf = self.xf * val
    #     return self.xf, self.f
    #
    # def plot(self):
    #     assert self.f is not None
    #     assert self.F is not None
    #     fig = psp.make_subplots(rows=1, cols=2, subplot_titles=("Function", "Transform"))
    #     fig.add_trace(pgo.Scatter(x=self.xf, y=self.f, line=dict(color='blue', width=1)),
    #                   row=1, col=1)
    #     fig.add_trace(pgo.Scatter(x=self.xF, y=self.F, line=dict(color='red', width=1)),
    #                   row=1, col=2)
    #     fig.show()
#
#