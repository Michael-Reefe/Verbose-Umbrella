from umbrella.fourier import AnalyticFourierSeries, FourierTerm
from umbrella.fourier import Sawtooth, SquareRect
import os
import numpy as np

def wacky_wave(x, A, P, p):
    return A*np.exp(-2*(x - p)**2/(P**2/8))

ft = FourierTerm(n=4, a=2, b=3, period=2*np.pi)
print(ft)
sw = Sawtooth(5, 3.2, 0.1, 0)
print(sw)

a = 2
sq = SquareRect(1, a, a/2, 0)
icfunc = lambda x: sq.function(x, 0)

fs = AnalyticFourierSeries(50, (0, a), 3, 1.5, 0.2, 'square', os.path.abspath(os.path.dirname(__file__)), plot_backend='plotly')
fs.evaluate_func()
fs.create_fourier_terms()
fs.evaluate_fourier_series()
fs.make_dataframe()
fs.save()
fs.plot(29, 30, 42, suffix='1')

fs.plot_backend = 'mpl'
fs.iterations = 80
fs.create_fourier_terms()
fs.evaluate_fourier_series()
fs.make_dataframe()
fs.plot(31, 18, 22, suffix='2')

fs.plot_backend = 'plotly'
fs.interval = (-30, 30)
fs.resolution = 3000
fs.evaluate_fourier_series()
fs.make_dataframe()
fs.plot(31, 18, 22, suffix='3')

fs.amplitude = 10
fs.period = 10.1
fs.phase = -0.2
fs.evaluate_func()
fs.create_fourier_terms()
fs.evaluate_fourier_series()
fs.make_dataframe()
fs.plot(31, 18, 22, suffix='4')


ss = AnalyticFourierSeries.gaussian(os.path.abspath(os.path.dirname(__file__)))
ss.evaluate_func()
ss.create_fourier_terms()
ss.evaluate_fourier_series()
ss.make_dataframe()
ss.save()
ss.plot(42, suffix='5')


def function(xi, amplitude, period, phase):
    factor = ((xi) - phase) // (period / 2)
    return amplitude * (
                (xi) - phase - (period / 2) * factor) if factor % 2 == 0 else \
        -amplitude * (-(xi) + phase + (period / 2) * factor)


fs2 = AnalyticFourierSeries(50, (-5, 5), 2, 1.5, 1, 'custom', os.path.abspath(os.path.dirname(__file__)), custom_func=function,
                            plot_backend='plotly')
realfunc_data = fs2.evaluate_func()
coefficients, fourierterms = fs2.create_fourier_terms()
full_data_array = fs2.evaluate_fourier_series()
full_dataframe = fs2.make_dataframe()
fs2.save()
fs2.plot(49, suffix='6')

import matplotlib.pyplot as plt
xi = fs2.fourierterms.fourier_iteration(4, fs2.x)
plt.plot(fs2.x, xi, 'b-')
plt.plot(fs2.x, fs2.y, 'k-')
plt.savefig('testiter4.png', dpi=300)