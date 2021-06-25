from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from scipy.integrate import quad
from umbrella.waves import AbstractWave, Sine, Cosine, Square, Triangle, SquareRect, TriangleRect, SawtoothRect, Sawtooth, Gaussian
try:
    import plotly.express as px
except ImportError:
    print('Warning: Could not import plotly!  Only matplotlib will be available as a backend.')


class FourierTerm:

    def __init__(self, n: int, a: float, b: float, period: float):
        self.a = a
        self.b = b
        self.n = n
        self.period = period

    def compute(self, x):
        if self.n == 0:
            return self.a / 2
        cos = self.a * np.cos(2 * np.pi * self.n * x / self.period)
        sin = self.b * np.sin(2 * np.pi * self.n * x / self.period)
        return cos + sin

    def __repr__(self):
        if self.n == 0:
            return "{:.8f} / 2".format(self.a)
        return "{:.1f}cos(2pi * {:d}x / {:.1f}) + {:.1f}sin(2pi * {:d}x / {:.1f})".format(
            self.a, self.n, self.period, self.b, self.n, self.period)


class FourierTerms(list):

    def __init__(self):
        super().__init__()

    def compute(self, x):
        c = np.array([self[i].compute(x) for i in range(len(self))])
        return c

    def cumsum(self, x):
        if isinstance(x, np.ndarray):
            s = np.zeros(shape=(len(self), len(x)))
            for i in range(len(x)):
                ci = self.compute(x[i])
                s[:, i] = np.cumsum(ci)
        else:
            s = np.cumsum(self.compute(x))
        return s

    def fourier_iteration(self, iteration, x):
        terms = self[:iteration]
        if isinstance(x, np.ndarray):
            result = np.ndarray(shape=(len(x),))
            for j, xi in enumerate(x):
                result[j] = np.nansum([terms[i].compute(xi) for i in range(len(terms))])
            return result
        return np.nansum(terms[i].compute(x) for i in range(len(terms)))

    def add_fourier(self, fourier):
        self.append(fourier)

    def pretty_print(self):
        for i in range(len(self)):
            print(repr(self[i]) + '\n')


class AnalyticFourierSeries:

    __slots__ = ['_iterations', '_interval', '_amplitude', '_period', '_phase', '_type', 'save_path',
                 '_resolution', 'styles', '_func', 'len', 'plot_backend', 'x', 'y', 'fs', 'dataframe',
                 'coefficients', 'fourierterms', 'generator', 'name', '__start_index']

    def __init__(self, iterations: int, interval: tuple, amplitude: float, period: float, phase: float, type: str, save_path: str,
                 custom_func=None, resolution: int = 1000, styles: dict = None, plot_backend='mpl', name='FourierSeries'):
        self._type = type
        self._amplitude = amplitude
        self._period = period
        self._phase = phase
        self.save_path = save_path
        self._iterations = iterations
        self._interval = interval
        self.len = self._interval[1] - self._interval[0]
        self._resolution = resolution
        self._func = self._initialize_func(type, custom_func)
        assert plot_backend in ('mpl', 'plotly'), "Currently only matplotlib and plotly are supported for plotting!"
        self.plot_backend = plot_backend
        if styles is None:
            styles = {0: 'r-', 1: 'b-', 2: 'g-', 3: 'y-', 4: 'c-', 5: 'm-'}
        assert len(styles) > 0, "Styles must not be an empty dict!"
        self.styles = styles
        self.x = np.linspace(self._interval[0], self._interval[1], self._resolution)
        self.y = np.full(shape=(len(self.x),), fill_value=np.nan)
        self.fs = np.full(shape=(self._iterations, self._resolution), fill_value=np.nan)
        self.dataframe = None
        self.coefficients = np.full(shape=(2, self._iterations), fill_value=np.nan, dtype=float)
        self.fourierterms = FourierTerms()
        self.generator = self._find_coefficients()
        self.name = name
        self.__start_index = 0
        self.evaluate_func()
        self.create_fourier_terms()
        self.evaluate_fourier_series()
        self.make_dataframe()

    def run(self):
        self.evaluate_func()
        self.create_fourier_terms()
        self.evaluate_fourier_series()
        self.make_dataframe()

    def __repr__(self):
        return "Fourier Series Approximation of a {} wave.\n" \
               "Iterations: {}, Interval: {}, Resolution: {}".format(self._type, self._iterations,
                                                                                    self._interval,
                                                                                    self._resolution)

    def _initialize_func(self, type, custom_func):
        valid_types_dict = {'square': Square,
                            'square_rect': SquareRect,
                            'sin': Sine,
                            'cos': Cosine,
                            'triangle': Triangle,
                            'triangle_rect': TriangleRect,
                            'sawtooth': Sawtooth,
                            'sawtooth_rect': SawtoothRect,
                            'gaussian': Gaussian,
                            'custom': custom_func}
        assert type in valid_types_dict.keys(), "Type must be one of {}".format(valid_types_dict.keys())
        if type == 'custom':
            if not hasattr(custom_func, '__call__'):
                raise ValueError('If you use a custom type, you must specify custom_func as a callable function with'
                                 'arguments (x, amplitude, period, phase)!')
            else:
                function = lambda x: custom_func(x, self._amplitude, self._period, self._phase)
        else:
            function = lambda x: valid_types_dict[type](self._amplitude, self._period, self._phase, 0).eval(x, 0)
        return function

    def evaluate_func(self):
        self.y[:] = [self._func(xi) for xi in self.x]
        return self.y

    def _find_coefficients(self):
        n = 0
        while True:
            def a_func(p, fun):
                return fun(p) * np.cos(2 * np.pi * n * p / self._period)

            def b_func(p, fun):
                return fun(p) * np.sin(2 * np.pi * n * p / self._period)

            an = (2 / self._period) * quad(a_func, -self._period / 2, self._period / 2, args=(self._func,))[0]
            bn = (2 / self._period) * quad(b_func, -self._period / 2, self._period / 2, args=(self._func,))[0]
            yield an, bn
            n += 1

    def create_fourier_terms(self):
        if self.__start_index == 0:
            self.fourierterms = FourierTerms()
            self.generator = self._find_coefficients()
        for i in range(self._iterations):
            an, bn = next(self.generator)
            self.coefficients[:, i] = an, bn
            ft = FourierTerm(n=i, a=an, b=bn, period=self._period)
            self.fourierterms.add_fourier(ft)
        return self.coefficients, self.fourierterms

    def evaluate_fourier_series(self):
        # for i in range(self._iterations):
        #     self.fs[i, :] = self.fourierterms.compute_fourier_iteration(i, self.x)
        self.fs = self.fourierterms.cumsum(self.x)
        return self.fs

    def make_dataframe(self):
        iter_arrays = tuple()
        dtype = [('iteration', object), ('x', float), ('y', float)]
        realfunc = pd.DataFrame(np.array([('realfunc', xi, yi) for xi, yi in zip(self.x, self.y)], dtype=dtype))
        for i in range(self._iterations):
            iter_arrays += (
                pd.DataFrame(np.array([('iter' + str(i+1), xi, yi) for xi, yi in zip(self.x, self.fs[i, :])], dtype=dtype)),
            )
        self.dataframe = pd.concat((realfunc,) + iter_arrays, axis=0)
        return self.dataframe

    def save(self, **kwargs):
        path = os.path.join(self.save_path, self.name + '.out.coefficients')
        path2 = os.path.join(self.save_path, self.name + 'out.data')
        if 'date' in kwargs.keys():
            if kwargs['date']:
                path += '.' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
                path2 += '.' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if 'compressed' in kwargs.keys():
            if kwargs['compressed']:
                np.savez_compressed(path, coeff=self.coefficients.T)
        elif 'txt' in kwargs.keys():
            if kwargs['txt']:
                np.savetxt(path + '.txt', self.coefficients.T)
        else:
            np.savez(path, coeff=self.coefficients.T)
        self.dataframe.to_csv(path2 + '.csv', sep=',', header=True)

    def plot(self, *iters, suffix=''):
        if self.plot_backend == 'mpl':
            plt.style.use('ggplot')
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y, 'k-', label='Real function')
            # ax.set_xticks([n * self.period / 2 for n in np.arange(-self.interval, self.interval, self.period/2)])
            # ax.set_yticks([n * self.amplitude for n in np.arange(-1.5, 1.5, 0.5)])
            if iters:
                for iter in iters:
                    ax.plot(self.x, self.fs[iter, :], self.styles[iter % len(self.styles)], label='Iteration {}'.format(iter + 1),
                            linewidth=0.5)
            else:
                for i in range(self._iterations):
                    label = 'Iteration {}'.format(i + 1)
                    ax.plot(self.x, self.fs[i, :], self.styles[i % len(self.styles)], label=label, linewidth=0.5)
            if self._iterations < 10 or iters and len(iters) < 10:
                leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                leg = None
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            plt.title('Fourier Series approximation of a {} wave'.format(self._type))
            if leg:
                plt.savefig(os.path.join(self.save_path, 'out.fourier_series{}.png'.format(suffix)), bbox_extra_artists=(leg,),
                            bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(self.save_path, self.name + '.out.fourier_series{}.png'.format(suffix)),
                            bbox_inches='tight', dpi=300)
            plt.close()
        elif self.plot_backend == 'plotly':
            if iters:
                querystr = 'iteration==("realfunc",'
                for iter in iters:
                    querystr += '"iter{}",'.format(str(iter))
                querystr += ')'
                df = self.dataframe.query(querystr)
            else:
                df = self.dataframe
            fig = px.line(df, x="x", y="y", color="iteration")
            fig.write_html(os.path.join(self.save_path, self.name + '.out.fourier_series{}.html'.format(suffix)))

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, new_func):
        raise AttributeError("A new function cannot be set once a Fourier Series has been initialized!")

    @func.getter
    def func(self):
        return self._func

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        raise AttributeError("A new type cannot be set once a Fourier Series has been initialized!")

    @type.getter
    def type(self):
        return self._type

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, new_value):
        self.__start_index = self._iterations
        self._iterations = new_value
        self.coefficients = np.concatenate((self.coefficients, np.full(shape=(2, self._iterations - self.coefficients.shape[1]), fill_value=np.nan)), axis=1)
        self.fs = np.concatenate((self.fs, np.full(shape=(self._iterations, self.resolution), fill_value=np.nan)))

    @iterations.getter
    def iterations(self):
        return self._iterations

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, new_value):
        self._interval = new_value
        self.x = np.linspace(self._interval[0], self._interval[1], self.resolution)
        self.y = self.evaluate_func()
        self.fs = self.fourierterms.cumsum(self.x)

    @interval.getter
    def interval(self):
        return self._interval

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_value):
        self._amplitude = new_value
        if self.type == 'custom':
            custom_func = self._func
        else:
            custom_func = None
        self._func = self._initialize_func(self.type, custom_func)
        self.__start_index = 0

    @amplitude.getter
    def amplitude(self):
        return self._amplitude

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, new_value):
        self._period = new_value
        if self.type == 'custom':
            custom_func = self._func
        else:
            custom_func = None
        self._func = self._initialize_func(self.type, custom_func)
        self.__start_index = 0

    @period.getter
    def period(self):
        return self._period

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_value):
        self._phase = new_value
        if self.type == 'custom':
            custom_func = self._func
        else:
            custom_func = None
        self._func = self._initialize_func(self.type, custom_func)
        self.__start_index = 0

    @phase.getter
    def phase(self):
        return self._phase

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, new_value):
        self._resolution = new_value
        self.x = np.linspace(self._interval[0], self._interval[1], self._resolution)
        self.y = np.full(shape=(len(self.x),), fill_value=np.nan)
        self.y = self.evaluate_func()
        self.fs = np.full(shape=(self._iterations, self._resolution), fill_value=np.nan)
        self.fs = self.fourierterms.cumsum(self.x)

    @classmethod
    def square(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'square', save_path, name='SquareWave')

    @classmethod
    def squarerect(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'square_rect', save_path, name='RectifiedSquareWave')

    @classmethod
    def triangle(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'triangle', save_path, name='TriangleWave')

    @classmethod
    def tranglerect(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'triangle_rect', save_path, name='RectifiedTriangleWave')

    @classmethod
    def sawtooth(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'sawtooth', save_path, name='SawtoothWave')

    @classmethod
    def sawtoothrect(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'sawtooth_rect', save_path, name='RectifiedSawtoothWave')

    @classmethod
    def sin(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'sin', save_path, name='SineWave')

    @classmethod
    def cos(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'cos', save_path, name='CosineWave')

    @classmethod
    def gaussian(cls, save_path):
        return cls(50, (-1.5, 1.5), 1, 2*np.pi, 0, 'gaussian', save_path, name='Gaussian')
