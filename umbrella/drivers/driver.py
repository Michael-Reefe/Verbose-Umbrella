from umbrella.fourier import FourierSeries
import umbrella.waves as w
import os
import numpy as np

class ComplexWave(w.AbstractWave):

    def __init__(self):
        super().__init__(amplitude=1, period=1, phase=0, velocity=0)
        self.name = 'complex'

    def eval(self, xi, ti=0):
        value = (1+complex('j')*2) + np.exp(complex('j')*2*np.pi*xi) + .5*np.exp(complex('j')*2*2*np.pi*xi) + \
            + .25*np.exp(complex('j')*10*2*np.pi*xi)

        return value.real, value.imag

    def __repr__(self):
        return "A complex waveform"


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    savepath = os.path.abspath(os.path.join(path, '..', 'test'))

    wave = ComplexWave()
    fs = FourierSeries(wave, save_path=savepath)
    fs.complex_animate(iterations=10, interval=(0, 1))
