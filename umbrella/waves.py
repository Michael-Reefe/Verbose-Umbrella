from abc import ABC, abstractmethod
import numpy as np


class AbstractWave(ABC):

    @abstractmethod
    def __init__(self, amplitude, period, phase, velocity):
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.velocity = velocity

    @abstractmethod
    def function(self, xi, ti):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class Sine(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        return self.amplitude * np.sin(2*np.pi*(xi - self.velocity*ti)/self.period - self.phase)

    def __repr__(self):
        return "{:.1f}sin(2pi * (x - {:.1f}t)/{:.1f} - {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class Cosine(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        return self.amplitude * np.cos(2 * np.pi * (xi - self.velocity*ti)/ self.period - self.phase)

    def __repr__(self):
        return "{:.1f}cos(2pi * (x - {:.1f}t)/{:.1f} - {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class Square(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period / 2)
        return self.amplitude if factor % 2 == 0 else -self.amplitude

    def __repr__(self):
        return "Square wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class SquareRect(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period/2)
        return self.amplitude if factor % 2 == 0 else 0

    def __repr__(self):
        return "Rectified Square wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class Triangle(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period / 4)
        return -self.amplitude * (2*((xi - self.velocity*ti) - self.phase) - self.period/2 - 2*(self.period/4)*factor) if factor % 4 == 1 else \
            -self.amplitude * (2*((xi - self.velocity*ti) - self.phase) - 2*(self.period/4)*factor) if factor % 4 == 2 else \
            self.amplitude * (2*((xi - self.velocity*ti) - self.phase) - self.period/2 - 2*(self.period/4)*factor) if factor % 4 == 3 else \
            self.amplitude * (2*((xi - self.velocity*ti) - self.phase) - 2*(self.period / 4)*factor)

    def __repr__(self):
        return "Triangle wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class TriangleRect(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period / 4)
        return -self.amplitude * (2 * ((xi - self.velocity*ti) - self.phase) - self.period / 2 - 2 * (self.period / 4) * factor) if factor % 4 == 1 else \
            0 if (factor % 4 == 2) or (factor % 4 == 3) else \
            self.amplitude * (2 * ((xi - self.velocity*ti) - self.phase) - 2 * (self.period / 4) * factor)

    def __repr__(self):
        return "Rectified Triangle wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class Sawtooth(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period/2)
        return self.amplitude * ((xi - self.velocity*ti) - self.phase - (self.period/2)*factor) if factor % 2 == 0 else \
            -self.amplitude * (-(xi - self.velocity*ti) + self.phase + (self.period/2)*factor)

    def __repr__(self):
        return "Sawtooth wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class SawtoothRect(AbstractWave):

    def __init__(self, amplitude, period, phase, velocity):
        super().__init__(amplitude, period, phase, velocity)

    def function(self, xi, ti):
        factor = ((xi - self.velocity*ti) - self.phase) // (self.period/2)
        return self.amplitude * ((xi - self.velocity*ti) - self.phase - (self.period/2)*factor) if factor % 2 == 0 else 0

    def __repr__(self):
        return "Rectified Sawtooth wave with amplitude {:.1f}, velocity {:.1f}, period {:.1f}, and phase {:.1f}".format(self.amplitude, self.velocity, self.period, self.phase)


class Gaussian(AbstractWave):

    def __init__(self, amplitude, fwhm, mean, velocity):
        super().__init__(amplitude, fwhm, mean, velocity)
        self.fwhm = fwhm
        self.mean = mean

    def function(self, xi, ti):
        return self.amplitude*np.exp(-(xi - self.mean - self.velocity*ti)**2/(2*self.fwhm**2))

    def __repr__(self):
        return "Gaussian wave with amplitude {:.1f}, velocity {:.1f}, fwhm {:.1f}, and mean {:.1f}".format(self.amplitude,
                                                                                                            self.velocity,
                                                                                                            self.fwhm,
                                                                                                            self.mean)