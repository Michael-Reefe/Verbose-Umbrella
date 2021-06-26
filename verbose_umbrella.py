import tkinter as tk
import numpy as np
import os
import streamlit as st

import umbrella.waves as w
from umbrella.fourier import FourierSeries

path = os.path.abspath(os.path.dirname(__file__)) + '/resources/'


def build_fourier(type, rectify, amp, period, phase, velocity,
                    timestamp, iterations, interval, resolution):
    if rectify and type not in ('Sine', 'Cosine', 'Gaussian'):
        type += 'Rect'
    function = eval(f'w.{type}({amp}, {period}, {phase}, {velocity})')
    fs = FourierSeries(function)
    _, _, _, _, pchart = fs.integrate(timestamp, iterations, interval, resolution)
    return pchart


def fourier_frame():
    selection_list = ['Sine', 'Cosine', 'Square', 'Triangle', 'Sawtooth', 'Gaussian']

    st.markdown('## Wave Options')
    type = st.selectbox('Wave Type', selection_list, 0)
    amp = st.slider('Amplitude', 0., 100., 1., 0.1, "%.1f")
    per = st.slider('Period (or Gaussian FWHM)', 0.01, 100., 2*np.pi, 0.01, "%.2f")
    phase = st.slider('Phase (or Gaussian Mean)', -np.pi, np.pi, 0., 0.01, "%.2f")
    velocity = st.number_input('Velocity (v)', 0, 100, 0, 1, "%d")
    rectify = st.checkbox('Rectify', False)

    st.markdown('## Fourier Series Options')
    ts = st.number_input('Draw Timestamp', 0., 100., 0., 0.1, "%.1f")
    iters = st.number_input('Iterations', 1, 100, 10, 1, "%d")
    cols = st.beta_columns(2)
    min = cols[0].number_input(label='Interval Min', value=-np.pi)
    max = cols[1].number_input(label='Interval Max', value=np.pi)
    interval = (min, max)
    res = st.number_input(label='Resolution', value=1000)

    pchart = build_fourier(type, rectify, amp, per, phase, velocity, ts, iters, interval, res)
    st.plotly_chart(pchart, use_container_width=True)


def main():
    # Create Titles
    st.title('Verbose Umbrella: ')
    st.markdown('# Create a Fourier Series:')
    st.latex(r'\hat{f}(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}a_n\cos\bigg(\frac{2\pi n}{P}x\bigg) + '
             r'\sum_{n=1}^{\infty}b_n\sin\bigg(\frac{2\pi n}{P}x\bigg)')
    fourier_frame()


if __name__ == '__main__':
    main()