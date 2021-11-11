import numpy as np
import os
import streamlit as st
import re

import umbrella.waves as w
from umbrella.fourier import FourierSeries
from umbrella import poly

path = os.path.abspath(os.path.dirname(__file__)) + '/resources/'


def build_fourier(type, rectify, amp, period, phase, velocity,
                    timestamp, iterations, interval, resolution):
    if rectify and type not in ('Sine', 'Cosine', 'Gaussian'):
        type += 'Rect'
    function = eval(f'w.{type}({amp}, {period}, {phase}, {velocity})')
    fs = FourierSeries(function)
    _, _, _, _, pchart = fs.integrate(timestamp, iterations, interval, resolution)
    return pchart


def build_hydrogen(n, l, m, d, type):
    if type == "3D":
        _, _, _, _, fig = poly.plot_hydrogen_3D(int(n), int(l), int(m), slice=d)
    elif type == "2D":
        _, _, _, _, fig = poly.plot_hydrogen(int(n), int(l), int(m))
    return fig


def build_legendre(l, m, type):
    if type == "Cartesian (x)":
        _, _, fig = poly.plot_legendre(int(m), l)
    elif type == "Polar (cos x)":
        _, _, fig = poly.polar_plot_legendre(int(m), l)
    return fig


def build_bessel(l, type):
    _, _, fig = poly.plot_bessel(l, type)
    return fig


def build_hermite(n):
    _, _, fig = poly.plot_hermite(n)
    return fig


def build_laguerre(q, p):
    _, _, fig = poly.plot_laguerre(p, q)
    return fig


def build_harmonic(l, m):
    _, _, _, fig = poly.plot_harmonics(m, l)
    return fig


def fourier_frame():
    with st.expander("Create a Fourier Series"):
        # st.markdown('# Create a Fourier Series:')
        st.latex(r'\hat{f}(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}a_n\cos\bigg(\frac{2\pi n}{P}x\bigg) + '
                 r'\sum_{n=1}^{\infty}b_n\sin\bigg(\frac{2\pi n}{P}x\bigg)')
        selection_list = ['Sine', 'Cosine', 'Square', 'Triangle', 'Sawtooth', 'Gaussian']

        st.markdown('## Wave Options')
        type = st.selectbox('Wave Type', selection_list, 0)
        amp = st.slider('Amplitude', 0., 100., 1., 0.1, "%.1f")
        per = st.slider('Period (or Gaussian FWHM)', 0.01, 100., 2*np.pi, 0.01, "%.2f")
        phase = st.slider('Phase (or Gaussian Mean)', -np.pi, np.pi, 0., 0.01, "%.2f")
        velocity = st.number_input('Velocity (v)', int(0), int(100), int(0), int(1), "%d")
        rectify = st.checkbox('Rectify', False)

        st.markdown('## Fourier Series Options')
        ts = st.number_input('Draw Timestamp', 0., 100., 0., 0.1, "%.1f")
        iters = st.number_input('Iterations', int(1), int(100), int(10), int(1), "%d")
        cols = st.columns(2)
        min = cols[0].number_input(label='Interval Min', value=-np.pi)
        max = cols[1].number_input(label='Interval Max', value=np.pi)
        interval = (min, max)
        res = int(st.number_input(label='Resolution', value=1000))

        pchart = build_fourier(type, rectify, amp, per, phase, velocity, ts, iters, interval, res)
        st.plotly_chart(pchart, use_container_width=True)


def legendre_frame():
    with st.expander("Look at Associated Legendre Functions"):
        # st.markdown('# Look at Associated Legendre Functions:')
        st.latex(r'P_{\ell}(x) = \frac{1}{2^\ell \ell!}\frac{d^\ell}{dx^\ell}(x^2-1)^\ell')
        st.latex(r'P_{\ell}^m(x) = (-1)^m(1-x^2)^{m/2}\frac{d^m}{dx^m}P_{\ell}(x);\ \ P_{\ell}^{-m}(x) = (-1)^m\frac{(\ell - m)!}{(\ell+m)!}P_{\ell}^{m}(x)')
        st.latex(r'\ell \in \mathbb{N}_0 \ \ \ \ \ m \in \{-\ell,-\ell+1,...,\ell-1, \ell\}')
        st.markdown('## Legendre Options')
        collect_nums = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
        filter_nums = lambda _list: [item for item in _list if (0 <= item <= 100)]
        nums = st.text_input("l [type any number of values to plot]", "0, 1, 2, 3, 4")
        lP = filter_nums(collect_nums(nums))
        mP = st.number_input('m ', int(-min(lP)), int(min(lP)), int(0), int(1), "%d")
        type = st.selectbox("Plot Type", ["Cartesian (x)", "Polar (cos x)"], 0)
        fig = build_legendre(lP, mP, type)
        st.plotly_chart(fig, use_container_width=True)


def bessel_frame():
    with st.expander("Look at Bessel Functions"):
        st.latex(r'j_\ell(x) = (-x)^\ell \bigg(\frac{1}{x}\frac{d}{dx}\bigg)^\ell \frac{\sin(x)}{x}')
        st.latex(r'n_\ell(x) = -(-x)^\ell \bigg(\frac{1}{x}\frac{d}{dx}\bigg)^\ell \frac{\cos(x)}{x}')
        st.latex(r'\ell \in \mathbb{N}_0')
        st.markdown('## Bessel Options')
        collect_nums = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
        filter_nums = lambda _list: [item for item in _list if (0 <= item <= 70)]
        nums = st.text_input("l [type any number of values to plot]  ", "0, 1, 2, 3, 4")
        type = st.selectbox("Plot Type", ["Spherical Bessel", "Spherical Neumann"], 0)
        lB = filter_nums(collect_nums(nums))
        fig = build_bessel(lB, type)
        st.plotly_chart(fig, use_container_width=True)


def hermite_frame():
    with st.expander("Look at Hermite Polynomials"):
        st.latex(r'H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}e^{-x^2}')
        st.latex(r'n \in \mathbb{N}_0')
        st.markdown('## Hermite Options')
        collect_nums = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
        filter_nums = lambda _list: [item for item in _list if (0 <= item <= 100)]
        nums = st.text_input("n [type any number of values to plot]", "0, 1, 2, 3, 4")
        nH = filter_nums(collect_nums(nums))
        fig = build_hermite(nH)
        st.plotly_chart(fig, use_container_width=True)


def laguerre_frame():
    with st.expander("Look at Associated Laguerre Functions"):
        st.latex(r'L_q^p(x) = (-1)^p \frac{d^p}{dx^p}L_{p+q}(x)')
        st.latex(r'L_q(x) = \frac{e^x}{q!}\frac{d^q}{dx^q}(e^{-x}x^q)')
        st.latex(r'q \in \mathbb{N}_0 \ \ \ \ \ p \in \{0,1,...,q\}')
        st.markdown('## Laguerre Options')
        collect_nums = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]
        filter_nums = lambda _list: [item for item in _list if (0 <= item <= 100)]
        nums = st.text_input("q [type any number of values to plot]", "0, 1, 2, 3, 4")
        q = filter_nums(collect_nums(nums))
        p = st.number_input('p', int(0), int(min(q)), int(0), int(1), "%d")
        fig = build_laguerre(q, p)
        st.plotly_chart(fig, use_container_width=True)


def harmonic_frame():
    with st.expander("Look at Spherical Harmonics"):
        st.latex('Y_\ell^m(\\theta, \\phi) = \sqrt{\\frac{(2\ell + 1)}{4\pi}\\frac{(\ell - m)!}{(\ell + m)!}}'
                 'e^{im\phi}P_\ell^m(\cos\\theta)')
        st.latex(r'\ell \in \mathbb{N}_0 \ \ \ \ \ m \in \{-\ell,-\ell+1,...,\ell-1, \ell\}')
        st.markdown('## Harmonic Options')
        lH = st.number_input("l  ", int(0), int(100), int(0), int(1), "%d")
        mH = st.number_input("m  ", int(-lH), int(lH), int(0), int(1), "%d")
        fig = build_harmonic(lH, mH)
        st.plotly_chart(fig, use_container_width=True)


def hydrogen_frame():
    with st.expander("Look at Hydrogen Atom Orbitals"):
        # st.markdown('# Look at Hydrogen Atom Orbitals:')
        st.latex(r'\psi_{n \ell m} = \sqrt{\bigg(\frac{2}{na}\bigg)^3 \frac{(n-\ell-1)!}{2n(n+\ell)!}} '
                 r'e^{-r/na}\bigg(\frac{2r}{na}\bigg)^{\ell}\bigg[L^{2\ell+1}_{n-\ell-1}(2r/na)\bigg]Y_{\ell}^m(\theta, \phi)')
        st.latex(r'n \in \mathbb{Z}^+ \ \ \ \ \ \ell \in \{0,1,...,n-1\} \ \ \ \ \ m \in \{-\ell,-\ell+1,...,\ell-1,\ell\}')
        st.markdown('## Hydrogen Atom Options')
        n = st.number_input('n', int(1), int(100), int(1), int(1), "%d")
        l = st.number_input('l', int(0), int(n-1), int(0), int(1), "%d")
        m = st.number_input('m', int(-l), int(l), int(0), int(1), "%d")
        d = st.slider('Density cutoff (nm^-3)', float(0.), float(5.), float(0.25), float(0.001), "%.3f")
        type = st.selectbox("Plot Type", ["2D", "3D"], 1)
        st.latex('\\big|\\psi_{%d %d %d}\\big|^2' % (n, l, m))
        fig = build_hydrogen(n, l, m, d, type)
        if type == "3D":
            st.plotly_chart(fig, use_container_width=True)
        elif type == "2D":
            st.pyplot(fig)


def main():
    # Create Titles
    st.title('Verbose Umbrella: ')
    fourier_frame()
    legendre_frame()
    bessel_frame()
    hermite_frame()
    laguerre_frame()
    harmonic_frame()
    hydrogen_frame()
    st.write(r'$\mathbb{N}_0 \text{ is shorthand for non-negative integers, i.e. the set } \{0,1,2,...\}$')
    st.write(r'$\mathbb{Z}^+ \text{ is shorthand for positive integers, i.e. the set } \{1,2,...\}$')

if __name__ == '__main__':
    main()