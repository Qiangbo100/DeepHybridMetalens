import numpy as np


def wavelength_to_rgb(wavelength, gamma=0.8):
    """Convert a wavelength in the range of 380-780 nm to an RGB color."""
    wavelength = float(wavelength)
    if wavelength < 380 or wavelength > 780:
        return (0, 0, 0)  # Wavelength is outside the visible range.

    def adjust(color, factor):
        if color == 0.0:
            return 0
        else:
            return round(255 * (color * factor) ** gamma)

    if wavelength < 440:
        R = -(wavelength - 440.) / (440. - 380.)
        G = 0.0
        B = 1.0
    elif wavelength < 490:
        R = 0.0
        G = (wavelength - 440.) / (490. - 440.)
        B = 1.0
    elif wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510.) / (510. - 490.)
    elif wavelength < 580:
        R = (wavelength - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif wavelength < 645:
        R = 1.0
        G = -(wavelength - 645.) / (645. - 580.)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0

    # Adjust colors to gamma
    R = adjust(R, 1.0)
    G = adjust(G, 1.0)
    B = adjust(B, 1.0)

    return (R, G, B)
