import warnings

import numpy as np
from astropy import units as u
from synphot import SourceSpectrum, BlackBody1D, Empirical1D

from ..utils import quantify, extract_type_from_unit, extract_base_from_unit


def make_emission_from_emissivity(temp, emiss_src_spec):
    """
    Create an emission SourceSpectrum using a blackbody and an emissivity curve

    Parameters
    ----------
    temp : float, Quantity
        [Kelvin] If float, then must be in Kelvin
    emiss_src_spec : synphot.SpectralElement
        An emissivity response curve in the range [0..1]

    Returns
    -------
    flux : synphot.SourceSpectrum

    """

    if isinstance(temp, u.Quantity):
        temp = temp.to(u.Kelvin, equivalencies=u.temperature()).value

    if emiss_src_spec is None:
        warnings.warn("Either emission or emissivity must be set")
        flux = None
    else:
        flux = SourceSpectrum(BlackBody1D, temperature=temp)
        flux.meta["solid_angle"] = u.sr**-1
        flux = flux * emiss_src_spec
        flux.meta["history"] = ["Created from Blackbody curve. Units are to be"
                                " understood as per steradian"]

    return flux


def make_emission_from_array(flux, wave, meta):
    """
    Create an emission SourceSpectrum using array.

    Takes care of bins and solid angles. The solid_angle is kept in the returned
    SourceSpectrum meta dictionary under self.meta["solid_angle"]

    Parameters
    ----------
    flux : array-like, Quantity
        if flux is not an array, the ``emission_unit`` must be in meta dict
    wave : array-like, Quantity
        if flux is not an array, the ``wavelength_unit`` must be in meta dict
    meta : dict

    Returns
    -------
    flux : synphot.SourceSpectrum

    """

    if not isinstance(flux, u.Quantity):
        if "emission_unit" in meta:
            flux = quantify(flux, meta["emission_unit"])
        else:
            warnings.warn("emission_unit must be set in self.meta, "
                          "or emission must be an astropy.Quantity")
            flux = None

    if isinstance(wave, u.Quantity) and isinstance(flux, u.Quantity):
        flux_unit, angle = extract_type_from_unit(flux.unit, "solid angle")
        flux = flux / angle

        if is_flux_binned(flux.unit):
            flux = normalise_binned_flux(flux, wave)

        orig_unit = flux.unit
        flux = SourceSpectrum(Empirical1D, points=wave,
                              lookup_table=flux)
        flux.meta["solid_angle"] = angle
        flux.meta["history"] = ["Created from emission array with units {}"
                                "".format(orig_unit)]
    else:
        warnings.warn("wavelength and emission must be "
                      "astropy.Quantity py_objects")
        flux = None

    return flux


def normalise_binned_flux(flux, wave):
    """
    Convert a binned flux Quantity array back into flux density

    The flux density normalising unit is taken from the wavelength Quantity unit

    Parameters
    ----------
    flux : array-like Quantity
        flux unit must include ``bin``, e.g. ``ph s-1 m-2 bin-1``
    wave : array-like Quantity

    Returns
    -------
    flux : array-like Quantity

    """

    bins = np.zeros(len(wave)) * wave.unit
    bins[:-1] = 0.5 * np.diff(wave)
    bins[1:] += 0.5 * np.diff(wave)
    # bins[0] *= 2.   # edge bins only have half the flux of other bins
    # bins[-1] *= 2.

    bin_unit = extract_base_from_unit(flux.unit, u.bin)[1]
    flux = flux / bins / bin_unit

    return flux


def is_flux_binned(unit):
    """
    Checks if the (flux) unit is a binned unit

    Parameters
    ----------
    unit : Unit

    Returns
    -------
    flag : bool

    """
    unit = unit**1
    flag = False
    if u.bin in unit._bases or "flux density" not in unit.physical_type:
        flag = True

    return flag
