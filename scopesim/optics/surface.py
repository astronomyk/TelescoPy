import os
import warnings

import numpy as np

from astropy import units as u
from astropy.io import ascii as ioascii
from astropy.table import Table

from synphot import SpectralElement
from synphot.models import Empirical1D

from ..effects import ter_curves_utils as ter_utils
from ..utils import get_meta_quantity, quantify, extract_type_from_unit
from ..utils import from_currsys, convert_table_comments_to_dict, find_file
from .surface_utils import make_emission_from_emissivity,\
    make_emission_from_array


class PoorMansSurface:
    """ Solely used by SurfaceList """
    def __init__(self, emission, throughput, meta):
        self.emission = emission
        self.throughput = throughput
        self.meta = meta


class SpectralSurface:
    """
    Initialised by a file containing one or more of the following columns:
    transmission, emissivity, reflection. The column wavelength must be given.
    Alternatively kwargs for the above mentioned quantities can be passed as
    arrays. If they are not Quantities, then a unit should also be passed with
    the <array_name>_unit syntax (i.e. emission_unit or wavelength_unit)

    If temperature is not given as a Quantity, it defaults to degrees Celsius.

    """
    def __init__(self, filename=None, **kwargs):
        filename = find_file(filename)
        self.meta = {"filename"         : filename,
                     "temperature"      : -270*u.deg_C,  # deg C
                     "emission_unit"    : "",
                     "wavelength_unit"  : u.um}

        self.table = Table()
        if filename is not None and os.path.exists(filename):
            self.table = ioascii.read(filename)
            tbl_meta = convert_table_comments_to_dict(self.table)
            if isinstance(tbl_meta, dict):
                self.meta.update(tbl_meta)

        self.meta.update(kwargs)

    @property
    def area(self):
        if "area" in self.meta:
            the_area = self.from_meta("area", u.m**2)
        elif "outer" in self.meta:
            outer_diameter = self.from_meta("outer", u.m)
            the_area = np.pi * (0.5 * outer_diameter)**2
            if "inner" in self.meta:
                inner_diameter = self.from_meta("inner", u.m)
                the_area -= np.pi * (0.5 * inner_diameter) ** 2
        else:
            the_area = None

        return the_area

    @property
    def mirror_angle(self):
        if "angle" in self.meta:
            mirr_angle = self.from_meta("angle", u.deg)
        else:
            mirr_angle = 0 * u.deg
        return mirr_angle

    @property
    def wavelength(self):
        return self._get_array("wavelength")

    @property
    def throughput(self):
        return self._get_ter_property(self.meta.get("action", "transmission"))

    @property
    def transmission(self):
        return self._get_ter_property("transmission")

    @property
    def emissivity(self):
        return self._get_ter_property("emissivity")

    @property
    def reflection(self):
        return self._get_ter_property("reflection")

    @property
    def emission(self):
        """
        Looks for an emission array in self.meta. If it doesn't find this, it
        defaults to creating a blackbody and multiplies this by the emissivity.
        Assumption is that self.meta["temperature"] is in deg_C, unless it is
        a u.Quantity with attached temperature unit.
        Return units are in PHOTLAM arcsec^-2, even though arcsec^-2 is not
        given
        """

        flux = self._get_array("emission")
        if flux is not None:
            wave = self._get_array("wavelength")
            flux = make_emission_from_array(flux, wave, meta=self.meta)
        elif "temperature" in self.meta:
            emiss = self.emissivity                     # SpectralElement [0..1]
            temp = from_currsys(self.meta["temperature"])
            if not isinstance(temp, u.Quantity):
                temp = quantify(temp, u.deg_C)
            temp = temp.to(u.Kelvin, equivalencies=u.temperature())
            flux = make_emission_from_emissivity(temp, emiss)
            flux.meta["temperature"] = temp
        else:
            flux = None

        has_solid_angle = extract_type_from_unit(flux.meta["solid_angle"],
                                                 "solid angle")[1] != u.Unit("")
        if flux is not None and has_solid_angle:
            conversion_factor = flux.meta["solid_angle"].to(u.arcsec ** -2)
            flux = flux * conversion_factor
            flux.meta["solid_angle"] = u.arcsec**-2
            flux.meta["history"] += ["Converted to arcsec-2: {}"
                                     "".format(conversion_factor)]

        if flux is not None and "rescale_emission" in self.meta:
            dic = from_currsys(self.meta["rescale_emission"])
            amplitude = dic["value"] * u.Unit(dic["unit"])
            filter_name = dic["filter_name"]
            if "filename_format" in dic:
                filter_name = dic["filename_format"].format(filter_name)
            flux = ter_utils.scale_spectrum(flux, filter_name, amplitude)

        return flux

    def from_meta(self, key, default_unit=None):
        """
        Converts a specific value in the meta dict to a ``Quantity``

        Parameters
        ----------
        key : str
            Which key to pull from self.meta
        default_unit : str, Unit
            In case self.meta doesn't contain a unit for the desired key

        Returns
        -------
        meta_quantity : Quantity

        """

        if default_unit is None:
            default_unit = ""
        meta_quantity = get_meta_quantity(self.meta, key, u.Unit(default_unit))

        return meta_quantity

    def _get_ter_property(self, ter_property, fmt="synphot"):
        """
        Looks for arrays for transmission, emissivity, or reflection

        Parameters
        ----------
        ter_property : str
            ``transmission``, ``emissivity``, ``reflection``

        Returns
        -------
            response_curve : ``synphot.SpectralElement``

        """

        compliment_names = ["transmission", "emissivity", "reflection"]
        ii = np.where([ter_property == name for name in compliment_names])[0][0]
        compliment_names.pop(ii)

        wave = self._get_array("wavelength")
        value_arr = self._get_array(ter_property)
        if value_arr is None:
            value_arr = self._compliment_array(*compliment_names)

        if value_arr is not None and wave is not None and fmt == "synphot":
            response_curve = SpectralElement(Empirical1D, points=wave,
                                             lookup_table=value_arr)
        elif fmt == "array":
            response_curve = value_arr
        else:
            response_curve = None
            warnings.warn("Both wavelength and {} must be set"
                          "".format(ter_property))

        return response_curve

    def _compliment_array(self, colname_a, colname_b):
        """
        Returns an complimentary array using: ``a + b + c = 1``

        E.g. ``Emissivity = 1 - (Transmission + Reflection)``

        Parameters
        ----------
        colname_a : str
            Name of the first TER property to look for
        colname_b
            Name of the second TER property to look for

        Returns
        -------
        actual : ``synphot.SpectralElement``
            Complimentary spectrum to those given

        """

        compliment_a = self._get_array(colname_a)
        compliment_b = self._get_array(colname_b)

        if compliment_a is not None and compliment_b is not None:
            actual = 1*compliment_a.unit - (compliment_a + compliment_b)
        elif compliment_a is not None and compliment_b is None:
            actual = 1*compliment_a.unit - compliment_a
        elif compliment_b is not None and compliment_a is None:
            actual = 1*compliment_b.unit - compliment_b
        elif compliment_b is None and compliment_a is None:
            actual = None

        return actual

    def _get_array(self, colname):
        """
        Looks for an array in either the self.meta or self.table attributes

        Order of search goes: 1. self.meta, 2. self.table

        Parameters
        ----------
        colname : str
            Array column (or key) name

        Returns
        -------
            val_out : array-like Quantity

        """

        if colname in self.meta:
            val = self.meta[colname]
        elif colname in self.table.colnames:
            val = self.table[colname].data
        else:
            warnings.warn("{} not found in either '.meta' or '.table'"
                          "".format(colname))
            return None

        col_units = colname+"_unit"
        if isinstance(val, u.Quantity):
            units = val.unit
        elif col_units in self.meta:
            units = u.Unit(self.meta[col_units])
        else:
            units = u.Unit("")

        if isinstance(val, u.Quantity):
            val_out = val.to(units)
        elif isinstance(val, (list, tuple, np.ndarray)):
            val_out = val * units
        elif val is None:
            val_out = None
        else:
            raise ValueError("{} must be of type: Quantity, array, list, tuple"
                             "".format(colname))

        return val_out

    def __repr__(self):
        meta = self.meta
        name = meta["name"] if "name" in meta else meta["filename"]
        cols = "".join([col[0].upper() for col in self.table.colnames])
        msg = '<SpectralSurface> [{}] "{}"'.format(cols, name)

        return msg
