import warnings

import numpy as np

import astropy.constants as const
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits

import synphot


class Cube:
    """
    class to hold the information of datacubes and eventually manipulate them
    it receives a datacube and converts it to a format understood by scopesim
    (photons/um/s/m2).

    If a file is provided, it will not retrieve the data until needed making all
    operations using the information in the headers.

    Usage
    -----

    cube = Cube(filename)

    cube.header contains the new header
    cube.wcs contains the wcs system
    cube.waves contains the the wavelength vector
    cube.data contains the data in photons/um/s/m2


    """

    def __init__(self, filename=None, hdu=None, ext=0):

        self.filename = filename
        self.hdu = hdu
        self.ext = ext

        if self.filename is not None:
            self._in_header = fits.getheader(self.filename, ext=self.ext)
        elif isinstance(hdu, fits.ImageHDU):
            self._in_header = hdu.header
        elif isinstance(hdu, fits.HDUList):
            self._in_header = hdu[ext].header
        else:
            raise ValueError("No cube has been defined")

        if 'BUNIT' in self._in_header:
            try:
                u.Unit(self._in_header["BUNIT"])
                self.bunit = self._in_header["BUNIT"]
            except ValueError:
                self.bunit = "erg / (s um cm2)"
                warnings.warn("Keyword 'BUNIT' not found, setting to %s by default" % self.bunit)
        else:
            self.bunit = "erg / (s um cm2)"
            warnings.warn("Keyword 'BUNIT' not found, setting to %s by default" % self.bunit)

        if self._in_header["CTYPE3"].lower() not in ["freq", 'wave', "awav", 'wavelength', "vel"]:
            self._in_header.update({"CTYPE3": "WAVE"})

        self._in_wcs = WCS(self._in_header)

        self._make_header()
        self.wcs = WCS(self.header)

    def _make_header(self):
        """
        Making a new header. Celestial header is untouched
        """

        self.header = fits.Header(dict(SIMPLE='T', NAXIS=3, EXTEND='T', BUNIT=self.bunit))
        self.header.update(self._in_wcs.to_header())

        if self.header["CTYPE3"].lower() in ['wave', "awav", 'wavelength']:
            self._convert_from_wave()
        if self.header["CTYPE3"].lower() in ['freq']:
            self._convert_from_freq()
        if self.header["CTYPE3"].lower() in ["vel"]:
            self._convert_from_vel()

    @property
    def _in_waves(self):
        """
        The original waveset of the datacube
        """
        specwcs = self._in_wcs.spectral
        zpix = np.arange(specwcs.spectral.array_shape[0])
        wave_unit = u.Unit(self._in_header["CUNIT3"])
        waves = specwcs.pixel_to_world(zpix).to(wave_unit)

        return waves

    @property
    def waves(self):
        """
        The waveset of the transformed cube
        """
        specwcs = self.wcs.spectral
        zpix = np.arange(specwcs.spectral.array_shape[0])
        waves = specwcs.pixel_to_world(zpix).to(u.um)

        return waves

    @property
    def data(self):

        if self.filename is not None:
            data = fits.get_data(self.filename, ext=self.ext)
        elif isinstance(self.hdu, fits.ImageHDU):
            data = self.hdu.data
        elif isinstance(self.hdu, fits.HDUList):
            data = self.hdu[self.ext].data
        else:
            raise ValueError("No cube has been defined")

        if self._in_header["CTYPE3"].lower() in ['wave', "awav", 'wavelength']:
            data = data * self._convert_from_wave()

        if self._in_header["CTYPE3"].lower() in ['freq']:
            data = data * self._convert_from_freq()

        if self._in_header["CTYPE3"].lower() in ["vel"]:
            data = data * self._convert_from_vel()

        return data

    def _convert_from_wave(self):
        """
        Returns a scaling array and update the output header in the process
        """

        specwcs = self._in_wcs.spectral
        temp_flux = np.ones(self._in_waves.shape[0]) * u.Unit(self._in_header["BUNIT"])
        conversion_factor = convert_flux_units(temp_flux, self._in_waves)
        conversion_factor = conversion_factor.value.reshape(self._in_waves.shape[0], 1, 1)

        self.header = self._in_header.copy()
        self.bunit = "ph / (m2 um s)"
        cdelt = specwcs.wcs.cdelt[0] * specwcs.wcs.cunit[0]
        cdelt = cdelt.to(u.um, equivalencies=u.spectral()).value
        cunit = specwcs.wcs.cunit
        crval = specwcs.wcs.crval * cunit

        self.header.update(dict(BUNIT=self.bunit,
                                CUNIT3="um",
                                CRVAL3=crval[0].to(u.um).value))

        #waves = self._in_waves.to(u.um, equivalencies=u.spectral()).value
        if 'CDELT3' in self.header:
            self.header.update(dict(CDELT3=cdelt))
        if 'CD3_3' in self.header:
            self.header.update(dict(CD3_3=cdelt))

        return conversion_factor

    def _convert_from_freq(self):
        specwcs = self._in_wcs.spectral
        temp_flux = np.ones(self._in_waves.shape[0]) * self.bunit
        conversion_factor = convert_flux_units(temp_flux, self._in_waves)

        freq_0 = self._in_waves[0].value
        c = const.c.to(u.um / u.s).value
        cdelt = specwcs.wcs.cdelt[0] * -c / freq_0**2

        self.bunit = "ph / (m2 um s)"
        self.header.update(dict(CTYPE3='WAVE-F2W',
                                CRVAL3=c / specwcs.wcs.crval[0],  # um
                                CRPIX3=specwcs.wcs.crpix[0],
                                CUNIT3='um',
                                BUNIT=self.bunit))

        if 'CDELT3' in self.header:
            self.header.update(dict(CDELT3=cdelt))
        if 'CD3_3' in self.header:
            self.header.update(dict(CD3_3=cdelt))

        return conversion_factor.value.reshape(self._in_waves.shape[0], 1, 1)

    def _convert_from_vel(self):
        return NotImplementedError

    def get_hdu(self, **kwargs):
        hdu = fits.ImageHDU(data=self.data, header=self.header, **kwargs)
        return hdu

    def write_to(self, filename, **kwargs):
        hdu = self.get_hdu()
        hdu.writeto(filename, **kwargs)

    # Methods that might be useful
    def get_slice(self, wave=None, index=None):
        specwcs = self.wcs.spectral
        if wave is not None:
            index = specwcs.world_to_pixel([wave], 0)[0]
            index = round(index[0])

        if (index < 0) or index > specwcs.pixel_shape[0]:
            raise ValueError("index or wave out of range")

        slice_data = self.data[index]
        slice_header = self.wcs.celestial.to_header()

        return fits.ImageHDU(data=slice_data, header=slice_header)

    def photons_in_range(self, wmin=None, wmax=None):
        specwcs = self.wcs.spectral

        if wmin is not None:
            start = round(specwcs.world_to_pixel([wmin], 0)[0][0])
        else:
            start = 0
        if wmax is not None:
            end = round(specwcs.world_to_pixel([wmax], 0)[0][0]) + 1
        else:
            end = specwcs.pixel_shape[0]

        flux = np.sum(self.data[start:end], axis=(1, 2))

        waves = self.waves[start:end]
        print(waves)
        nphot = np.trapz(flux, waves.value)

        return nphot

    def sum(self, wmin=None, wmax=None):
        """
        Returns an image (ImageHDU) with the summed data.

        It should return the correct number of photons per pixel

        Parameters
        ----------
        wmin
        wmax

        Returns
        -------
        fits.ImageHDU

        """
        specwcs = self.wcs.spectral

        if wmin is not None:
            start = round(specwcs.world_to_pixel([wmin], 0)[0][0])
        else:
            start = 0
        if wmax is not None:
            end = round(specwcs.world_to_pixel([wmax], 0)[0][0]) + 1
        else:
            end = specwcs.pixel_shape[0]

        sum_data = np.sum(self.data[start:end], axis=0)
        sum_data = self.photons_in_range(wmin=wmin, wmax=wmax) * sum_data/np.sum(sum_data)
        sum_header = self.wcs.celestial.to_header()

        return fits.ImageHDU(data=sum_data, header=sum_header)

    def collapse(self):
        """
        Create a summed synphot.SourceSpectrom from the summed datacube

        TODO: from only a section
        Returns
        -------
        synphot.SourceSpectrum
        """
        specwcs = self.wcs.spectral
        start = 0
        end = specwcs.pixel_shape[0]
        flux = np.sum(self.data[start:end], axis=(1, 2))
        sp = synphot.SourceSpectrum(synphot.Empirical1D, points=self.waves, lookup_table=flux)

        return sp


def convert_flux_units(influx, waves=None):
    """
    --- Copied from SpeCADO ---

    Convert flux units
    "Flux units" refers to both integrated fluxes for point sources
    and to surface brightnesses.
    The internal units are:
    - Flux:   ph / (m2 um s)
    - surface flux:  ph / (m2 um s arcsec2)
    Parameters
    ----------
    influx : list of astropy.unit.Quantity
        These can be energy or photon fluxes
    waves : float, nd.array
        Wavelengths at which influx is given. Default is None, which
        is okay if the conversion is independent of wavelength.
    """
    if waves is not None:
        useequivalencies = u.spectral_density(waves)
    else:
        useequivalencies = None

    # Check whether we have a surface brightness
    inunit = influx.unit
    factor = 1
    for un, power in zip(inunit.bases, inunit.powers):
        if un.is_equivalent(u.arcsec):
            conversion = (un.to(u.arcsec) / un) ** power
            influx *= conversion
            factor = u.arcsec ** (-2)

    outflux = influx.to(u.ph / u.m ** 2 / u.um / u.s,
                        useequivalencies)

    return outflux * factor
