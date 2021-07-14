import warnings

import numpy as np
from astropy import wcs, units as u
from astropy.io import fits
from astropy.table import Table
from synphot import SourceSpectrum, Empirical1D, SpectralElement

from .. import utils


def validate_source_input(**kwargs):
    if "filename" in kwargs and kwargs["filename"] is not None:
        filename = kwargs["filename"]
        if utils.find_file(filename) is None:
            warnings.warn("filename was not found: {}".format(filename))

    if "image" in kwargs and kwargs["image"] is not None:
        image_hdu = kwargs["image"]
        if not isinstance(image_hdu, (fits.PrimaryHDU, fits.ImageHDU)):
            raise ValueError("image must be fits.HDU object with a WCS."
                             "type(image) == {}".format(type(image_hdu)))

        if len(wcs.find_all_wcs(image_hdu.header)) == 0:
            warnings.warn("image does not contain valid WCS. {}"
                          "".format(wcs.WCS(image_hdu)))

    if "table" in kwargs and kwargs["table"] is not None:
        tbl = kwargs["table"]
        if not isinstance(tbl, Table):
            raise ValueError("table must be an astropy.Table object:"
                             "{}".format(type(tbl)))

        if not np.all([col in tbl.colnames for col in ["x", "y", "ref"]]):
            raise ValueError("table must contain at least column names: "
                             "'x, y, ref': {}".format(tbl.colnames))

    if "cube" in kwargs and kwargs["cube"] is not None:
        cube = kwargs["cube"]
        if isinstance(cube, (fits.PrimaryHDU, fits.ImageHDU, fits.HDUList)) is False:
            try:
                utils.find_file(cube)
            except (AttributeError, ValueError, FileNotFoundError):
                print("cube must be a fits.HDU object or a fits file".format(cube))
                raise ValueError

    return True


def convert_to_list_of_spectra(spectra, lam):
    spectra_list = []
    if isinstance(spectra, SourceSpectrum):
        spectra_list += [spectra]

    elif lam is None and\
            isinstance(spectra, (tuple, list)) and \
            isinstance(spectra[0], SourceSpectrum):
        spectra_list += spectra

    elif lam is not None and len(spectra.shape) == 1 and \
            isinstance(spectra, np.ndarray) and \
            isinstance(lam, np.ndarray):
        spec = SourceSpectrum(Empirical1D, points=lam, lookup_table=spectra)
        spectra_list += [spec]

    elif ((isinstance(spectra, np.ndarray) and
           len(spectra.shape) == 2) or
          (isinstance(spectra, (list, tuple)) and
           isinstance(spectra[0], np.ndarray))) and \
            isinstance(lam, np.ndarray):

        for sp in spectra:
            spec = SourceSpectrum(Empirical1D, points=lam, lookup_table=sp)
            spectra_list += [spec]

    return spectra_list


def photons_in_range(spectra, wave_min, wave_max, area=None, bandpass=None):
    """

    Parameters
    ----------
    spectra
    wave_min
        [um]
    wave_max
        [um]
    area : Quantity
        [m2]
    bandpass : SpectralElement


    Returns
    -------
    counts : u.Quantity array

    """

    if isinstance(wave_min, u.Quantity):
        wave_min = wave_min.to(u.Angstrom).value
    else:
        wave_min *= 1E4

    if isinstance(wave_max, u.Quantity):
        wave_max = wave_max.to(u.Angstrom).value
    else:
        wave_max *= 1E4

    counts = []
    for spec in spectra:
        waveset = spec.waveset.value
        mask = (waveset > wave_min) * (waveset < wave_max)
        x = waveset[mask]
        x = np.append(np.append(wave_min, x), wave_max)
        y = spec(x).value

        # flux [ph s-1 cm-2] == y [ph s-1 cm-2 AA-1] * x [AA]
        if isinstance(bandpass, SpectralElement):
            bp = bandpass(x)
            bandpass.model.bounds_error = True
            counts += [np.trapz(bp * y, x)]
        else:
            counts += [np.trapz(y, x)]

    # counts = flux [ph s-1 cm-2]
    counts = 1E4 * np.array(counts)    # to get from cm-2 to m-2
    counts *= u.ph * u.s**-1 * u.m**-2
    if area is not None:
        counts *= utils.quantify(area, u.m ** 2)

    return counts


def make_imagehdu_from_table(x, y, flux, pix_scale=1*u.arcsec):

    pix_scale = pix_scale.to(u.deg)
    unit = pix_scale.unit
    x = utils.quantify(x, unit)
    y = utils.quantify(y, unit)

    xpixmin = int(np.floor(np.min(x) / pix_scale))
    ypixmin = int(np.floor(np.min(y) / pix_scale))
    xvalmin = (xpixmin * pix_scale).value
    yvalmin = (ypixmin * pix_scale).value

    the_wcs = wcs.WCS(naxis=2)
    the_wcs.wcs.crpix = [0.25, 0.25]
    the_wcs.wcs.cdelt = [pix_scale.value, pix_scale.value]
    the_wcs.wcs.crval = [xvalmin, yvalmin]
    the_wcs.wcs.cunit = [unit, unit]
    the_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    ypix, xpix = the_wcs.wcs_world2pix(y.to(u.deg), x.to(u.deg), 1)
    yint, xint  = ypix.astype(int), xpix.astype(int)

    image = np.zeros((np.max(xint) + 1, np.max(yint) + 1))
    for ii in range(len(xint)):
        image[xint[ii], yint[ii]] += flux[ii]

    hdu = fits.ImageHDU(data=image)
    hdu.header.extend(the_wcs.to_header())

    return hdu


def scale_imagehdu(imagehdu, waverange, area=None):
    # ..todo: implement this
    # For the moment, all imagehdu must be accompanied by a spectrum in PHOTLAM
    #
    # Future functionality will include scaling here of:
    # ph s-1
    # ph s-1 m-2
    # ph s-1 m-2
    # ph s-1 m-2 um-1
    # ph s-1 m-2 um-1 arcsec-2
    # J s-1 m-2 Hz-1
    # J s-1 m-2 Hz-1 arcsec-2
    # ABMAG
    # ABMAG arcsec-2
    # VEGAMAG
    # VEGAMAG arcsec-2

    if "SPEC_REF" not in imagehdu.header:
        raise ValueError("For this version, an ImageHDU must be associated "
                         "with a spectrum. This will change in the future.")

    return imagehdu

#     unit = extract_unit_from_imagehdu(imagehdu)
#
#     per_unit_area = False
#     if area is None:
#         per_unit_area = True
#         area = 1*u.m**2
#
#     unit, sa_unit = utils.extract_type_from_unit(unit, "solid angle")
#     unit = convert_flux(waverange, 1 * unit, "count", area=area)
#     # [ct] comes out of convert_flux
#     unit *= u.s
#
#     if sa_unit != "":
#         cunit1 = u.deg
#         cunit2 = u.deg
#         if "CUNIT1" in imagehdu.header and "CUNIT2" in imagehdu.header:
#             cunit1 = u.Unit(imagehdu.header["CUNIT1"])
#             cunit2 = u.Unit(imagehdu.header["CUNIT2"])
#         cdelt1 = imagehdu.header["CDELT1"] * cunit1
#         cdelt2 = imagehdu.header["CDELT2"] * cunit2
#
#         pix_angle_area = cdelt1 * cdelt2
#         unit *= (pix_angle_area * sa_unit).si.value
#
#     if per_unit_area is True:
#         unit *= u.m**-2
#
#     zero  = 0 * u.Unit(unit)
#     scale = 1 * u.Unit(unit)
#
#     if "BSCALE" in imagehdu.header:
#         scale *= imagehdu.header["BSCALE"]
#         imagehdu.header["BSCALE"] = 1
#     if "BZERO" in imagehdu.header:
#         zero = imagehdu.header["BZERO"]
#         imagehdu.header["BZERO"] = 0
#
#     imagehdu.data = imagehdu * scale + zero
#     imagehdu.header["BUNIT"] = str(imagehdu.data.unit)
#     imagehdu.header["FLUXUNIT"] = str(imagehdu.data.unit)
#
#     return imagehdu
#
# def extract_unit_from_imagehdu(imagehdu):
#     if "BUNIT" in imagehdu.header:
#         unit = u.Unit(imagehdu.header["BUNIT"])
#     elif "FLUXUNIT" in imagehdu.header:
#         unit = u.Unit(imagehdu.header["FLUXUNIT"])
#     elif isinstance(imagehdu.data, u.Quantity):
#         unit = imagehdu.data.unit
#         imagehdu.data = imagehdu.data.value
#     else:
#         unit = ""
#         warnings.warn("No flux unit found on ImageHDU. Please add BUNIT or "
#                       "FLUXUNIT to the header.")
#
#     return unit

