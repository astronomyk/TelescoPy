import numpy as np
from astropy import units as u, wcs
from astropy.io import fits
from astropy.table import Table
from synphot import SourceSpectrum, Empirical1D

from scopesim.source.source import Source
from scopesim.source.source_templates import vega_spectrum


def _table_source():
    n = 100
    unit = u.Unit("ph s-1 m-2 um-1")
    wave = np.linspace(0.5, 2.5, n) * u.um
    specs = [SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=4 * np.ones(n) * unit),
             SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.linspace(0, 4, n) * unit),
             SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.linspace(0, 4, n)[::-1] * unit)]
    tbl = Table(names=["x", "y", "ref", "weight"],
                data=[[5,  0, -5,  0]*u.arcsec,
                      [5, -10, 5,  0] * u.arcsec,
                      [2,  0,  1,  0],
                      [1,  1,  1,  2]])
    tbl_source = Source(table=tbl, spectra=specs)

    return tbl_source


def _image_source(dx=0, dy=0, angle=0, weight=1):
    """
    An image with 3 point sources on a random BG

    Parameters
    ----------
    dx, dy : float
        [arcsec] Offset from optical axis
    angle : float
        [deg]
    weight : float

    Returns
    -------

    """
    n = 50
    unit = u.Unit("ph s-1 m-2 um-1")
    wave = np.linspace(0.5, 2.5, n) * u.um
    specs = [SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.linspace(0, 4, n) * unit)]

    n = 50
    im_wcs = wcs.WCS(naxis=2)
    im_wcs.wcs.cunit = [u.arcsec, u.arcsec]
    im_wcs.wcs.cdelt = [0.2, 0.2]
    im_wcs.wcs.crval = [0, 0]
    im_wcs.wcs.crpix = [n//2, n//2]
    im_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    im = np.random.random(size=(n+1, n+1)) * 1e-9 * weight
    im[n-1, 1] += 5 * weight
    im[1, 1] += 5 * weight
    im[n//2, n//2] += 10 * weight
    im[n//2, n-1] += 5 * weight

    im_hdu = fits.ImageHDU(data=im, header=im_wcs.to_header())
    im_hdu.header["SPEC_REF"] = 0
    im_source = Source(image_hdu=im_hdu, spectra=specs)

    angle = angle * np.pi / 180
    im_source.fields[0].header["CRVAL1"] += dx / 3600
    im_source.fields[0].header["CRVAL2"] += dy / 3600
    im_source.fields[0].header["PC1_1"] = np.cos(angle)
    im_source.fields[0].header["PC1_2"] = np.sin(angle)
    im_source.fields[0].header["PC2_1"] = -np.sin(angle)
    im_source.fields[0].header["PC2_2"] = np.cos(angle)

    return im_source


def _combined_source(im_angle=0, dx=[0, 0, 0], dy=[0, 0, 0], weight=[1, 1, 1]):
    tblsrc1 = _table_source()

    tblsrc2 = _table_source()
    tblsrc2.fields[0]["x"] += dx[0]
    tblsrc2.fields[0]["y"] += dy[0]
    tblsrc2.fields[0]["weight"] *= weight[0]

    tblsrc3 = _table_source()
    tblsrc3.fields[0]["x"] += dx[1]
    tblsrc3.fields[0]["y"] += dy[1]
    tblsrc3.fields[0]["weight"] *= weight[1]

    imsrc = _image_source(dx[2], dy[2], im_angle, weight[2])

    src = tblsrc1 + tblsrc2 + tblsrc3 + imsrc

    return src


def _single_table_source(weight=1, n=3):
    unit = u.Unit("ph s-1 m-2 um-1")
    wave = np.linspace(0.5, 2.5, n) * u.um
    specs = [SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.ones(n) * weight * unit)]
    tbl = Table(names=["x", "y", "ref", "weight"],
                data=[[0]*u.arcsec, [0]*u.arcsec, [0], [1]])
    tbl_source = Source(table=tbl, spectra=specs)

    return tbl_source


def _unity_source(dx=0, dy=0, angle=0, weight=1, n=100):
    unit = u.Unit("ph s-1 m-2 um-1")
    wave = np.linspace(0.5, 2.5, n) * u.um
    specs = [SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.ones(n) * unit)]

    im_wcs = wcs.WCS(naxis=2)
    im_wcs.wcs.cunit = [u.arcsec, u.arcsec]
    im_wcs.wcs.cdelt = [1, 1]
    im_wcs.wcs.crval = [0, 0]
    im_wcs.wcs.crpix = [n/2, n/2]
    im_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    im = np.ones((n, n))

    im_hdu = fits.ImageHDU(data=im, header=im_wcs.to_header())
    im_hdu.header["SPEC_REF"] = 0
    im_source = Source(image_hdu=im_hdu, spectra=specs)

    angle = angle * np.pi / 180
    im_source.fields[0].header["CRVAL1"] += dx * u.arcsec.to(u.deg)
    im_source.fields[0].header["CRVAL2"] += dy * u.arcsec.to(u.deg)
    im_source.fields[0].header["PC1_1"] = np.cos(angle)
    im_source.fields[0].header["PC1_2"] = np.sin(angle)
    im_source.fields[0].header["PC2_1"] = -np.sin(angle)
    im_source.fields[0].header["PC2_2"] = np.cos(angle)
    im_source.fields[0].data *= weight

    return im_source


def _empty_sky():
    n = 3
    unit = u.Unit("ph s-1 m-2 um-1")
    wave = np.linspace(0.5, 2.5, n) * u.um
    specs = [SourceSpectrum(Empirical1D, points=wave,
                            lookup_table=np.zeros(n) * unit)]
    tbl = Table(names=["x", "y", "ref", "weight"],
                data=[[0], [0], [0], [0]])
    tbl_source = Source(table=tbl, spectra=specs)

    return tbl_source


def _vega_source(mag=0, x=0, y=0):
    specs = [vega_spectrum(mag)]
    tbl = Table(names=["x", "y", "ref", "weight"],
                data=[[x]*u.arcsec, [y]*u.arcsec, [0], [1]])
    tbl_source = Source(table=tbl, spectra=specs)

    return tbl_source


def _make_dummy_cube(scale, wave_unit, ref_wave, wave_step, wave_type, bunit):

    if isinstance(scale, u.Quantity) is False:
        scale = scale * u.arcsec
    if isinstance(wave_unit, u.core.Unit) is False:
        wave_unit = u.AA
    if isinstance(ref_wave, u.Quantity) is False:
        ref_wave = ref_wave * u.AA
    if isinstance(wave_step, u.Quantity) is False:
        wave_step = wave_step * u.AA

    data = np.ones(shape=(100, 20, 20))
    header = fits.Header(dict(NAXIS=3,
                              WCSAXES=3,
                              NAXIS1=data.shape[2] + 1,
                              NAXIS2=data.shape[1] + 1,
                              NAXIS3=data.shape[0] + 1,
                              CRPIX1=data.shape[2] // 2,
                              CRPIX2=data.shape[1] // 2,
                              CRPIX3=1,
                              CRVAL1=0,
                              CRVAL2=0,
                              CRVAL3=ref_wave.to(wave_unit).value,
                              CDELT1=-1 * scale.to(u.deg).value,
                              CDELT2=scale.to(u.deg).value,
                              CDELT3=wave_step.to(wave_unit).value,
                              CUNIT1="DEG",
                              CUNIT2="DEG",
                              CUNIT3=wave_unit.to_string(),
                              CTYPE1='RA---TAN',
                              CTYPE2='DEC--TAN',
                              CTYPE3=wave_type,
                              BUNIT=bunit))

    hdu = fits.ImageHDU(data=data, header=header)

    return hdu
