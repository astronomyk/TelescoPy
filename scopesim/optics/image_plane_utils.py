import warnings

import numpy as np
from astropy import units as u, wcs
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import interpolation as ndi

from .. import utils


###############################################################################
# Make Canvas


def get_canvas_header(hdu_or_table_list, pixel_scale=1 * u.arcsec):
    """
    Generate a fits.Header with a WCS that covers everything in the FOV

    Parameters
    ----------
    hdu_or_table_list : list
        A list of Tables and/or ImageHDU py_objects

    pixel_scale : astropy.Quantity
        [arcsec] The pixel scale of the projection. Default in 1 arcsec

    Returns
    -------
    header : fits.Header
        A Header containing a WCS and NAXISn values to build an ImageHDU

    """

    size_warning = "Header dimension are {} large: {}. Any image made from " \
                   "this header will use more that >{} in memory"

    headers = [ht.header for ht in hdu_or_table_list
               if isinstance(ht, fits.ImageHDU)]
    if sum([isinstance(ht, Table) for ht in hdu_or_table_list]) > 0:
        tbls = [ht for ht in hdu_or_table_list if isinstance(ht, Table)]
        tbl_hdr = _make_bounding_header_for_tables(tbls,
                                                   pixel_scale=pixel_scale)
        headers += [tbl_hdr]

    if len(headers) > 0:
        hdr = _make_bounding_header_from_imagehdus(headers,
                                                   pixel_scale=pixel_scale)
        num_pix = hdr["NAXIS1"] * hdr["NAXIS2"]
        if num_pix > 2 ** 25:  # 2 * 4096**2
            warnings.warn(size_warning.format("", num_pix, "256 MB"))
        elif num_pix > 2 ** 28:
            raise MemoryError(size_warning.format("too", num_pix, "8 GB"))
    else:
        warnings.warn("No tables or ImageHDUs were passed")
        hdr = None

    return hdr


def _make_bounding_header_from_imagehdus(imagehdus, pixel_scale=1*u.arcsec):
    """
    Returns a Header with WCS and NAXISn keywords bounding all input ImageHDUs

    Parameters
    ----------
    imagehdus : list of fits.ImageHDU
    pixel_scale : u.Quantity
        [arcsec]

    Returns
    -------
    hdr : fits.Header

    """

    x = []
    y = []
    if pixel_scale.unit.physical_type == "angle":
        s = ""
    elif pixel_scale.unit.physical_type == "length":
        s = "D"

    for imagehdu in imagehdus:
        if isinstance(imagehdu, fits.ImageHDU):
            x_foot, y_foot = calc_footprint(imagehdu.header, s)
        else:
            x_foot, y_foot = calc_footprint(imagehdu, s)
        x += list(x_foot)
        y += list(y_foot)
    unit = u.mm if s == "D" else u.deg
    pixel_scale = pixel_scale.to(unit).value
    hdr = header_from_list_of_xy(x, y, pixel_scale, s)
    hdr["NAXIS1"] += 2
    hdr["NAXIS2"] += 2

    return hdr


def _make_bounding_header_for_tables(tables, pixel_scale=1*u.arcsec):
    """
    Returns a Header with WCS and NAXISn keywords bounding all input Tables

    Parameters
    ----------
    tables : list of astropy.Tables
        [arcsec] Must contain columns: "x", "y"

    pixel_scale : u.Quantity
        [arcsec]

    Returns
    -------
    hdr : fits.Header

    """

    x = []
    y = []

    s = "D" if pixel_scale.unit.physical_type == "length" else ""
    unit_new = u.mm if s == "D" else u.deg
    unit_orig = u.mm if s == "D" else u.arcsec
    x_name = "x_mm" if s == "D" else "x"
    y_name = "y_mm" if s == "D" else "y"

    pixel_scale = pixel_scale.to(unit_new).value
    for table in tables:
        x_col = utils.quantity_from_table(x_name, table, unit_orig).to(unit_new)
        y_col = utils.quantity_from_table(y_name, table, unit_orig).to(unit_new)
        x_col = list(x_col.value)
        y_col = list(y_col.value)
        x += [np.min(x_col), np.max(x_col) + 2 * pixel_scale]
        y += [np.min(y_col), np.max(y_col) + 2 * pixel_scale]

    hdr = header_from_list_of_xy(x, y, pixel_scale, s)

    return hdr


def header_from_list_of_xy(x, y, pixel_scale, wcs_suffix=""):
    """
    Makes a header large enough to contain all x,y on-sky coordinates

    Parameters
    ----------
    x, y : list of floats
        [deg, mm] List of sky coordinates to be bounded by the NAXISn keys
    pixel_scale : float
        [deg, mm]

    Returns
    -------
    hdr : fits.Header

    """

    s = wcs_suffix
    if wcs_suffix != "D":
        x = np.array(x)
        x[x > 270] -= 360
        x[x < -90] += 360
        x = list(x)

    hdr = fits.Header()

    crval1 = divmod(min(x), pixel_scale)[0] * pixel_scale
    crval2 = divmod(min(y), pixel_scale)[0] * pixel_scale

    # naxis1 = int((max(x) - crval1) // pixel_scale) # + 2
    # naxis2 = int((max(y) - crval2) // pixel_scale) # + 2

    # crval1 = min(x)
    # crval2 = min(y)

    # ..todo:: give the 9 a variable in !SIM.computing
    dx = (max(x) - min(x)) / pixel_scale
    dy = (max(y) - min(y)) / pixel_scale
    naxis1 = int(np.round(dx))
    naxis2 = int(np.round(dy))

    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis1
    hdr["NAXIS2"] = naxis2
    hdr["CTYPE1"+s] = "LINEAR" if s == "D" else "RA---TAN"
    hdr["CTYPE2"+s] = "LINEAR" if s == "D" else "DEC--TAN"
    hdr["CUNIT1"+s] = "mm" if s == "D" else "deg"
    hdr["CUNIT2"+s] = "mm" if s == "D" else "deg"
    hdr["CDELT1"+s] = pixel_scale
    hdr["CDELT2"+s] = pixel_scale
    hdr["CRVAL1"+s] = crval1
    hdr["CRVAL2"+s] = crval2
    hdr["CRPIX1"+s] = 0.
    hdr["CRPIX2"+s] = 0.

    xpcen, ypcen = naxis1 // 2, naxis2 // 2
    xscen, yscen = pix2val(hdr, xpcen, ypcen, s)
    hdr["CRVAL1"+s] = float(xscen)
    hdr["CRVAL2"+s] = float(yscen)
    hdr["CRPIX1"+s] = xpcen
    hdr["CRPIX2"+s] = ypcen

    return hdr




###############################################################################
# Table overlays


def add_table_to_imagehdu(table, canvas_hdu, sub_pixel=True, wcs_suffix=""):
    """
    Add files from an astropy.Table to the image of an fits.ImageHDU

    Parameters
    ----------
    table : astropy.Table
        Must contain the columns "x_mm", "y_mm", "flux" with the units in the
        column attribute .unit, or in the table.meta dictionary as
        "<colname>_unit". Default units are ``mm`` and ``ph / s / pix``

    canvas_hdu : fits.ImageHDU
        The ImageHDU onto which the table files should be projected.
        This must include a valid WCS

    sub_pixel : bool, optional
        Default is True. If True, sub-pixel shifts of files will be taken into
        account when projecting onto the canvas pixel grid. This takes about 5x
        longer than ignoring the sub-pixel shifts

    wcs_suffix : str, optional

    Returns
    -------
    canvas_hdu : fits.ImageHDU

    """

    s = wcs_suffix
    if not utils.has_needed_keywords(canvas_hdu.header, s):
        raise ValueError("canvas_hdu must include an appropriate WCS: {}"
                         "".format(s))

    f = utils.quantity_from_table("flux", table, default_unit=u.Unit("ph s-1"))
    if s == "D":
        x = utils.quantity_from_table("x_mm", table, default_unit=u.mm).to(u.mm)
        y = utils.quantity_from_table("y_mm", table, default_unit=u.mm).to(u.mm)
    else:
        arcsec = u.arcsec
        x = utils.quantity_from_table("x", table, default_unit=arcsec).to(u.deg)
        y = utils.quantity_from_table("y", table, default_unit=arcsec).to(u.deg)

    xpix, ypix = val2pix(canvas_hdu.header, x.value, y.value, s)

    # Weird FITS / astropy behaviour. Axis1 == y, Axis2 == x.
    naxis1 = canvas_hdu.header["NAXIS1"]
    naxis2 = canvas_hdu.header["NAXIS2"]
    # Also occasionally 0 is returned as ~ -1e-11
    eps = -1e-7
    mask = (xpix >= eps) * (xpix < naxis1) * (ypix >= eps) * (ypix < naxis2)

    if sub_pixel is True:
        canvas_hdu = _add_subpixel_sources_to_canvas(canvas_hdu, xpix, ypix, f,
                                                     mask)
    else:
        canvas_hdu = _add_intpixel_sources_to_canvas(canvas_hdu, xpix, ypix, f,
                                                     mask)

    return canvas_hdu


def _add_intpixel_sources_to_canvas(canvas_hdu, xpix, ypix, flux, mask):
    canvas_hdu.header["comment"] = "Adding {} int-pixel files" \
                                   "".format(len(flux))
    xpix = xpix.astype(int)
    ypix = ypix.astype(int)
    for ii in range(len(xpix)):
        if mask[ii]:
            canvas_hdu.data[ypix[ii], xpix[ii]] += flux[ii].value

    return canvas_hdu


def _add_subpixel_sources_to_canvas(canvas_hdu, xpix, ypix, flux, mask):
    canvas_hdu.header["comment"] = "Adding {} sub-pixel files" \
                                   "".format(len(flux))
    canvas_shape = canvas_hdu.data.shape
    for ii in range(len(xpix)):
        if mask[ii]:
            xx, yy, fracs = sub_pixel_fractions(xpix[ii], ypix[ii])
            for x, y, frac in zip(xx, yy, fracs):
                if y < canvas_shape[0] and x < canvas_shape[1]:
                    canvas_hdu.data[y, x] += frac * flux[ii].value

    return canvas_hdu


def sub_pixel_fractions(x, y):
    """
    Makes a list of pixel coordinates and weights to reflect sub-pixel shifts

    A point source which isn't centred on a pixel can be modelled by a centred
    PSF convolved with a shifted delta function. A fraction of the delta
    function in moved into each of the adjoining pixels. For example, a star
    at ``(x,y)=(0.2, 0.2)`` would be represented by a following pixel weights::

       ---------------
       | 0.16 | 0.04 |
       ---------------
       | 0.64 | 0.16 |
       ---------------

    where (0,0) is the centre of the bottom-left pixel

    Given (x,y) pixel coordinates, this function returns the fractions of flux
    that should go into the surrounding pixels, as well as the coordinates of
    those neighbouring pixels.

    Parameters
    ----------
    x, y : float

    Returns
    -------
    x_pix, y_pix, fracs : list of (int, int, float)
       The x and y pixel coordinates and their corresponding flux fraction

    """

    x0, dx = divmod(x, 1)
    y0, dy = divmod(y, 1)

    xi0 = int(x0)
    xi1 = xi0 + bool(dx)
    yi0 = int(y0)
    yi1 = yi0 + bool(dy)

    f00 = (1. - dx) * (1. - dy)
    f01 = (1. - dx) * dy
    f10 = dx * (1 - dy)
    f11 = dx * dy

    x_pix = [xi0, xi1, xi0, xi1]
    y_pix = [yi0, yi0, yi1, yi1]
    fracs = [f00, f10, f01, f11]

    return x_pix, y_pix, fracs


###############################################################################
# Image overlays
#
#
# def overlay_image_old(small_im, big_im, coords, mask=None, sub_pixel=False):
#     """
#     Overlay small_im on top of big_im at the position specified by coords
#
#     ``small_im`` will be centred at ``coords``
#
#     Adapted from:
#     ``https://stackoverflow.com/questions/14063070/
#         overlay-a-smaller-image-on-a-larger-image-python-opencv``
#
#     """
#
#     # TODO - Add in a catch for sub-pixel shifts
#     if sub_pixel:
#         raise NotImplementedError
#
#     x, y = np.array(coords, dtype=int) - np.array(small_im.shape) // 2
#
#     # Image ranges
#     x1, x2 = max(0, x), min(big_im.shape[0], x + small_im.shape[0])
#     y1, y2 = max(0, y), min(big_im.shape[1], y + small_im.shape[1])
#
#     # Overlay ranges
#     x1o, x2o = max(0, -x), min(small_im.shape[0], big_im.shape[0] - x)
#     y1o, y2o = max(0, -y), min(small_im.shape[1], big_im.shape[1] - y)
#
#     # Exit if nothing to do
#     if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
#         return big_im
#
#     if mask is None:
#         big_im[x1:x2, y1:y2] += small_im[x1o:x2o, y1o:y2o]
#     else:
#         mask = mask[x1o:x2o, y1o:y2o].astype(bool)
#         big_im[x1:x2, y1:y2][mask] += small_im[x1o:x2o, y1o:y2o][mask]
#
#     return big_im
#

def overlay_image(small_im, big_im, coords, mask=None, sub_pixel=False):
    """
    Overlay small_im on top of big_im at the position specified by coords

    ``small_im`` will be centred at ``coords``

    Adapted from:
    ``https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv``

    """

    # TODO - Add in a catch for sub-pixel shifts
    if sub_pixel:
        raise NotImplementedError

    y, x = np.array(coords, dtype=int)[::-1] - np.array(small_im.shape) // 2

    # Image ranges
    x1, x2 = max(0, x), min(big_im.shape[1], x + small_im.shape[1])
    y1, y2 = max(0, y), min(big_im.shape[0], y + small_im.shape[0])

    # Overlay ranges
    x1o, x2o = max(0, -x), min(small_im.shape[1], big_im.shape[1] - x)
    y1o, y2o = max(0, -y), min(small_im.shape[0], big_im.shape[0] - y)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return big_im

    if mask is None:
        big_im[y1:y2, x1:x2] += small_im[y1o:y2o, x1o:x2o]
    else:
        mask = mask[y1o:y2o, x1o:x2o].astype(bool)
        big_im[y1:y2, x1:x2][mask] += small_im[y1o:y2o, x1o:x2o][mask]

    return big_im


def rescale_imagehdu(imagehdu, pixel_scale, wcs_suffix="", conserve_flux=True,
                     order=1):
    """
    Scales the .data array by the ratio of pixel_scale [deg] and CDELTn

    ``pixel_scale`` should NOT be passed as a Quantity!

    Parameters
    ----------
    imagehdu : fits.ImageHDU
    pixel_scale : float
        [deg] NOT to be passed as a Quantity
    wcs_suffix : str

    conserve_flux : bool

    order : int
        [1..5] Order of the spline interpolation used by
        ``scipy.ndimage.rotate``

    Returns
    -------
    imagehdu : fits.ImageHDU

    """
    wcs_suffix = wcs_suffix
    s0 = wcs_suffix[0] if len(wcs_suffix) > 0 else ""
    cdelt1 = imagehdu.header["CDELT1"+s0]
    cdelt2 = imagehdu.header["CDELT2"+s0]

    zoom1 = np.abs(cdelt1 / pixel_scale)
    zoom2 = np.abs(cdelt2 / pixel_scale)  # making sure that zoom1,zoom2 are positive

    if zoom1 != 1 or zoom2 != 1:
        sum_orig = np.sum(imagehdu.data)
        new_im = ndi.zoom(imagehdu.data, (zoom1, zoom2), order=order)

        if conserve_flux:
            new_im = np.nan_to_num(new_im, copy=False)
            sum_new = np.sum(new_im)
            if sum_new != 0:
                new_im *= sum_orig / sum_new

        imagehdu.data = new_im

        for ii in range(max(1, len(wcs_suffix))):
            si = wcs_suffix[ii] if len(wcs_suffix) > 0 else ""
            imagehdu.header["CRPIX1"+si] *= zoom1
            imagehdu.header["CRPIX2"+si] *= zoom2
            imagehdu.header["CDELT1"+si] = pixel_scale
            imagehdu.header["CDELT2"+si] = pixel_scale
            imagehdu.header["CUNIT1"+si] = "deg"
            imagehdu.header["CUNIT1"+si] = "deg"

    return imagehdu


def reorient_imagehdu(imagehdu, wcs_suffix="", conserve_flux=True, order=1):
    """
    Applies an affine transformation to the image, as given in its header

    Parameters
    ----------
    imagehdu : fits.ImageHDU

    wcs_suffix : str

    conserve_flux : bool

    order : int
        [1..5] Order of the spline interpolation used by
        ``scipy.ndimage.rotate``

    Returns
    -------
    imagehdu : fits.ImageHDU

    """

    s = wcs_suffix
    rad2deg = np.pi / 180

    hdr = imagehdu.header
    pc_keys = ["PC1_1", "PC1_2", "PC2_1", "PC2_2"]
    if all(key+s in hdr for key in pc_keys) and imagehdu.data is not None:
        xscen, yscen = pix2val(hdr, hdr["NAXIS1"] / 2., hdr["NAXIS2"] / 2., s)
        hdr["CRVAL1" + s] = xscen
        hdr["CRVAL2" + s] = yscen

        mat = np.array([[hdr["PC1_1" + s], hdr["PC1_2" + s]],
                        [hdr["PC2_1" + s], hdr["PC2_2" + s]]])      #
        new_im = affine_map(imagehdu.data, mat, reshape=True, order=order)
        # new_im = ndi.rotate(imagehdu.data, angle, reshape=True, order=order)

        if conserve_flux:
            new_im = np.nan_to_num(new_im, copy=False)
            new_im *= np.sum(imagehdu.data) / np.sum(new_im)

        imagehdu.data = new_im
        hdr["CRPIX1"+s] = hdr["NAXIS1"] / 2.
        hdr["CRPIX2"+s] = hdr["NAXIS2"] / 2.
        for card in ["PC1_1"+s, "PC1_2"+s, "PC2_1"+s, "PC2_2"+s]:
            hdr.remove(card)
        imagehdu.header = hdr

    elif any(["PC1_1" in key for key in imagehdu.header]):
        warnings.warn("PC Keywords were found, but not used due to different "
                      "wcs_suffix given: {} \n {}"
                      "".format(wcs_suffix, dict(imagehdu.header)))

    return imagehdu


def add_imagehdu_to_imagehdu(image_hdu, canvas_hdu, order=1, wcs_suffix="",
                             conserve_flux=True):
    """
    Re-project one ``fits.ImageHDU`` onto another ``fits.ImageHDU``

    ..assumption:: of equal grid coordinate lengths

    Parameters
    ----------
    image_hdu : fits.ImageHDU
        The ``ImageHDU`` which will be reprojected onto `canvas_hdu`

    canvas_hdu : fits.ImageHDU
        The ``ImageHDU`` onto which the image_hdu should be projected.
        This must include a valid WCS

    order : int, optional
        Default is 1. The order of the spline interpolator used by the
        ``scipy.ndimage`` functions

    wcs_suffix : str
        To determine which WCS to use. "" for sky HDUs and "D" for
        ImagePlane HDUs

    conserve_flux : bool
        Default is True. Used when zooming and rotating to keep flux constant.

    Returns
    -------
    canvas_hdu : fits.ImageHDU

    """

    # .. todo: Add a catch for projecting a large image onto a small canvas

    if isinstance(image_hdu.data, u.Quantity):
        image_hdu.data = image_hdu.data.value
    pixel_scale = float(canvas_hdu.header["CDELT1"+wcs_suffix])

    new_hdu = rescale_imagehdu(image_hdu, pixel_scale=pixel_scale,
                               wcs_suffix=wcs_suffix, order=order,
                               conserve_flux=conserve_flux)
    new_hdu = reorient_imagehdu(new_hdu,
                                wcs_suffix=wcs_suffix, order=order,
                                conserve_flux=conserve_flux)

    xcen_im = new_hdu.header["NAXIS1"] // 2
    ycen_im = new_hdu.header["NAXIS2"] // 2

    xsky0, ysky0 = pix2val(new_hdu.header, xcen_im, ycen_im, wcs_suffix)
    xpix0, ypix0 = val2pix(canvas_hdu.header, xsky0, ysky0, wcs_suffix)

    # again, I need to add this transpose operation - WHY????
    # Image plane tests need the transpose operation, but FOV broadcast tests don't. Weird
    canvas_hdu.data = overlay_image(new_hdu.data, canvas_hdu.data,
                                    coords=(xpix0, ypix0))

    return canvas_hdu


def pix2val(header, x, y, wcs_suffix=""):
    """
    Returns the real coordinates [deg, mm] for coordinates from a Header WCS

    Parameters
    ----------
    header : fits.Header
    x, y : float, list, array
    wcs_suffix : str

    Returns
    -------
    a, b : float, array
        [deg, mm] Real coordinates as given by the Header WCS

    """

    s = wcs_suffix
    if "CD1_1" + s in header:
        pc11 = header["CD1_1" + s]
        pc12 = header["CD1_2" + s]
        pc21 = header["CD2_1" + s]
        pc22 = header["CD2_2" + s]
        da = 1  # CD1_1, etc contain both rotation and scale
        db = 1

    elif "PC1_1" + s in header:
        pc11 = header["PC1_1" + s]
        pc12 = header["PC1_2" + s]
        pc21 = header["PC2_1" + s]
        pc22 = header["PC2_2" + s]
        da = header["CDELT1" + s]
        db = header["CDELT2" + s]
    else:
        pc11, pc12, pc21, pc22 = 1, 0, 0, 1
        da = header["CDELT1" + s]
        db = header["CDELT2" + s]

    x0 = header["CRPIX1" + s]
    y0 = header["CRPIX2" + s]
    a0 = header["CRVAL1" + s]
    b0 = header["CRVAL2" + s]

    a = a0 + da * ((x - x0) * pc11 + (y - y0) * pc12)
    b = b0 + db * ((x - x0) * pc21 + (y - y0) * pc22)

    return a, b


def val2pix(header, a, b, wcs_suffix=""):
    """
    Returns the pixel coordinates for real coordinates [deg, mm] from a WCS

    Parameters
    ----------
    header : fits.Header
    a, b : float, list, array
        [deg, mm]

    Returns
    -------
    x, y : float, array
        [pixel] Pixel coordinates as given by the Header WCS

    """

    s = wcs_suffix

    if isinstance(header, fits.ImageHDU):
        header = header.header

    if "CD1_1" + s in header:
        pc11 = header["CD1_1" + s]
        pc12 = header["CD1_2" + s]
        pc21 = header["CD2_1" + s]
        pc22 = header["CD2_2" + s]
        da = 1  # CD1_1, etc contain both rotation and scale
        db = 1

    elif "PC1_1" + s in header:
        pc11 = header["PC1_1" + s]
        pc12 = header["PC1_2" + s]
        pc21 = header["PC2_1" + s]
        pc22 = header["PC2_2" + s]
        da = header["CDELT1" + s]
        db = header["CDELT2" + s]
    else:
        pc11, pc12, pc21, pc22 = 1, 0, 0, 1
        da = header["CDELT1" + s]
        db = header["CDELT2" + s]

    x0 = header["CRPIX1" + s]
    y0 = header["CRPIX2" + s]
    a0 = header["CRVAL1" + s]
    b0 = header["CRVAL2" + s]

    x = x0 + 1. / da * ((a - a0) * pc11 - (b - b0) * pc21)
    y = y0 + 1. / db * ((a - a0) * pc12 + (b - b0) * pc22)

    return x, y


def calc_footprint(header, wcs_suffix=""):
    """
    Returns the on sky-positions [deg] of the corners of a header WCS

    The on-sky positions returned correspond to the corners of the header's
    image array, in this order::

        (ra, dec) = (0,0), (w, 0), (w, h), (0, h)

    where ``w``, ``h`` are equal to NAXIS1 and NAXIS2 from the header.

    Parameters
    ----------
    header : fits.Header

    wcs_suffix : str
        Letter suffix for the WCS keywords, e.g. CDELT1D for image-plane coords

    Returns
    -------
    xsky, ysky : arrays of floats
        [deg] xsky are the coordinates for pixels [0, w, w, 0]
        [deg] ysky are the coordinates for pixels [0, 0, h, h]

    """

    if isinstance(header, fits.ImageHDU):
        header = header.header

    w, h = header["NAXIS1"], header["NAXIS2"]
    x = np.array([0, w, w, 0])
    y = np.array([0, 0, h, h])

    xsky, ysky = pix2val(header, x, y, wcs_suffix)

    return xsky, ysky


def split_header(hdr, chunk_size, wcs_suffix=""):
    """
    Splits a header into many smaller parts of the chunk_size

    Parameters
    ----------
    hdr
    chunk_size
    wcs_suffix

    Returns
    -------

    """
    # ..todo:: test that this works
    s = wcs_suffix
    naxis1, naxis2 = hdr["NAXIS1"+s], hdr["NAXIS2"+s]
    x0_pix, y0_pix = hdr["CRPIX1"+s], hdr["CRPIX2"+s]       # pix
    x0_sky, y0_sky = hdr["CRVAL1"+s], hdr["CRVAL2"+s]       # deg
    x_delt, y_delt = hdr["CDELT1"+s], hdr["CDELT2"+s]       # deg / pix

    hdr_list = []
    for x1_pix in range(0, naxis1, chunk_size):
        for y1_pix in range(0, naxis2, chunk_size):
            x1_sky = x0_sky + (x1_pix - x0_pix) * x_delt
            y1_sky = y0_sky + (y1_pix - y0_pix) * y_delt
            x2_sky = x1_sky + x_delt * min(chunk_size, naxis1 - x1_pix)
            y2_sky = y1_sky + y_delt * min(chunk_size, naxis2 - y1_pix)

            hdr_sky = header_from_list_of_xy([x1_sky, x2_sky], [y1_sky, y2_sky],
                                             pixel_scale=x_delt, wcs_suffix=s)
            hdr_list += [hdr_sky]

    return hdr_list


def affine_map(input, matrix=None, rotation_angle=0, shear_angle=0,
               scale_factor=None, reshape=True, order=3):
    """
    Applies an affine transformation matrix to an image around its centre

    Similar functionality to ``scipy.ndimage.rotate`` but for the
    ``affine_transformation`` function

    Either a 2x2 affine transformation matrix can be supplied, or the rotation,
    shear, and scaling values which are the basis of an affine transformation.


    Parameters
    ----------
    input : array_like
        The input array
    matrix : ndarray, optional
        A 2x2 affine transformation matrix
    rotation_angle : float, optional
        [deg] If matrix==None, a rotation matrix is built from this angle
    shear_angle : float, optional
        [deg] If matrix==None, a y-axis shear matrix is built from this angle
    scale_factor : list, array
        [mx, my] If matrix==None, a scaling matrix is built from this list
    reshape : bool, optional
        If True, the array is re-sized to contain the whole transformed image
    order : int, optional
        Default is 3. Spline interpolation order

    Returns
    -------
    output : array-like
        The new mapping of the image

    """

    d2r = np.pi / 180.
    if matrix is None:
        c = np.cos(rotation_angle * d2r) if rotation_angle != 0. else 1.
        s = np.sin(rotation_angle * d2r) if rotation_angle != 0. else 0.
        t = np.tan(shear_angle * d2r) if shear_angle != 0. else 0.
        sx = scale_factor[0] if scale_factor is not None else 1.
        sy = scale_factor[1] if scale_factor is not None else 1.
        mat = np.array([[c, s], [-s, c]]) @ \
              np.array([[1, t], [0, 1]]) @ \
              np.array([[sx, 0], [0, sy]])
    else:
        mat = np.array(matrix)

    h, w = np.array(input.shape)
    if reshape:
        # Find the new corner coordinates using the py3.5+ matmul operator
        x, y = mat @ np.array([[0, w, w, 0], [0, 0, h, h]])
        out = np.ceil(np.array([y.max() - y.min(),
                                x.max() - x.min()])).astype(int)
    else:
        out = np.array([h, w])

    c_out = 0.5 * out
    c_in = 0.5 * np.array([h, w])
    offset = c_in - c_out.dot(mat)

    output = ndi.affine_transform(input, mat.T, output_shape=out,
                                  offset=offset, order=order)

    return output
