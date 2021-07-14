import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column

from . import fov_utils
from . import image_plane_utils as imp_utils

from ..base_classes import SourceBase, FieldOfViewBase, PoorMansHeader
from .. import utils


class FieldOfView(FieldOfViewBase):
    """
    A FOV is a monochromatic image. Flux units after extracting the fields from
    the Source are in ph/s/pixel

    The initial header should contain an on-sky WCS description:
    - CDELT-, CUNIT-, NAXIS- : for pixel scale and size (assumed CUNIT in deg)
    - CRVAL-, CRPIX- : for positioning the final image
    - CTYPE- : is assumed to be "RA---TAN", "DEC---TAN"

    and an image-plane WCS description
    - CDELT-D, CUNIT-D, NAXISn : for pixel scale and size (assumed CUNIT in mm)
    - CRVAL-D, CRPIX-D : for positioning the final image
    - CTYPE-D : is assumed to be "LINEAR", "LINEAR"

    The wavelength range is given by waverange

    """

    def __init__(self, header, waverange, **kwargs):
        self.meta = {"id": None,
                     "wave_min": utils.quantify(waverange[0], u.um),
                     "wave_max": utils.quantify(waverange[1], u.um),
                     "area": 0 * u.m**2,
                     "sub_pixel": "!SIM.sub_pixel.flag",
                     "distortion": {"scale": [1, 1],
                                    "offset": [0, 0],
                                    "shear": [1, 1],
                                    "rotation": 0,
                                    "radius_of_curvature": None},
                     "conserve_image": True,
                     }
        self.meta.update(kwargs)

        if not any([utils.has_needed_keywords(header, s) for s in ["", "S"]]):
            raise ValueError("header must contain a valid sky-plane WCS: {}"
                             "".format(dict(header)))
        if not utils.has_needed_keywords(header, "D"):
            raise ValueError("header must contain a valid image-plane WCS: {}"
                             "".format(dict(header)))

        # do we actually need this distinction?
        if isinstance(header, PoorMansHeader):
            self.hdu = fits.ImageHDU()
            self.hdu.header.update(header)
        else:
            self.hdu = fits.ImageHDU(header=header)
        self.hdu.header["NAXIS"] = 2
        self.hdu.header["NAXIS1"] = header["NAXIS1"]
        self.hdu.header["NAXIS2"] = header["NAXIS2"]

        self.fields = []
        self.spectra = []
        self.image_plane_id = 0

        self._wavelength = None

    def extract_from(self, src):
        """ ..assumption: Bandpass has been applied
        point sources
        - extracts relevant entries from all tables and adds to a new table
        - extracts relevant spectra and updates table entries
        extended sources
        - extracts relevant spatial region from source._file .
        """





        if not isinstance(src, SourceBase):
            raise ValueError("source must be a Source object: {}"
                             "".format(type(src)))

        wave_min = utils.quantify(self.meta["wave_min"], u.um).value
        wave_max = utils.quantify(self.meta["wave_max"], u.um).value
        area = self.meta["area"]

        # determine which fields are inside the field of view
        fields_mask = [fov_utils.is_field_in_fov(self.hdu.header, field)
                       for field in src.fields]
        fields_indexes = np.where(fields_mask)[0]
        tbl_fields_mask = np.array([isinstance(field, Table)
                                    for field in src.fields])
        img_fields_mask = np.array([isinstance(field, fits.ImageHDU)
                                    for field in src.fields])
        # cube_fields_mask

        # combine all Table fields
        if sum(tbl_fields_mask * fields_mask) > 0:
            combined_table = fov_utils.combine_table_fields(self.hdu.header,
                                                            src, fields_indexes)
            tbl = fov_utils.make_flux_table(combined_table, src,
                                            wave_min, wave_max, area)
            xd, yd = fov_utils.sky2fp(self.hdu.header, tbl["x"], tbl["y"])
            tbl.add_columns([Column(name="x_mm", data=xd, unit=u.mm),
                             Column(name="y_mm", data=yd, unit=u.mm)])
            self.fields += [tbl]

        # combine all ImageHDU fields
        if sum(img_fields_mask * fields_mask) > 0:
            imagehdu = fov_utils.combine_imagehdu_fields(self.hdu.header, src,
                                                         fields_indexes,
                                                         wave_min, wave_max,
                                                         area)
            self.fields += [imagehdu]

    def view(self, sub_pixel=None):
        if sub_pixel is None:
            sub_pixel = self.meta["sub_pixel"]

        self.hdu.data = np.zeros((self.hdu.header["NAXIS2"],
                                  self.hdu.header["NAXIS1"]))
        if len(self.fields) > 0:
            for field in self.fields:
                if isinstance(field, Table):
                    self.hdu = imp_utils.add_table_to_imagehdu(field, self.hdu,
                                                               sub_pixel)
                elif isinstance(field, fits.ImageHDU):
                    self.hdu.data += field.data

        if self.meta["conserve_image"] is False and self.mask is not None:
            flux = np.sum(self.hdu.data) / np.sum(self.mask)
            self.hdu.data = np.zeros(self.hdu.data.shape)
            self.hdu.data[self.mask] = flux

        return self.hdu.data

    @property
    def header(self):
        return self.hdu.header

    @property
    def data(self):
        if self.hdu.data is None:
            self.view(self.meta["sub_pixel"])

        return self.hdu.data

    @property
    def image(self):
        return self.data

    @property
    def corners(self):
        sky_corners = imp_utils.calc_footprint(self.header)
        imp_corners = imp_utils.calc_footprint(self.header, "D")
        return sky_corners, imp_corners

    @property
    def wavelength(self):
        if self._wavelength is None:
            self._wavelength = 0.5 * (self.meta["wave_min"] +
                                      self.meta["wave_max"])
        return self._wavelength

    def __repr__(self):
        msg = "FOV id: {}, with dimensions ({}, {})\n" \
              "".format(self.meta["id"], self.header["NAXIS1"],
                        self.header["NAXIS2"])
        msg += "Sky centre: ({}, {})\n" \
               "".format(self.header["CRVAL1"], self.header["CRVAL2"])
        msg += "Image centre: ({}, {})\n" \
               "".format(self.header["CRVAL1D"], self.header["CRVAL2D"])
        msg += "Wavelength range: ({}, {})um\n" \
               "".format(self.meta["wave_min"], self.meta["wave_max"])

        return msg
