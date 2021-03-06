from copy import deepcopy
import numpy as np
from scipy.signal import convolve

from astropy import units as u
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
import anisocado as aniso

from .effects import Effect
from . import ter_curves_utils as tu
from . import psf_utils as pu
from ..base_classes import ImagePlaneBase, FieldOfViewBase
from .. import utils


class PoorMansFOV:
    def __init__(self, recursion_call=False):
        pixel_scale = utils.from_currsys("!INST.pixel_scale")
        self.header = {"CDELT1": pixel_scale / 3600.,
                       "CDELT2": pixel_scale / 3600.,
                       "NAXIS1": 128,
                       "NAXIS2": 128,}
        self.meta = utils.from_currsys("!SIM.spectral")
        self.wavelength = self.meta["wave_mid"] * u.um
        if not recursion_call:
            self.hdu = PoorMansFOV(recursion_call=True)


class PSF(Effect):
    def __init__(self, **kwargs):
        self.kernel = None
        self.valid_waverange = None
        self._waveset = None
        super().__init__(**kwargs)

        params = {"flux_accuracy": "!SIM.computing.flux_accuracy",
                  "sub_pixel_flag": "!SIM.sub_pixel.flag",
                  "z_order": [40, 640],
                  "convolve_mode": "same",      # "full", "same"
                  "bkg_width": -1,
                  "wave_key": "WAVE0",
                  "normalise_kernel": True,
                  "rotational_blur_angle": 0,
                  "report_plot_include": True,
                  "report_table_include": False,
                  }
        self.meta.update(params)
        self.meta.update(kwargs)
        self.meta = utils.from_currsys(self.meta)
        self.apply_to_classes = (FieldOfViewBase, ImagePlaneBase)

    def apply_to(self, obj):
        if isinstance(obj, self.apply_to_classes):
            if (hasattr(obj, "fields") and len(obj.fields) > 0) or \
                    obj.hdu.data is not None:

                if obj.hdu.data is None:
                    obj.view(self.meta["sub_pixel_flag"])

                kernel = self.get_kernel(obj).astype(float)

                rot_blur_angle = self.meta["rotational_blur_angle"]
                if abs(rot_blur_angle) > 0:
                    kernel = pu.rotational_blur(kernel, rot_blur_angle)         # makes a copy of kernel

                if self.meta["normalise_kernel"] is True:
                    kernel /= np.sum(kernel)

                image = obj.hdu.data.astype(float)
                ny_old, nx_old = image.shape

                bkg_width = self.meta["bkg_width"]
                if bkg_width < 0:
                    bkg_level = np.median(image)
                elif bkg_width == 0:
                    bkg_level = 0
                else:
                    mask = np.ones((ny_old, nx_old), dtype=np.bool)
                    mask[bkg_width:(ny_old - bkg_width),
                         bkg_width:(nx_old - bkg_width)] = 0
                    bkg_level = np.median(image[mask])

                # y = min(image.shape[0], kernel.shape[0])
                # x = min(image.shape[1], kernel.shape[1])
                # new_image = convolve(image[:y, :x] - bkg_level, kernel[:y, :x], mode=mode)
                mode = self.meta["convolve_mode"]
                new_image = convolve(image - bkg_level, kernel, mode=mode)
                ny_new, nx_new = new_image.shape
                obj.hdu.data = new_image + bkg_level

                # ..todo: careful with which dimensions mean what
                for s in ["", "D"]:
                    if "CRPIX1"+s in obj.hdu.header:
                        obj.hdu.header["CRPIX1"+s] += (nx_new - nx_old) / 2
                        obj.hdu.header["CRPIX2"+s] += (ny_new - ny_old) / 2

        return obj

    def fov_grid(self, which="waveset", **kwargs):
        waveset = []
        if which == "waveset":
            if self._waveset is not None:
                _waveset = self._waveset
                waves = 0.5 * (np.array(_waveset)[1:] +
                               np.array(_waveset)[:-1])
                wave_min = kwargs["wave_min"] if "wave_min" in kwargs else np.min(_waveset)
                wave_max = kwargs["wave_max"] if "wave_max" in kwargs else np.max(_waveset)
                mask = (wave_min < waves) * (waves < wave_max)
                waveset = np.unique([wave_min] + list(waves[mask]) + [wave_max])

        return waveset

    def get_kernel(self, obj):
        self.valid_waverange = None
        if self.kernel is None:
            self.kernel = np.ones((1, 1))
        return self.kernel

    def plot(self, obj=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.gcf().clf()

        kernel = self.get_kernel(obj)
        plt.imshow(kernel, norm=LogNorm(), origin='lower', **kwargs)

        return plt.gcf()

################################################################################
# Analytical PSFs - Vibration, Seeing, NCPAs

class AnalyticalPSF(PSF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["z_order"] = [41, 641]
        self.apply_to_classes = FieldOfViewBase


class Vibration(AnalyticalPSF):
    """
    Creates a wavelength independent kernel image
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["z_order"] = [244, 744]
        self.meta["width_n_fwhms"] = 4
        self.apply_to_classes = ImagePlaneBase

        self.required_keys = ["fwhm", "pixel_scale"]
        utils.check_keys(self.meta, self.required_keys, action="error")
        self.kernel = None

    def get_kernel(self, obj):
        if self.kernel is None:
            utils.from_currsys(self.meta)
            fwhm_pix = self.meta["fwhm"] / self.meta["pixel_scale"]
            sigma = fwhm_pix / 2.35
            width = max(1, int(fwhm_pix * self.meta["width_n_fwhms"]))
            self.kernel = Gaussian2DKernel(sigma, x_size=width, y_size=width,
                                           mode="center").array

        return self.kernel.astype(float)


class NonCommonPathAberration(AnalyticalPSF):
    """
    Needed: pixel_scale
    Accepted: kernel_width, strehl_drift
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["z_order"] = [241, 641]
        self.meta["kernel_width"] = None
        self.meta["strehl_drift"] = 0.02
        self.meta["wave_min"] = "!SIM.spectral.wave_min"
        self.meta["wave_max"] = "!SIM.spectral.wave_max"

        self._total_wfe = None

        self.valid_waverange = [0.1 * u.um, 0.2 * u.um]

        self.required_keys = ["pixel_scale"]
        utils.check_keys(self.meta, self.required_keys, action="error")

    def fov_grid(self, which="waveset", **kwargs):

        if which == "waveset":
            self.meta.update(kwargs)
            self.meta = utils.from_currsys(self.meta)

            min_sr = pu.wfe2strehl(self.total_wfe, self.meta["wave_min"])
            max_sr = pu.wfe2strehl(self.total_wfe, self.meta["wave_max"])

            srs = np.arange(min_sr, max_sr, self.meta["strehl_drift"])
            waves = 6.2831853 * self.total_wfe * (-np.log(srs))**-0.5
            waves = utils.quantify(waves, u.um).value
            waves = (list(waves) + [self.meta["wave_max"]]) * u.um
        else:
            waves = [] * u.um

        return waves

    def get_kernel(self, obj):
        waves = obj.meta["wave_min"], obj.meta["wave_max"]

        old_waves = self.valid_waverange
        wave_mid_old = 0.5 * (old_waves[0] + old_waves[1])
        wave_mid_new = 0.5 * (waves[0] + waves[1])
        strehl_old = pu.wfe2strehl(wfe=self.total_wfe, wave=wave_mid_old)
        strehl_new = pu.wfe2strehl(wfe=self.total_wfe, wave=wave_mid_new)

        if np.abs(1 - strehl_old / strehl_new) > self.meta["strehl_drift"]:
            self.valid_waverange = waves
            self.kernel = pu.wfe2gauss(wfe=self.total_wfe, wave=wave_mid_new,
                                       width=self.meta["kernel_width"])
        return self.kernel

    @property
    def total_wfe(self):
        if self._total_wfe is None:
            if self.table is not None:
                self._total_wfe = pu.get_total_wfe_from_table(self.table)
            else:
                self._total_wfe = 0
        return self._total_wfe

    def plot(self):
        import matplotlib.pyplot as plt
        plt.gcf().clf()

        wave_min, wave_max = utils.from_currsys([self.meta["wave_min"],
                                                 self.meta["wave_max"]])
        waves = np.linspace(wave_min, wave_max, 1001) * u.um
        wfe = self.total_wfe
        strehl = pu.wfe2strehl(wfe=wfe, wave=waves)

        plt.plot(waves, strehl)
        plt.xlabel("Wavelength [{}]".format(waves.unit))
        plt.ylabel("Strehl Ratio \n[Total WFE = {}]".format(wfe))

        return plt.gcf()


class SeeingPSF(AnalyticalPSF):
    """
    Currently only returns gaussian kernel with a ``fwhm`` [arcsec]
    """
    def __init__(self, fwhm=1.5, **kwargs):
        super().__init__(**kwargs)
        self.meta["fwhm"] = fwhm
        self.meta["z_order"] = [242, 642]

    def fov_grid(self, which="waveset", **kwargs):
        wavelengths = []
        if which == "waveset" and \
                "waverange" in kwargs and \
                "pixel_scale" in kwargs:
            waverange = utils.quantify(kwargs["waverange"], u.um)
            wavelengths = waverange
            # ..todo: return something useful

        # .. todo: check that this is actually correct
        return wavelengths

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = utils.quantify(pixel_scale, u.arcsec)
        wave = fov.wavelength

        # add in the conversion to fwhm from seeing and wavelength here
        fwhm = self.meta["fwhm"] * u.arcsec / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array

        return kernel

    def plot(self):
        return super().plot(PoorMansFOV())


class GaussianDiffractionPSF(AnalyticalPSF):
    def __init__(self, diameter, **kwargs):
        super().__init__(**kwargs)
        self.meta["diameter"] = diameter
        self.meta["z_order"] = [242, 642]

    def fov_grid(self, which="waveset", **kwargs):
        wavelengths = []
        if which == "waveset" and \
                "waverange" in kwargs and \
                "pixel_scale" in kwargs:
            waverange = utils.quantify(kwargs["waverange"], u.um)
            diameter = utils.quantify(self.meta["diameter"], u.m).to(u.um)
            fwhm = 1.22 * (waverange / diameter).value  # in rad

            pixel_scale = utils.quantify(kwargs["pixel_scale"], u.deg)
            pixel_scale = pixel_scale.to(u.rad).value
            fwhm_range = np.arange(fwhm[0], fwhm[1], pixel_scale)
            wavelengths = list(fwhm_range / 1.22 * diameter.to(u.m))

        # .. todo: check that this is actually correct
        return wavelengths

    def update(self, **kwargs):
        if "diameter" in kwargs:
            self.meta["diameter"] = kwargs["diameter"]

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = utils.quantify(pixel_scale, u.arcsec)

        wave = 0.5 * (fov.meta["wave_max"] + fov.meta["wave_min"])

        wave = utils.quantify(wave, u.um)
        diameter = utils.quantify(self.meta["diameter"], u.m).to(u.um)
        fwhm = 1.22 * (wave / diameter) * u.rad.to(u.arcsec) / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array

        return kernel

    def plot(self):
        return super().plot(PoorMansFOV())


################################################################################
# Semi-analytical PSFs - AnisoCADO PSFs


class SemiAnalyticalPSF(PSF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["z_order"] = [42]
        self.apply_to_classes = FieldOfViewBase
        # self.apply_to_classes = ImagePlaneBase


class AnisocadoConstPSF(SemiAnalyticalPSF):
    """
    Makes a SCAO on-axis PSF with a desired Strehl ratio at a given wavelength

    To make the PSFs a map connecting Strehl, Wavelength, and residual wavefront
    error is required

    Parameters
    ----------
    filename : str
        Path to Strehl map with axes (x, y) = (wavelength, wavefront error)
    strehl : float
        Desired Strehl ratio. Either percentage [1, 100] or fractional [1e-3, 1]
    wavelength : float
        [um] The given strehl is valid for this wavelength
    psf_side_length : int
        [pixel] Default is 512. Side length of the kernel images
    offset : tuple
        [arcsec] SCAO guide star offset from centre (dx, dy)
    rounded_edges : bool
        Default is True. Sets all halo values below a threshold to zero.
        The threshold is determined from the max values of the edge rows of the
        kernel image

    Other Parameters
    ----------------
    convolve_mode : str
        ["same", "full"] convolution keywords from scipy.signal.convolve

    Examples
    --------
    Add an AnisocadoConstPSF with code::

        from scopesim.effects import AnisocadoConstPSF
        psf = AnisocadoConstPSF(filename="test_AnisoCADO_rms_map.fits",
                                strehl=0.5,
                                wavelength=2.15,
                                convolve_mode="same",
                                psf_side_length=512)

    Add an AnisocadoConstPSF to a yaml file::

        effects:
        -   name: Ks_Stehl_40_PSF
            description: A 40% Strehl PSF over the field of view
            class: AnisocadoConstPSF
            kwargs:
                filename: "test_AnisoCADO_rms_map.fits"
                strehl: 0.5
                wavelength: 2.15
                convolve_mode: full
                psf_side_length: 512

    """
    def __init__(self, **kwargs):
        super(AnisocadoConstPSF, self).__init__(**kwargs)
        params = {"z_order": [42, 652],
                  "psf_side_length": 512,
                  "offset": (0, 0),
                  "rounded_edges": True}
        self.meta.update(params)
        self.meta.update(kwargs)

        self.required_keys = ["filename", "strehl", "wavelength"]
        utils.check_keys(self.meta, self.required_keys, action="error")
        self.nmRms      # check to see if it throws an error

        self._psf_object = None
        self._kernel = None

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        if self._kernel is None:
            if isinstance(fov, FieldOfViewBase):
                pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
            elif isinstance(fov, float):
                pixel_scale = fov

            n = self.meta["psf_side_length"]
            wave = self.wavelength
            self._psf_object = aniso.AnalyticalScaoPsf(pixelSize=pixel_scale,
                                                       N=n, wavelength=wave,
                                                       nmRms=self.nmRms)
            if np.any(self.meta["offset"]):
                self._psf_object.shift_off_axis(self.meta["offset"][0],
                                                self.meta["offset"][1])

            self._kernel = self._psf_object.psf_latest
            if self.meta["rounded_edges"]:
                self._kernel = pu.round_kernel_edges(self._kernel)

        return self._kernel

    def remake_kernel(self, x):
        """
        Remake the kernel based on either a pixel_scale of FieldOfView

        Parameters
        ----------
        x: float, FieldOfView
            [um] if float

        """
        self._kernel = None
        return self.get_kernel(x)

    @property
    def wavelength(self):
        wave = utils.from_currsys(self.meta["wavelength"])
        if isinstance(wave, str) and wave in tu.FILTER_DEFAULTS:
            wave = tu.get_filter_effective_wavelength(wave)
        wave = utils.quantify(wave, u.um).value

        return wave

    @property
    def strehl_ratio(self):
        strehl = None
        if self._psf_object is not None:
            strehl = self._psf_object.strehl_ratio

        return strehl

    @property
    def nmRms(self):
        strehl = utils.from_currsys(self.meta["strehl"])
        wave = self.wavelength
        hdu = self._file[0]
        nm_rms = pu.nmrms_from_strehl_and_wavelength(strehl, wave, hdu)

        return nm_rms

    def plot(self, obj=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.figure(figsize=(10, 10))

        pixel_scale = utils.from_currsys("!INST.pixel_scale")

        kernel = self.get_kernel(pixel_scale)
        # subplot2grid((n_vert, n_horiz), (from_top, from_left), colspan=1, rowspan=1)
        plt.subplot2grid((2, 2), (0, 0))
        im = kernel
        r_sky = pixel_scale * im.shape[0]
        plt.imshow(im, norm=LogNorm(), origin='lower',
                   extent= [-r_sky, r_sky, -r_sky, r_sky], **kwargs)
        plt.ylabel("[arcsec]")

        plt.subplot2grid((2, 2), (0, 1))
        x = kernel.shape[1] // 2
        y = kernel.shape[0] // 2
        r = 16
        im = kernel[y-r:y+r, x-r:x+r]
        r_sky = pixel_scale * im.shape[0]
        plt.imshow(im, norm=LogNorm(), origin='lower',
                   extent= [-r_sky, r_sky, -r_sky, r_sky], **kwargs)
        plt.ylabel("[arcsec]")
        plt.gca().yaxis.set_label_position('right')

        plt.subplot2grid((2, 2), (1, 0), colspan=2)
        hdr = self._file[0].header
        data = self._file[0].data

        hdr = self._file[0].header
        data = self._file[0].data

        wfes = np.arange(hdr["NAXIS1"]) * hdr["CDELT1"] + hdr["CRVAL1"]
        waves = np.arange(hdr["NAXIS2"]) * hdr["CDELT2"] + hdr["CRVAL2"]
        for i in np.arange(len(waves))[::-1]:
            plt.plot(wfes, data[i, :],
                     label="{} $\mu m$".format(round(waves[i], 3)))

        plt.xlabel("RMS Wavefront Error [um]")
        plt.ylabel("Strehl Ratio")
        plt.legend()

        return plt.gcf()



################################################################################
# Discreet PSFs - MAORY and co PSFs


class DiscretePSF(PSF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["z_order"] = [43]
        self.apply_to_classes = FieldOfViewBase
        # self.apply_to_classes = ImagePlaneBase


class FieldConstantPSF(DiscretePSF):
    def __init__(self, **kwargs):
        # sub_pixel_flag and flux_accuracy are taken care of in PSF base class
        super().__init__(**kwargs)

        self.required_keys = ["filename"]
        utils.check_keys(self.meta, self.required_keys, action="error")

        self.meta["z_order"] = [262, 662]
        self._waveset, self.kernel_indexes = pu.get_psf_wave_exts(self._file,
                                                                  self.meta["wave_key"])
        self.current_layer_id = None
        self.current_ext = None
        self.current_data = None
        self.kernel = None

    # def apply_to(self, fov):
    #   Taken care of by PSF base class

    # def fov_grid(self, which="waveset", **kwargs):
    #   Taken care of by PSF base class

    def get_kernel(self, fov):
        # find nearest wavelength and pull kernel from file
        ii = pu.nearest_index(fov.wavelength, self._waveset)
        ext = self.kernel_indexes[ii]
        if ext != self.current_layer_id:
            self.kernel = self._file[ext].data
            self.current_layer_id = ext
            hdr = self._file[ext].header

            # compare kernel and fov pixel scales, rescale if needed
            if "CUNIT1" in hdr:
                unit_factor = u.Unit(hdr["CUNIT1"]).to(u.deg)
            else:
                unit_factor = 1

            kernel_pixel_scale = hdr["CDELT1"] * unit_factor
            fov_pixel_scale = fov.hdu.header["CDELT1"]

            # rescaling kept inside loop to avoid rescaling for every fov
            pix_ratio = kernel_pixel_scale / fov_pixel_scale
            if abs(pix_ratio - 1) > self.meta["flux_accuracy"]:
                self.kernel = pu.rescale_kernel(self.kernel, pix_ratio)

            if fov.header["NAXIS1"] < hdr["NAXIS1"] or \
                    fov.header["NAXIS2"] < hdr["NAXIS2"]:
                self.kernel = pu.cutout_kernel(self.kernel, fov.header)

        return self.kernel

    def plot(self):
        return super().plot(PoorMansFOV())


class FieldVaryingPSF(DiscretePSF):
    """
    kwargs
    ------
    sub_pixel_flag
    flux_accuracy : float
        Default 1e-3. Level of flux conservation during rescaling of kernel
    """
    def __init__(self, **kwargs):
        # sub_pixel_flag and flux_accuracy are taken care of in PSF base class
        super(FieldVaryingPSF, self).__init__(**kwargs)

        self.required_keys = ["filename"]
        utils.check_keys(self.meta, self.required_keys, action="error")

        self.meta["z_order"] = [261, 661]

        ws, ki = pu.get_psf_wave_exts(self._file, self.meta["wave_key"])
        self._waveset, self.kernel_indexes = ws, ki
        self.current_ext = None
        self.current_data = None
        self._strehl_imagehdu = None

    def apply_to(self, fov):
        # .. todo: add in field rotation
        # accept "full", "dit", "none

        # check if there are any fov.fields to apply a psf to
        if len(fov.fields) > 0:
            if fov.hdu.data is None:
                fov.view(self.meta["sub_pixel_flag"])

            old_shape = fov.hdu.data.shape

            # get the kernels that cover this fov, and their respective masks
            # kernels and masks are returned by .get_kernel as a list of tuples
            canvas = None
            kernels_masks = self.get_kernel(fov)
            for kernel, mask in kernels_masks:

                # renormalise the kernel if needs be
                sum_kernel = np.sum(kernel)
                if abs(sum_kernel - 1) > self.meta["flux_accuracy"]:
                    kernel /= sum_kernel

                # image convolution
                image = fov.hdu.data.astype(float)
                kernel = kernel.astype(float)
                new_image = convolve(image, kernel, mode="same")
                if canvas is None:
                    canvas = np.zeros(new_image.shape)

                # mask convolution + combine with convolved image
                if mask is not None:
                    new_mask = convolve(mask, kernel, mode="same")
                    canvas += new_image * new_mask
                else:
                    canvas = new_image

            # reset WCS header info
            new_shape = canvas.shape
            fov.hdu.data = canvas

            # ..todo: careful with which dimensions mean what
            if "CRPIX1" in fov.hdu.header:
                fov.hdu.header["CRPIX1"] += (new_shape[0] - old_shape[0]) / 2
                fov.hdu.header["CRPIX2"] += (new_shape[1] - old_shape[1]) / 2

            if "CRPIX1D" in fov.hdu.header:
                fov.hdu.header["CRPIX1D"] += (new_shape[0] - old_shape[0]) / 2
                fov.hdu.header["CRPIX2D"] += (new_shape[1] - old_shape[1]) / 2

        return fov

    # def fov_grid(self, which="waveset"):
    #   This is taken care of by the PSF base class

    def get_kernel(self, fov):
        # 0. get file extension
        # 1. pull out strehl map for fov header
        # 2. get number of unique psfs
        # 3. pull out those psfs
        # 4. if more than one, make masks for the fov on the fov pixel scale
        # 5. make list of tuples with kernel and mask

        # find which file extension to use - keep a pointer in self.current_data
        fov_wave = 0.5 * (fov.meta["wave_min"] + fov.meta["wave_max"])
        jj = pu.nearest_index(fov_wave, self._waveset)
        ext = self.kernel_indexes[jj]
        if ext != self.current_ext:
            self.current_ext = ext
            self.current_data = self._file[ext].data

        # compare the fov and psf pixel scales
        kernel_pixel_scale = self._file[ext].header["CDELT1"]
        fov_pixel_scale = fov.hdu.header["CDELT1"]

        # get the spatial map of the kernel cube layers
        strl_hdu = self.strehl_imagehdu
        strl_cutout = pu.get_strehl_cutout(fov.hdu.header, strl_hdu)

        # get the kernels and mask that fit inside the fov boundaries
        layer_ids = np.round(np.unique(strl_cutout.data)).astype(int)
        if len(layer_ids) > 1:
            kernels = [self.current_data[ii] for ii in layer_ids]
            masks = [strl_cutout.data == ii for ii in layer_ids]
            self.kernel = [[krnl, msk] for krnl, msk in zip(kernels, masks)]
        else:
            self.kernel = [[self.current_data[layer_ids[0]], None]]

        # .. todo: re-scale kernel and masks to pixel_scale of FOV
        # .. todo: can this be put somewhere else to save on iterations?
        # .. todo: should the mask also be rescaled?
        # rescale the pixel scale of the kernel to match the fov images
        pix_ratio = fov_pixel_scale / kernel_pixel_scale
        if abs(pix_ratio - 1) > self.meta["flux_accuracy"]:
            for ii in range(len(self.kernel)):
                self.kernel[ii][0] = pu.rescale_kernel(self.kernel[ii][0], pix_ratio)

        return self.kernel

    @property
    def strehl_imagehdu(self):
        """ The HDU containing the positional info for kernel layers """
        if self._strehl_imagehdu is None:
            ecat = self._file[0].header["ECAT"]
            if isinstance(self._file[ecat], fits.ImageHDU):
                self._strehl_imagehdu = self._file[ecat]

            # ..todo: impliment this case
            elif isinstance(self._file[ecat], fits.BinTableHDU):
                cat = self._file[ecat]
                self._strehl_imagehdu = pu.make_strehl_map_from_table(cat)

        return self._strehl_imagehdu

    def plot(self):
        return super().plot(PoorMansFOV())