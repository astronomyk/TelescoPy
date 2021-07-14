# actually for Source

import os

from astropy.io import fits
from astropy import units as u
from scopesim.tests.mocks.py_objects.source_objects import _make_dummy_cube

import scopesim as sim
from scopesim.source.source import Source

import matplotlib.pyplot as plt

MOCK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "../mocks/files/"))
sim.rc.__search_path__.insert(0, MOCK_DIR)

PLOTS = False


class TestSourceInitWithCube:
    def test_accepts_raw_image_hdu(self):
        """
        Testing basics
        """

        cube = _make_dummy_cube(scale=0.2 * u.arcsec,
                                wave_unit=u.AA, ref_wave=1000*u.AA,
                                wave_step=1*u.AA, wave_type="WAVE",
                                bunit="erg / (s angstrom cm2 arcsec2)")

        src_cube = Source(cube=cube)

        if PLOTS:
            src_cube.plot()
            plt.show()

        assert isinstance(src_cube.fields[0], fits.ImageHDU)

    def test_accepts_multiple_cubes_as_hdu_list(self):
        """
        Testing basics
        """

        cube = _make_dummy_cube(scale=0.2 * u.arcsec,
                                wave_unit=u.AA, ref_wave=1000*u.AA, wave_step=1*u.AA, wave_type="WAVE",
                                bunit="erg / (s angstrom cm2 arcsec2)")

        cube_list = fits.HDUList(hdus=[cube, cube])

        src_cube = Source(cube=cube_list, ext=1)

        if PLOTS:
            src_cube.plot()
            plt.show()

        assert isinstance(src_cube.fields[0], fits.ImageHDU)

    def test_file_to_cube(self):
        """
        Testing basics
        """

        cube = _make_dummy_cube(scale=0.2 * u.arcsec,
                                wave_unit=u.AA, ref_wave=1000*u.AA, wave_step=1*u.AA, wave_type="WAVE",
                                bunit="erg / (s angstrom cm2 arcsec2)")

        filename = os.path.join(MOCK_DIR, "cube.fits")
        cube.writeto(filename, overwrite=True)

        src_cube = Source(cube=filename, ext=1)

        if PLOTS:
            src_cube.plot()
            plt.show()

        assert isinstance(src_cube.fields[0], fits.ImageHDU)

