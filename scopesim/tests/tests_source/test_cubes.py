import pytest
import os
import inspect

import numpy as np
import astropy.units as u
from astropy.io import fits

import synphot

from scopesim.tests.mocks.py_objects.source_objects import _make_dummy_cube
from scopesim.source.cube import convert_flux_units, Cube
from scopesim.source.source_utils import photons_in_range


def mock_dir():
    cur_dirname = os.path.dirname(inspect.getfile(inspect.currentframe()))
    rel_dirname = "../mocks/files/"
    return os.path.abspath(os.path.join(cur_dirname, rel_dirname))


MOCK_DIR = mock_dir()


class TestCube:
    def test_if_working(self):
        dummy_cube = _make_dummy_cube(scale=0.2, wave_unit=u.AA, ref_wave=1000*u.AA,
                                      wave_step=1, wave_type="WAVE", bunit="erg / (s cm2 Angstrom)")
        cube = Cube(hdu=dummy_cube)

        assert isinstance(cube, Cube)

    def test_load_from_file(self):
        dummy_cube = _make_dummy_cube(scale=0.2, wave_unit=u.AA, ref_wave=1000 * u.AA,
                                      wave_step=1, wave_type="WAVE", bunit="erg / (s cm2 Angstrom)")

        filename = os.path.join(MOCK_DIR, "dummy_cube.fits")
        dummy_cube.writeto(filename, overwrite=True)

        cube = Cube(filename=filename, ext=1)

        assert isinstance(cube, Cube)

    def test_waves_input_output_waves(self):
        wave_unit = u.AA
        dummy_cube = _make_dummy_cube(scale=0.2, wave_unit=wave_unit, ref_wave=1000 * u.AA,
                                      wave_step=1, wave_type="WAVE", bunit="erg / (s cm2 Angstrom)")
        cube = Cube(hdu=dummy_cube)
        in_waves = cube._in_waves.to(u.um)
        out_waves = cube.waves

        assert cube._in_waves.unit == wave_unit
        assert cube.waves.unit == u.Unit('um')
        assert in_waves.value.all() == out_waves.value.all()

    def test_easy_flux_conversion(self):
        ref_wave = 1000 * u.AA
        bunit = "erg / (s cm2 Angstrom)"
        wave_step = 1*u.AA
        dummy_cube = _make_dummy_cube(scale=0.2, wave_unit=u.AA, ref_wave=ref_wave,
                                      wave_step=wave_step, wave_type="WAVE", bunit=bunit)

        cube = Cube(hdu=dummy_cube)

        nphot1 = cube.photons_in_range()

        lam = cube._in_waves
        flux = np.ones(lam.shape) * u.Unit(bunit)

        sp = synphot.SourceSpectrum(synphot.Empirical1D, points=lam, lookup_table=flux )
        nphot2 = photons_in_range([sp], 0.1000, 0.1100) * cube.data.shape[1] * cube.data.shape[2]

        nphot2 = nphot2.value
        print(nphot1, nphot2[0])
        assert np.isclose(nphot1, nphot2[0], rtol=0.1)

    def test_conversion(self):
        ref_wave = 1 * u.Hz
        bunit = "Jansky"
        wave_step = 1*u.Hz
        cube = _make_dummy_cube(scale=0.2, wave_unit=u.Hz, ref_wave=ref_wave,
                                wave_step=wave_step, wave_type="FREQ", bunit=bunit)

        cube = Cube(hdu=cube)
        nphot1 = cube.photons_in_range()

        data = cube.data
        lam = cube._in_waves
        flux = np.ones(lam.shape) * u.Unit(bunit)
        sp = synphot.SourceSpectrum(synphot.Empirical1D, points=lam, lookup_table=flux )
        nphot2 = photons_in_range([sp], 1000, 1100) * 400
        nphot2 = nphot2.value
        print(nphot1, nphot2)

        assert np.isclose(nphot1, nphot2[0], rtol=0.1)
