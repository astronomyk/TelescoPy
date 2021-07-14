'''Tests for SpectralSurface'''
# pylint: disable=no-self-use,missing-class-docstring
# pylint: disable=missing-function-docstring

# 1 read in a table
# 2 compliment the table based on columns in file
# 3 have @property methods for: transmission, ermission, reflection

import inspect
import os
import sys
import warnings

import pytest

import numpy as np
from astropy.table import Table, Column
from astropy import units as u
from astropy.io import ascii as ioascii

from synphot import SpectralElement, SourceSpectrum
from synphot.models import BlackBody1D
from synphot.models import Empirical1D
from synphot.units import PHOTLAM

from scopesim.optics import surface as opt_surf
from scopesim.optics import surface_utils as surf_utils
from scopesim import utils


def mock_dir():
    cur_dirname = os.path.dirname(inspect.getfile(inspect.currentframe()))
    rel_dirname = "../mocks/MICADO_SCAO_WIDE/"

    return os.path.abspath(os.path.join(cur_dirname, rel_dirname))


MOCK_DIR = mock_dir()


@pytest.fixture(name="ter_table", scope="class")
def fixture_ter_table():
    return ioascii.read(os.path.join(MOCK_DIR, "TER_dichroic.dat"))


@pytest.fixture(name="input_tables", scope="module")
def fixture_input_tables():
    filenames = ["TER_dichroic.dat", "TC_filter_Ks.dat"]
    abs_paths = [os.path.join(MOCK_DIR, fname) for fname in filenames]

    return abs_paths


@pytest.fixture(name="unity_flux", scope="module")
def fixture_unity_flux():
    flux = np.ones(100)
    wave = np.logspace(-1, 1, 100) * u.um

    return flux, wave


@pytest.mark.usefixtures("input_tables")
class TestSpectralSurfaceInit:
    def test_can_exist_with_no_input(self):
        srf = opt_surf.SpectralSurface()
        assert isinstance(srf, opt_surf.SpectralSurface)

    def test_reads_in_table_which_exists(self, input_tables):
        srf = opt_surf.SpectralSurface(filename=input_tables[0])
        assert isinstance(srf.table, Table)

    def test_returns_empty_table_if_path_is_bogus(self):
        srf = opt_surf.SpectralSurface(filename="bogus.txt")
        assert isinstance(srf, opt_surf.SpectralSurface)
        assert len(srf.table) == 0


@pytest.mark.usefixtures("input_tables")
class TestSpectralSurfaceWavelengthProperty:
    def test_returns_quantity_array_from_file(self, input_tables):
        srf = opt_surf.SpectralSurface(filename=input_tables[0])
        assert isinstance(srf.wavelength, u.Quantity)
        assert srf.wavelength.unit == u.um

    def test_returns_quantity_array_from_file_with_no_unit(self, input_tables):
        srf = opt_surf.SpectralSurface(filename=input_tables[1])
        assert isinstance(srf.wavelength, u.Quantity)
        assert srf.wavelength.unit == u.um

    def test_returns_quantity_if_wavelength_overridden_by_list(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0])
        assert np.all(srf.wavelength == [0.3, 3.0]*u.um)

    def test_returns_quantity_if_wavelength_unit_overridden_by_quantity(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0],
                                       wavelength_unit=u.Angstrom)
        assert srf.meta["wavelength_unit"] == u.Angstrom
        assert np.all(srf.wavelength == [0.3, 3.0]*u.Angstrom)

    def test_returns_quantity_if_wavelength_unit_overridden_by_string(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0],
                                       wavelength_unit="Angstrom")
        assert srf.meta["wavelength_unit"] == u.Angstrom
        assert np.all(srf.wavelength == [0.3, 3.0]*u.Angstrom)


@pytest.mark.usefixtures("input_tables")
class TestSpectralSurfaceTransmissionProperty:
    def test_returns_synphot_object_array_from_file(self, input_tables):
        srf = opt_surf.SpectralSurface(filename=input_tables[0])
        assert isinstance(srf.transmission, SpectralElement)

    def test_returns_synphot_object_when_given_arrays(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0]*u.um,
                                       transmission=[1, 1])
        assert isinstance(srf.transmission, SpectralElement)

    def test_returns_synphot_object_if_only_reflection_is_given(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0]*u.um,
                                       reflection=[1, 1])
        assert isinstance(srf.transmission, SpectralElement)

    def test_returns_none_when_none_of_the_three_TER_info_is_given(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0]*u.um)
        assert srf.transmission is None

    def test_returns_none_when_no_wavelength_info_given(self):
        srf = opt_surf.SpectralSurface(transmission=[1, 1])
        assert srf.transmission is None


class TestSpectralSurfaceEmissionProperty:
    def test_returns_synphot_object_per_arcsec2_scaled_to_per_arcsec2(self):
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0] * u.um,
                                       emission=[1, 1] * PHOTLAM * u.arcsec**-2)
        assert isinstance(srf.emission, SourceSpectrum)
        assert np.all(srf.emission([0.3, 3.0] * u.um) == [1, 1] * PHOTLAM)

    def test_returns_synphot_object_per_steradian_scaled_to_per_arcsec2(self):
        sr2arcsec = u.sr.to(u.arcsec**2)
        srf = opt_surf.SpectralSurface(wavelength=[0.3, 3.0] * u.um,
                                       emission=[sr2arcsec]*2 * PHOTLAM / u.sr)
        assert isinstance(srf.emission, SourceSpectrum)
        assert np.all(srf.emission([0.3, 3.0] * u.um) == [1, 1] * PHOTLAM)

    def test_returned_bb_curve_is_scaled_to_per_arcsec2(self):
        n = 11
        sr2arcsec = u.sr.to(u.arcsec ** 2)
        wave = np.logspace(-1, 3, n) * u.um
        temp = (0 * u.deg_C).to(u.Kelvin, equivalencies=u.temperature())
        srf = opt_surf.SpectralSurface(wavelength=wave,
                                       transmission=np.zeros(n),
                                       temperature=temp)
        emission_raw = SourceSpectrum(BlackBody1D, temperature=temp)

        assert isinstance(srf.emission, SourceSpectrum)
        assert np.allclose(emission_raw(wave) / sr2arcsec,
                           srf.emission(wave))


class TestSpectralSurfaceComplimentArray:
    @pytest.mark.parametrize("colname1, colname2, col1, col2, expected",
                             [("A", "B", [0.8]*u.um, [0.1]*u.um, [0.1]*u.um),
                              ("A", "B", [0.8]*u.um, None,       [0.2]*u.um),
                              ("A", "B", None,       [0.8]*u.um, [0.2]*u.um)])
    def test_the_right_answers_for_valid_input(self, colname1, colname2,
                                               col1, col2, expected):
        srf = opt_surf.SpectralSurface()
        srf.meta[colname1] = col1
        srf.meta[colname2] = col2
        col3 = srf._compliment_array(colname1, colname2)
        if sys.version_info.major >= 3:
            assert np.all(np.isclose(col3.data, expected.data))
            assert col3.unit == expected.unit
        else:
            warnings.warn("Data equality isn't tested for 2.7")
            assert col3.unit == expected.unit

    @pytest.mark.parametrize("colname1, colname2, col1, col2, expected",
                             [("A",     "B",      None, None, None)])
    def test_returns_none_for_none_input(self, colname1, colname2,
                                         col1, col2, expected):
        srf = opt_surf.SpectralSurface()
        srf.meta[colname1] = col1
        srf.meta[colname2] = col2
        col3 = srf._compliment_array(colname1, colname2)
        assert col3 is None

    @pytest.mark.parametrize("col2_arr, expected",
                             [([0.1],      [0.1]),
                              ([0.1, 0.1], [0.1, 0.1]),
                              ([0.1, 0.1, 0.1], [0.1, 0.1, 0.1]),
                              (None,       [0.2])])
    def test_returns_right_answers_for_valid_table(self, col2_arr, expected):
        srf = opt_surf.SpectralSurface()
        srf.table.add_column(Column(name="col1", data=[0.8]*len(expected)))
        if col2_arr:
            srf.table.add_column(Column(name="col2", data=col2_arr))

        col3 = srf._compliment_array("col1", "col2")
        if sys.version_info.major >= 3:
            assert col3.data == pytest.approx(expected)
            assert len(col3.data) == len(expected)
        else:
            warnings.warn("Data equality isn't tested for 2.7")


class TestSpectralSurfaceAreaProperty:
    def test_returns_area_when_area_in_meta_dict(self):
        srf = opt_surf.SpectralSurface(area=1*u.m**2)
        assert srf.area == 1 * u.m**2

    def test_returns_area_for_only_outer_in_meta_dict(self):
        srf = opt_surf.SpectralSurface(outer=1*u.m)
        assert srf.area == (1/4.) * np.pi * u.m ** 2

    def test_returns_area_for_outer_and_inner_in_meta_dict(self):
        srf = opt_surf.SpectralSurface(outer=1*u.m, inner=0.5*u.m)
        assert srf.area == (1/4. - 1/16.) * np.pi * u.m ** 2

    def test_returns_none_when_no_info_in_meta(self):
        srf = opt_surf.SpectralSurface()
        assert srf.area is None


class TestQuantify:
    @pytest.mark.parametrize("item, unit, expected",
                             [(1, "m", 1*u.m),
                              (1, u.m, 1*u.m),
                              ([1], u.m, [1] * u.m),
                              (1 * u.m, u.m, 1 * u.m),
                              ([1] * u.m, u.m, [1] * u.m)])
    def test_return_quantity_for_valid_data(self, item, unit, expected):
        assert utils.quantify(item, unit) == expected


@pytest.mark.usefixtures("unity_flux")
class TestMakeEmissionFromArray:
    @pytest.mark.parametrize("emission_unit",
                             ["ph s-1 m-2 um-1",
                              "ph s-1 m-2 Hz-1",
                              "erg s-1 m-2 um-1",
                              "jansky"])
    def test_source_spectrum_returned_for_synphot_units(self, emission_unit,
                                                        unity_flux):
        meta = {"emission_unit": emission_unit}
        out = surf_utils.make_emission_from_array(*unity_flux, meta=meta)
        assert isinstance(out, SourceSpectrum)

    @pytest.mark.parametrize("emission_unit",
                             ["ph s-1 m-2 um-1 arcsec-2",
                              "jansky sr-1"])
    def test_source_spectrum_returned_for_solid_angles(self, emission_unit,
                                                       unity_flux):
        meta = {"emission_unit": emission_unit}
        out = surf_utils.make_emission_from_array(*unity_flux, meta=meta)
        assert isinstance(out, SourceSpectrum)

    @pytest.mark.parametrize("emission_unit",
                             ["ph s-1 m-2",
                              "ph s-1 m-2 bin-1",
                              "erg s-1 m-2 arcsec-2",
                              "erg s-1 m-2 bin-1 sr-1"])
    def test_source_spectrum_returned_for_bin_units(self, emission_unit,
                                                    unity_flux):
        meta = {"emission_unit": emission_unit}
        out = surf_utils.make_emission_from_array(*unity_flux, meta=meta)
        assert isinstance(out, SourceSpectrum)


@pytest.mark.usefixtures("input_tables")
class TestMakeEmissionFromEmissivity:
    # .. todo:: write this test class
    @pytest.mark.parametrize("temp", [273, 273*u.deg_C])
    def test_source_spectrum_returned_for_temp(self, input_tables, temp):
        srf = opt_surf.SpectralSurface(filename=input_tables[0])
        out = surf_utils.make_emission_from_emissivity(273, srf.emissivity)
        assert isinstance(out, SourceSpectrum)
        assert out.model.temperature_0 == 273


    @pytest.mark.parametrize("temp", [283, 283*u.deg_C, 283*u.Kelvin])
    def test_blackbody_maximum_agrees_with_wien(self, temp):
        '''Check the maximum of emission against Wien's law for photon rate'''
        emissivity = SpectralElement(Empirical1D, points=[1, 20],
                                     lookup_table=[1., 1.])
        flux = surf_utils.make_emission_from_emissivity(temp, emissivity)
        dlam = 0.1
        wave = np.arange(3, 20, dlam) * u.um
        wavemax = wave[np.argmax(flux(wave))]
        if isinstance(temp, u.Quantity):
            temp = temp.to(u.Kelvin, equivalencies=u.temperature()).value
        wienmax = 3669.7 * u.um / temp
        assert np.abs(wavemax - wienmax.to(u.um)) < dlam * u.um


class TestNormaliseBinnedFlux:
    # .. todo:: write this test class
    def test_returns_correct_normalisation(self):
        pass


@pytest.mark.usefixtures("ter_table")
class TestIntegration:
    @pytest.mark.parametrize("col_name",
                             ["transmission", "emissivity", "reflection"])
    def test_ter_property_of_object_from_file(self, col_name, ter_table):
        filename = os.path.join(MOCK_DIR, "TER_dichroic.dat")
        surf = opt_surf.SpectralSurface(filename=filename)

        tbl_ter_prop = ter_table[col_name]
        surf_ter_prop = getattr(surf, col_name).model.lookup_table

        assert isinstance(surf, opt_surf.SpectralSurface)
        assert np.all(surf_ter_prop == tbl_ter_prop)

    @pytest.mark.parametrize("col_name",
                             ["transmission", "emissivity", "reflection"])
    def test_ter_property_of_object_from_arrays(self, col_name, ter_table):
        surf = opt_surf.SpectralSurface(wavelength=ter_table["wavelength"],
                                        wavelength_unit="um")
        surf.meta[col_name] = ter_table[col_name]

        tbl_ter_prop = ter_table[col_name]
        surf_ter_prop = getattr(surf, col_name).model.lookup_table

        assert isinstance(surf, opt_surf.SpectralSurface)
        assert np.all(surf_ter_prop == tbl_ter_prop)

    def test_return_emission_curve_from_file(self):
        filename = os.path.join(MOCK_DIR, "emission_file.dat")
        surf = opt_surf.SpectralSurface(filename=filename)
        integal = surf.emission.integrate().to(u.Unit("ph s-1 m-2"))
        assert np.isclose(integal.value, 2)   # ph s-1 m-2

    def test_return_emission_curve_from_basic_arrays_with_all_keywords(self):
        surf = opt_surf.SpectralSurface(wavelength=[0.5, 2.5],
                                        emission=[1., 1.],
                                        wavelength_unit="um",
                                        emission_unit="ph s-1 m-2 um-1")
        integal = surf.emission.integrate().to(u.Unit("ph s-1 m-2"))
        assert np.isclose(integal.value, 2)  # ph s-1 m-2

    @pytest.mark.parametrize("n, expected",
                             [(2, 2 * u.Unit("ph s-1 m-2")),
                              (10, 2 * u.Unit("ph s-1 m-2")),
                              (100, 2 * u.Unit("ph s-1 m-2"))])
    def test_return_emission_curve_from_quantity_arrays(self, n, expected):
        wavelength = np.linspace(0.5, 2.5, n)*u.um
        emission = [1] * n * u.Unit("ph s-1 m-2 um-1")
        surf = opt_surf.SpectralSurface(wavelength=wavelength,
                                        emission=emission)
        integal = surf.emission.integrate().to(u.Unit("ph s-1 m-2"))
        assert np.isclose(integal.value, expected.value)  # ph s-1 m-2

    @pytest.mark.parametrize("n, unit_str, expected",
                             [(2, "ph s-1 m-2 bin-1", 2 * u.Unit("ph s-1 m-2")),
                              (10, "ph s-1 m-2",     10 * u.Unit("ph s-1 m-2")),
                              (100, "ph s-1 m-2",  100 * u.Unit("ph s-1 m-2"))])
    def test_return_emission_curve_from_binned_quantity_arrays(self, n,
                                                               unit_str,
                                                               expected):
        wavelength = np.linspace(0.5, 2.5, n)*u.um
        emission = [1] * n * u.Unit(unit_str)
        surf = opt_surf.SpectralSurface(wavelength=wavelength,
                                        emission=emission)
        integal = surf.emission.integrate().to(u.Unit("ph s-1 m-2"))
        assert np.isclose(integal.value, expected.value)  # ph s-1 m-2


class TestGetMetaQuantity:
    @pytest.mark.parametrize("dic, unit, expected",
                 [({"area": 1},                               "m2", 1*u.m**2),
                  ({"area": 1,         "area_unit": u.cm**2}, "m2", 1*u.cm**2),
                  ({"area": 1,         "area_unit": "cm2"},   "m2", 1*u.cm**2),
                  ({"area": 1*u.mm**2, "area_unit": "m2"},    "m2", 1*u.mm**2)])
    def test_returns_quantity_for_all_valid_inputs(self, dic, unit, expected):
        quant = utils.get_meta_quantity(dic, "area", unit)
        assert quant == expected

    def test_raise_error_when_key_not_in_dict(self):
        with pytest.raises(KeyError):
            utils.get_meta_quantity({}, "area", u.um ** 2)
