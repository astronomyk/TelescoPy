import os
import shutil
import pytest

from scopesim import rc
from scopesim.commands.user_commands import UserCommands
from scopesim.server.database import download_package

LOCAL_PKGS_PATH = "./scopesim_pkg_dir_tmp/"
rc.__config__["!SIM.file.local_packages_path"] = os.path.abspath(LOCAL_PKGS_PATH)


def setup_module():
    if not os.path.exists(LOCAL_PKGS_PATH):
        os.mkdir(LOCAL_PKGS_PATH)
    download_package("instruments/test_package.zip")


def teardown_module():
        shutil.rmtree(LOCAL_PKGS_PATH)


class TestInit:
    def test_initialise_with_nothing(self):
        assert isinstance(UserCommands(), UserCommands)

    def test_initialised_when_passed_a_dict_of_properties(self):
        cmd = UserCommands(properties={"!OBS.dit": 60, "!ATMO.pwv": 9001})
        assert cmd["!ATMO.pwv"] > 9000

    def test_initialised_when_passed_a_instrument_yaml_dict(self):
        cmd = UserCommands(yamls=[{"alias": "ATMO",
                                   "properties": {"pwv": 9001}}])
        assert cmd["!ATMO.pwv"] > 9000

    def test_initialised_when_passed_a_list_of_yaml_names(self):
        cmd = UserCommands(packages=["test_package"],
                           yamls=["test_telescope.yaml"])
        assert cmd["!TEL.temperature"] > 9000

    def test_initialised_when_combining_yaml_dict_filename_properties(self):
        cmd = UserCommands(packages=["test_package"],
                           yamls=["test_telescope.yaml",
                                  {"alias": "ATMO",
                                   "properties": {"pwv": 9001}}],
                           properties={"!ATMO.pwv": 8999})
        assert cmd["!TEL.temperature"] > 9000
        assert cmd["!ATMO.pwv"] < 9000

    def test_initialise_with_correct_keywords(self):
        cmd = UserCommands(packages=["test_package"],
                           yamls=["test_instrument.yaml"],
                           properties={"!ATMO.life": 42})
        assert isinstance(cmd, UserCommands)
        assert cmd["!ATMO.life"] == 42
        assert cmd["!INST.pixel_scale"] == 0.5

    def test_initialised_with_filename_for_default_file(self):
        cmd = UserCommands(packages=["test_package"], yamls=["default.yaml"])
        assert cmd["!TEL.temperature"] < 9000
        assert len(cmd.yaml_dicts) == 7     # 3 yamls filenames + default

    def test_initialised_with_use_instrument(self):
        cmd = UserCommands(use_instrument="test_package")
        assert cmd["!TEL.temperature"] < 9000
        assert len(cmd.yaml_dicts) == 7     # 3 yamls filenames + default

    def test_mode_yamls(self):
        yamls = [{"alias": "OBS", "properties": {"modes": ["mode1"],
                                                 "life": 9001}}]
        mode_yamls = [{"name": "mode1",
                       "yamls": [{"alias": "OBS",
                                  "properties": {"life": 42}}]}]
        cmd = UserCommands(yamls=yamls, mode_yamls=mode_yamls)
        assert cmd["!OBS.life"] == 42

        print(cmd.list_modes())

    def test_throws_error_for_wrong_mode_name(self):
        yamls = [{"alias": "OBS", "properties": {"modes": ["mode1"],
                                                 "life": 9001}}]
        mode_yamls = [{"name": "mode1",
                       "yamls": [{"alias": "OBS",
                                  "properties": {"life": 42}}]}]
        with pytest.raises(ValueError):
            UserCommands(yamls=yamls, mode_yamls=mode_yamls, set_modes=["mode2"])

    def test_mode_yamls_read_from_file(self):
        cmd = UserCommands(use_instrument="test_package")
        assert cmd["!TEL.temperature"] < 9000
        assert cmd["!OBS.airmass"] == 2
        assert cmd.yaml_dicts[-1]["effects"][0]["kwargs"]["meaning_of_life"] == 42


class TestMiscFeatures:
    def test_updates_with_yaml_dict(self):
        yaml_input = {"alias": "TEL",
                      "properties": {"temperature": 8999}}
        cmd = UserCommands(use_instrument="test_package")
        cmd.update(yamls=[yaml_input])
        assert cmd["!TEL.temperature"] < 9000

    def test_update_works_via_setitem(self):
        cmd = UserCommands(use_instrument="test_package")
        cmd["!TEL.gigawatts"] = 1.21
        assert cmd["!TEL.gigawatts"] == 1.21


class TestListLocalPackages:
    def test_all_packages_listed(self):
        from scopesim.commands import user_commands as uc2
        real_pkgs, ext_pkgs = uc2.list_local_packages(action="return")
        assert len(real_pkgs) > 0


class TestTrackIpAddress:
    def test_see_if_theres_an_entry_on_the_server_log_file(self):
        cmds = UserCommands(use_instrument="test_package")


