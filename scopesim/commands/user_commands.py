import os
import warnings
import copy

import numpy as np
import yaml
import requests

from .. import rc
from ..utils import find_file

__all__ = ["UserCommands"]


class UserCommands:
    """
    Contains all the setting that a user may wish to alter for an optical train

    Most of the important settings are kept in the ``.cmds`` system dictionary
    Setting can be accessed by using the alias names. Currently these are:

    - ATMO: atmospheric and observatory location settings
    - TEL: telescope related settings
    - RO: relay optics settings, i.e. between telescope and instrument
    - INST: instrument optics settings
    - DET: detector settings
    - OBS: observation settings, and
    - SIM: simulation settings

    All of the settings are contained in a special ``SystemDict`` dictionary
    that allows the user to access all the settings via a bang-string (!). E.g::

        cmds = UserCommands()
        cmds["!SIM.file.local_packages_path]

    .. note::

       To use this format for accessing hierarchically-stored values, the bang
       string must always begin with a "!"

    Alternatively the same value can be accessed via the normal dictionary
    format. E.g::

        cmds["SIM"]["file"]["local_packages_path"]


    Parameters
    ----------
    use_instrument : str, optional
        The name of the main instrument to use

    packages : list, optional
        list of package names needed for the optical system, so that ScopeSim
        can find the relevant files. E.g. ["Armazones", "ELT", "MICADO"]

    yamls : list, optional
        list of yaml filenames that are needed for the combined optical system
        E.g. ["MICADO_Standalone_RO.yaml", "MICADO_H4RG.yaml", "MICADO_.yaml"]

    properties : dict, optional
        Any extra "OBS" properties that should be added

    ignore_effects : list
        Not yet implemented

    add_effects : list
        Not yet implemented

    override_effect_values : dict
        Not yet implemented


    Attributes
    ----------
    cmds : SystemDict
        Built from the ``properties`` dictionary of a yaml dictionary. All
        values here are accessible globally by all ``Effects`` objects in an
        ``OpticalTrain`` once the ``UserCommands`` has been passed to the
        ``OpticalTrain``.

    yaml_dicts : list of dicts
        Where all the effects dictionaries are stored


    Examples
    --------
    Here we use a combination of the main parameters: ``packages``, ``yamls``,
    and ``properties``. When not using the ``use_instrument`` key, ``packages``
    and ``yamls`` must be specified, otherwise scopesim will not know which
    where to look for yaml files (only relevant if reading in yaml files)::

        >>> from scopesim.server.database import download_package
        >>> from scopesim.commands import UserCommands
        >>>
        >>> download_package("test_package")
        >>> cmd = UserCommands(packages=["test_package"],
        ...                    yamls=["test_telescope.yaml",
        ...                           {"alias": "ATMO",
        ...                            "properties": {"pwv": 9001}}],
        ...                    properties={"!ATMO.pwv": 8999})

    Notes
    -----
    .. attention:: We track your IP address when ``ScopeSim`` checks for updates

        When initialising a UserCommands object via ``use_instrument=``,
        ``ScopeSim`` checks on the database whether there are updates to the
        instrument package. Our server records the IP address of each query for
        out own statistics only.

        WE DO NOT STORE OR TRACK PERSONAL DATA. THESE STATISTICS ARE NEEDED FOR
        GETTING MORE FUNDING TO CONTINUE DEVELOPING THIS PROJECT.

        We are doing this solely as a way of showing the austrian funding agency
        that people are indeed using this software (or not). Your participation
        in this effort greatly helps our chances of securing the next grant.

        However, if you would still like to avoid your IP address being stored,
        you can run ``scopesim`` 100% anonymously by setting::

            >>> scopsim.rc.__config__["!SIM.reports.ip_tracking"] = True

        at the beginning of each session. Alternatively you can also pass the
        same bang keyword when generating a ``UserCommand`` object::

            >>> from scopesim import UserCommands
            >>> UserCommands(use_instrument="MICADO",
            >>>              properties={"!SIM.reports.ip_tracking": False})

        If you use a custom ``yaml`` configuration file, you can also add this
        keyword to the ``properties`` section of the ``yaml`` file.

    """

    def __init__(self, **kwargs):

        self.cmds = copy.deepcopy(rc.__config__)
        self.yaml_dicts = []
        self.kwargs = kwargs
        self.ignore_effects = []
        self.package_name = ""
        self.default_yamls = []
        self.modes_dict = {}

        self.update(**kwargs)

    def update(self, **kwargs):
        """
        Updates the current parameters with a yaml dictionary

        See the ``UserCommands`` main docstring for acceptable kwargs
        """

        if "use_instrument" in kwargs:
            self.package_name = kwargs["use_instrument"]
            self.update(packages=[kwargs["use_instrument"]],
                        yamls=["default.yaml"])

            check_for_updates(self.package_name)

        if "packages" in kwargs:
            add_packages_to_rc_search(self["!SIM.file.local_packages_path"],
                                      kwargs["packages"])

        if "yamls" in kwargs:
            for yaml_input in kwargs["yamls"]:
                if isinstance(yaml_input, str):
                    yaml_file = find_file(yaml_input)
                    if yaml_file is not None:
                        yaml_dict = load_yaml_dicts(yaml_file)
                        self.update(yamls=yaml_dict)
                        if yaml_input == "default.yaml":
                            self.default_yamls = yaml_dict
                    else:
                        warnings.warn("{} could not be found".format(yaml_input))

                elif isinstance(yaml_input, dict):
                    self.cmds.update(yaml_input)
                    self.yaml_dicts += [yaml_input]

                    for key in ["packages", "yamls", "mode_yamls"]:
                        if key in yaml_input:
                            self.update(**{key: yaml_input[key]})

                else:
                    raise ValueError("yaml_dicts must be a filename or a "
                                     "dictionary: {}".format(yaml_input))

        if "mode_yamls" in kwargs:
            # Convert the yaml list of modes to a dict object
            self.modes_dict = {my["name"]: my for my in kwargs["mode_yamls"]}
            if "modes" in self.cmds["!OBS"]:
                if not isinstance(self.cmds["!OBS.modes"], list):
                    self.cmds["!OBS.modes"] = [self.cmds["!OBS.modes"]]
                for mode_name in self.cmds["!OBS.modes"]:
                    mode_yaml = self.modes_dict[mode_name]
                    self.update(yamls=[mode_yaml])

        if "set_modes" in kwargs:
            self.set_modes(modes=kwargs["set_modes"])

        if "properties" in kwargs:
            self.cmds.update(kwargs["properties"])

        if "ignore_effects" in kwargs:
            self.ignore_effects = kwargs["ignore_effects"]

        if "add_effects" in kwargs:
            # ..todo: implement this
            pass

        if "override_effect_values" in kwargs:
            # ..todo: implement this
            pass

    def set_modes(self, modes=None):
        if not isinstance(modes, list):
            modes = [modes]
        for defyam in self.default_yamls + self.yaml_dicts:
            if "properties" in defyam and "modes" in defyam["properties"]:
                defyam["properties"]["modes"] = []
                for mode in modes:
                    if mode in self.modes_dict:
                        defyam["properties"]["modes"] += [mode]
                    else:
                        raise ValueError("mode '{}' was not recognised"
                                         "".format(mode))

        self.__init__(yamls=self.default_yamls)

    def list_modes(self):
        if isinstance(self.modes_dict, dict):
            modes = {}
            for mode_name in self.modes_dict:
                dic = self.modes_dict[mode_name]
                desc = dic["description"] if "description" in dic else "<None>"
                modes[mode_name] = desc

            msg = "\n".join(["{}: {}".format(key, modes[key]) for key in modes])
        else:
            msg = "No modes found"
        return msg

    @property
    def modes(self):
        print(self.list_modes())

    def __setitem__(self, key, value):
        self.cmds.__setitem__(key, value)

    def __getitem__(self, item):
        return self.cmds.__getitem__(item)

    def __contains__(self, item):
        return self.cmds.__contains__(item)

    def __repr__(self):
        return self.cmds.__repr__()


def check_for_updates(package_name):
    """
    Asks IRDB server if there are newer versions of the instrument package
    """
    response = {}

    # tracking **exclusively** your IP address for our internal stats
    if rc.__currsys__["!SIM.reports.ip_tracking"] and \
            "TRAVIS" not in os.environ:
        front_matter = rc.__currsys__["!SIM.file.server_base_url"]
        back_matter = "api.php?package_name={}".format(package_name)
        try:
            response = requests.get(url=front_matter+back_matter).json()
        except:
            print("Offline. Cannot check for updates for {}"
                  "".format(package_name))
    return response


def add_packages_to_rc_search(local_path, package_list):
    """
    Adds the paths of a list of locally saved packages to the search path list

    Parameters
    ----------
    local_path : str
        Where the pacakges are located. Generally given by the value in
        scopesim.rc.__config__["!SIM.file.local_packages_path"]

    package_list : list
        A list of the package names to add

    """

    for pkg in package_list:
        pkg_dir = os.path.abspath(os.path.join(local_path, pkg))
        if not os.path.exists(pkg_dir):
            # todo: keep here, but add test for this by downloading test_package
            # raise ValueError("Package could not be found: {}".format(pkg_dir))
            warnings.warn("Package could not be found: {}".format(pkg_dir))

        if pkg_dir in rc.__search_path__:
            # if package is already in search_path, move it to the first place
            ii = np.where(np.array(rc.__search_path__) == pkg_dir)[0][0]
            rc.__search_path__.pop(ii)

        rc.__search_path__ = [pkg_dir] + rc.__search_path__


def load_yaml_dicts(filename):
    """
    Loads one or more dicts stored in a YAML file under ``filename``

    Parameters
    ----------
    filename : str
        Path to the YAML file

    Returns
    -------
    yaml_dicts : list
        A list of dicts

    """

    yaml_dicts = []
    with open(filename) as f:
        yaml_dicts += [dic for dic in yaml.load_all(f)]

    return yaml_dicts


def list_local_packages(action="display"):
    """
    Lists the packages on the local disk that ScopeSim can find

    Packages can only be found in the directory listed under::

        scopesim.rc.__config__["!SIM.file.local_packages_path"]

    Packages are divided into "main" packages and "extension" packages.

    - Main packages contain a ``default.yaml`` file which tell ScopeSim which
      other packages are required to generate the full optical system
    - Extension packages contain only the data files needed to support the
      effects listed in the package YAML file

    .. note::
       Only "main" packages can be passed to a UserCommands object using the
       ``use_instrument=...`` parameter

    Parameters
    ----------
    action : str, optional
        ["display", "return"] What to do with the output.
        - "display": the list of packages are printed to the screen
        - "return": package names are returned in lists

    Returns
    -------
    main_pkgs, ext_pkgs : lists
        If action="return": Lists containing the names of locally saved packages

    """

    local_path = os.path.abspath(rc.__config__["!SIM.file.local_packages_path"])
    pkgs = [d for d in os.listdir(local_path) if
            os.path.isdir(os.path.join(local_path, d))]

    main_pkgs = [pkg for pkg in pkgs if
                 os.path.exists(os.path.join(local_path, pkg, "default.yaml"))]
    ext_pkgs = [pkg for pkg in pkgs if not
                os.path.exists(os.path.join(local_path, pkg, "default.yaml"))]

    if action == "display":
        msg = "\nLocal package directory:\n  {}\n" \
              "Full packages [can be used with 'use_instrument=...']\n  {}\n" \
              "Support packages\n  {}".format(local_path, main_pkgs, ext_pkgs)
        print(msg)
    else:
        return main_pkgs, ext_pkgs
