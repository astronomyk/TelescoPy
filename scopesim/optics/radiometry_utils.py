from collections import OrderedDict
from copy import deepcopy
import warnings

import numpy as np
from astropy import units as u
from astropy.io import ascii as ioascii
from astropy.table import Table, vstack

from .surface import SpectralSurface
from ..utils import real_colname, insert_into_ordereddict, quantify, \
    change_table_entry, convert_table_comments_to_dict


def combine_emissions(tbl, surfaces, row_indexes, etendue, use_area=False):
    if len(tbl) == 0:
        return None

    etendue = quantify(etendue, "m2 arcsec2")

    r_name = real_colname("name", tbl.colnames)
    r_action = real_colname("action", tbl.colnames)

    emission = None
    for ii, row_num in enumerate(row_indexes):
        row = tbl[row_num]
        surf = surfaces[row[r_name]]
        action_attr = row[r_action]

        if isinstance(surf, SpectralSurface):
            surf_throughput = getattr(surf, action_attr)
            if emission is not None:
                emission = emission * surf_throughput

            area = surf.area
            if area is not None:
                surf_emission = surf.emission       # PHOTLAM * arcsec**-2
                surf_eff_area = area * np.cos(surf.mirror_angle)
                surf_eff_solid_angle = (etendue / surf_eff_area).to(u.arcsec**2)
                surf_emission *= surf_eff_solid_angle.value

                if use_area:
                    surf_emission *= area.to(u.cm**2).value
                    surf_emission.meta["use_area"] = use_area

                msg = "Etendue scale factor applied. Effective pixel solid " \
                      "angle for surface is {}".format(surf_eff_solid_angle)
                surf_emission.meta["solid_angle"] = None
                surf_emission.meta["history"] += [msg]

                if ii == 0:
                    emission = deepcopy(surf_emission)
                else:
                    emission = emission + surf_emission

            else:
                warnings.warn('Ignoring emission from surface: "{}". Area came '
                              'back as "None"'.format(surf.meta["name"]))

    return emission


def combine_throughputs(tbl, surfaces, rows_indexes):
    if len(tbl) == 0:
        return None

    r_name = real_colname("name", tbl.colnames)
    r_action = real_colname("action", tbl.colnames)

    throughput = None
    for ii, row_num in enumerate(rows_indexes):

        row = tbl[row_num]
        surf = surfaces[row[r_name]]
        action_attr = row[r_action]
        if action_attr == "":
            raise ValueError("No action in surf.meta: {}".format(surf.meta))

        if isinstance(surf, SpectralSurface):
            surf_throughput = getattr(surf, action_attr)

            if ii == 0:
                throughput = deepcopy(surf_throughput)
            else:
                throughput = throughput * surf_throughput

    return throughput


def combine_tables(new_tables, old_table=None, prepend=False):
    if isinstance(old_table, str):
        old_table = string_to_table(old_table)

    if isinstance(new_tables, (str, Table)):
        new_tables = [new_tables]

    for new_table in new_tables:
        if isinstance(new_table, str):
            new_table = string_to_table(new_table)
        if old_table is None:
            old_table = new_table
        else:
            if prepend:
                old_table = vstack([new_table, old_table])
            else:
                old_table = vstack([old_table, new_table])

    return old_table


def string_to_table(tbl):
    tbl = ioascii.read(tbl, fast_reader=False)
    meta_dict = convert_table_comments_to_dict(tbl)
    tbl.meta.update(meta_dict)

    return tbl


def add_surface_to_table(tbl, surf, name, position, silent=True):
    if position < 0:
        position += len(tbl) + 1

    # here is why we need a deepcopy of the first table
    # no idea why the False works, but it does. Don't screw with working code!
    new_tbl = tbl.copy(False)
    new_tbl.insert_row(position)
    for colname in new_tbl.colnames:
        surf_col = real_colname(colname, surf.meta)
        if surf_col is not None:
            surf_val = surf.meta[surf_col]
            if isinstance(surf_val, u.Quantity):
                surf_val = surf_val.value
            new_tbl = change_table_entry(new_tbl, colname, surf_val,
                                         position=position)
        else:
            if not silent:
                warnings.warn("{} was not found in the meta dictionary of {}. "
                              "This could cause problems".format(colname, name))

    colname = real_colname("name", new_tbl.colnames)
    new_tbl = change_table_entry(new_tbl, colname, name, position=position)

    return new_tbl


def add_surface_to_dict(dic, surf, name, position=0):
    new_entry = OrderedDict({name : surf})
    dic = insert_into_ordereddict(dic, new_entry, position)

    return dic


def make_surface_dict_from_table(tbl):
    surf_dict = OrderedDict({})
    if tbl is not None and len(tbl) > 0:
        names = tbl[real_colname("name", tbl.colnames)]
        for ii in range(len(tbl)):
            surf_dict[names[ii]] = make_surface_from_row(tbl[ii], **tbl.meta)

    return surf_dict


def make_surface_from_row(row, **kwargs):
    row_dict = {colname.lower(): row[colname] for colname in row.colnames}
    kwargs.update(row_dict)
    surface = SpectralSurface(**kwargs)

    return surface

