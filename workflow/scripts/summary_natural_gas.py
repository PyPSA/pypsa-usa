"""
Module for summarizing natural gas results.
"""

from typing import List

import constants
import pandas as pd
import pypsa

CODE_2_STATE = {v: k for k, v in constants.STATE_2_CODE.items()}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to States.
    """
    return df.rename(
        columns={
            x: CODE_2_STATE[x.split(" ")[0]] if x.split(" ")[0] in CODE_2_STATE else x
            for x in df.columns
        },
    )


def get_gas_demand(
    n: pypsa.Network,
    carriers: list[str] = ["CCGT", "OCGT"],
) -> pd.DataFrame:
    """
    Get energy sources attached to gas buses.

    This in input energy required (ie. not applying efficiency losses)
    """

    links = n.links[n.links.carrier.isin(carriers)]
    links_t = n.links_t.p0[links.index]
    links_t = links_t.T.groupby(n.links.carrier).sum().T
    return _rename_columns(links_t)


def get_imports_exports(n: pypsa.Network, international: bool = True) -> pd.DataFrame:
    """
    Gets gas flow into and out of the interconnect.
    """

    regex = ".{2}(?:MX|AB|BC|MB|NB|NL|NT|NS|NU|ON|PE|QC|SK|YT)"

    def get_import_export(df: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Input data must be stores dataframe.
        """
        assert direction in ("import", "export")
        return df[df.carrier == f"gas {direction}"]

    def get_international(df: pd.DataFrame) -> pd.DataFrame:
        """
        Input data must be stores dataframe.
        """
        return df[df.bus.str.contains(regex)]

    def get_domestic(df: pd.DataFrame) -> pd.DataFrame:
        """
        Input data must be stores dataframe.
        """
        return df[~df.bus.str.contains(regex)]

    df = n.stores.copy()

    imports = get_import_export(df, "import")
    exports = get_import_export(df, "export")

    if international:
        imports = get_international(imports)
        exports = get_international(exports)
    else:
        imports = get_domestic(imports)
        exports = get_domestic(exports)

    imports_t = n.stores_t.e[imports.index].sum(axis=1)
    exports_t = n.stores_t.e[exports.index].sum(axis=1)

    imports_t.name = "Imports"
    exports_t.name = "Exports"

    return pd.concat([imports_t, exports_t], axis=1)


def get_gas_processing(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets timeseries gas processing.
    """
    processing = n.generators[n.generators.carrier == "gas"]
    processing = n.generators_t.p[processing.index]
    return _rename_columns(processing)


def get_linepack(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets linepack data.
    """
    stores = n.stores[n.stores.carrier == "gas pipeline"]
    stores = n.stores_t.e[stores.index]
    return _rename_columns(stores)


def get_underground_storage(n: pypsa.Network) -> pd.DataFrame:
    """
    Gets underground storage data.
    """
    stores = n.stores[n.stores.carrier == "gas storage"]
    stores = n.stores_t.e[stores.index]
    return _rename_columns(stores)
