import numpy as np
import pandas as pd
import pandera as pa
from pathlib import Path


def preprocess_installed_generation_data(path_to_file: Path) -> pd.DataFrame:
    temp_generation_data = pd.read_csv(path_to_file)
    temp_generation_data = temp_generation_data[
        ["Year", "Production Type", "Installed Capacity (MW)"]
    ]
    temp_generation_data = temp_generation_data[
        ~temp_generation_data["Installed Capacity (MW)"].isin(("n/e", "-"))
    ].dropna()
    temp_generation_data["Installed Capacity (MW)"] = pd.to_numeric(
        temp_generation_data["Installed Capacity (MW)"]
    )
    temp_generation_data = temp_generation_data.pivot_table(
        values="Installed Capacity (MW)", columns="Production Type", index="Year"
    )
    temp_generation_data.index = pd.to_datetime(temp_generation_data.index, format="%Y")
    return temp_generation_data


def preprocess_outages_data(
    raw_data: pd.DataFrame,
    outage_duration: pd.Timedelta | None,
    generation_type: list[str] = ["Nuclear"],
) -> pd.DataFrame:
    """
    Preprocessing of raw ENTSOE outage data

    Args:
        raw_data (pd.DataFrame)
        outage_duration (pd.Timedelta | None):
            Function filters out outages with duration below `outage_duration` argument
        generation_type (str, optional)

    Returns:
        pd.DataFrame:
    """
    raw_data["end"] = pd.to_datetime(raw_data["end"], utc=True).dt.tz_convert("CET")
    raw_data["start"] = pd.to_datetime(raw_data["start"], utc=True).dt.tz_convert("CET")
    # some outages are duplicated in the dataframe with all fields identical but the time resolution
    raw_data = raw_data.drop(columns=["resolution"]).drop_duplicates()
    # give unique id to each outage
    # easier to read than mrid
    # TODO: go back to mrid if outages for all countries stored in the same space
    raw_data["unique_id"] = np.arange(0, raw_data.shape[0])

    if generation_type not in raw_data["plant_type"].unique():
        raise ValueError(
            f"Available generation types are {raw_data['plant_type'].unique().tolist()}"
        )
    # filter outages of nuclear generation units
    filter_generation_type = raw_data["plant_type"] == "Nuclear"
    # filter cancelled outages, maintained outages have a None field
    filter_cancelled = raw_data["docstatus"].isna()
    # filter outages shorter than x days
    filter_duration = (
        (raw_data["end"] - raw_data["start"] > outage_duration)
        if outage_duration is not None
        else True
    )
    filtered_data = raw_data.loc[
        filter_generation_type & filter_cancelled & filter_duration,
        (
            "nominal_power",
            "start",
            "end",
            "businesstype",
            "avail_qty",
            "unique_id",
            "production_resource_psr_name",
            "mrid",
        ),
    ]
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data["delta"] = filtered_data["nominal_power"] - pd.to_numeric(
        filtered_data["avail_qty"]
    )
    return filtered_data.drop_duplicates()


def aggregate_outages(filtered_data: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Aggregates ENTSOE outage data

    General idea is to transform a list of outages with a start date, end date and produciton per row
    into a timeseries of available capacity


    Args:
        filtered_data (pd.DataFrame)
        by (str): Options:
            - "total"
            - "production_unit"
            - "businesstype"

    Returns:
        pd.DataFrame
    """
    pivot_data = filtered_data.pivot_table(
        values="delta",
        index=("start", "end"),
        columns=("unique_id", "businesstype", "production_resource_psr_name"),
    )
    # outage start and end date in same index
    # list of outages to timeseries "moment"
    pivot_data = pd.concat(
        [
            pivot_data.droplevel("start", axis=0),
            pivot_data.droplevel("end", axis=0),
        ],
        axis=0,
    )
    # after previous steps, some indices will be duplicated because
    # for example, some outages start at the same moment for different units
    # remove duplicated elements in index by grouping by date and taking the max
    # max: because outages can overlap but each time a new availability is given
    pivot_data = (
        pivot_data.groupby(pivot_data.index)
        .max()
        .replace({0: np.nan})
        .sort_index()
        .ffill(limit_area="inside", axis=0)
    )
    # max: because outages can overlap but for each new outage a new availability is given
    # conservative side: take the max unavailability value
    pivot_data = (
        pivot_data.droplevel(["unique_id"], axis=1)
        .T.groupby(
            [
                pivot_data.columns.get_level_values("production_resource_psr_name"),
                pivot_data.columns.get_level_values("businesstype"),
            ]
        )
        .max()
    )
    match by:
        case "total":
            return pivot_data.T.sum(axis=1)
        case "production_unit":
            return (
                pivot_data.groupby(
                    pivot_data.index.get_level_values("production_resource_psr_name")
                )
                .max()
                .T
            )
        case "businesstype":
            return (
                pivot_data.groupby(pivot_data.index.get_level_values("businesstype"))
                .sum()
                .T
            )
        case _:
            raise ValueError(
                "Only implemented aggregations are 'total', 'businesstype' (planned/unplanned), 'production_unit'"
            )
