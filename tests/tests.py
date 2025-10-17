import pytest
import numpy as np
import pandas as pd

from iaomai.outages import aggregate_outages


@pytest.mark.parametrize(
    "raw_data",
    [
        pytest.lazy_fixture("swiss_outages"),
        pytest.lazy_fixture("french_outages"),
    ],
)
def test_outages_per_production_unit(data: str):
    """
    Outages per production unit at any time t should never exceed
    the maximum recorded nominal capacity in the dataset
    """
    max_nominal_power = (
        data[["nominal_power", "production_resource_psr_name"]]
        .groupby("production_resource_psr_name")
        .max()
        .T
    )
    outages_per_unit = aggregate_outages(filtered_data=data, by="production_unit")
    # before, check that no production unit was left out in the aggregation step
    assert set(max_nominal_power.columns) == set(outages_per_unit.columns)
    assert (
        (
            pd.DataFrame(
                np.broadcast_to(max_nominal_power, outages_per_unit.shape),
                index=outages_per_unit.index,
                columns=max_nominal_power.columns,
            )
            >= outages_per_unit
        )
        .all()
        .all()
    )
