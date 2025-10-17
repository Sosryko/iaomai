import pytest
import pandas as pd
from pathlib import Path

from iaomai.outages import preprocess_outages_data

MIN_OUTAGE_DURATION = pd.Timedelta(days=0)
DATA_PATH = Path("../data/unavailability")


@pytest.fixture(scope="package")
def swiss_outages():
    raw_data = pd.read_csv("unavailability_ch_20160101_20310101.csv", parse_dates=True)
    return preprocess_outages_data(
        raw_data=raw_data, outage_duration=MIN_OUTAGE_DURATION
    )


@pytest.fixture(scope="package")
def french_outages():
    raw_data = pd.read_csv("unavailability_fr_20160101_20310101.csv", parse_dates=True)
    return preprocess_outages_data(
        raw_data=raw_data, outage_duration=MIN_OUTAGE_DURATION
    )
