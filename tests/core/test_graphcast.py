import pytest
from skyrim.core import Skyrim

model = Skyrim("graphcast")


@pytest.mark.integ
def test_output_mapping():
    # TODO: define a data container validation for prediction.
    pred, _ = model.predict("20240404", "0000", 6, save=False)
    assert set(pred.prediction.dims) == {"time", "channel", "lat", "lon"}
    breakpoint()
    print("done")
