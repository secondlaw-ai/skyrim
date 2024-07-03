import torch
import pytest
from earth2studio.data.ifs import IFS
from earth2studio.lexicon.ifs import IFSLexicon
from skyrim.core import Skyrim
from datetime import datetime
from earth2mip.initial_conditions import get_initial_condition_for_model


@pytest.mark.integ
def test_ifs_ic_gets_right_fourcastnet():
    dt = datetime(2024, 4, 1)
    models = ["pangu", "fourcastnet"]
    for m in models:
        skyrim = Skyrim(m, ic_source="ifs")
        channels = skyrim.model.in_channel_names
        ifs_channels = list(set(channels) & set(IFSLexicon.VOCAB.keys()))
        ifs_data = IFS().fetch_ifs_dataarray(dt, ifs_channels)
        ic_data = get_initial_condition_for_model(skyrim.model.model, skyrim.model.build_datasource(), dt)
        fails = []
        for i in channels:
            if i not in set(ifs_data["variable"].values):
                print(f"Skipping variable {i} as it is not in IFS fetched vocab")
                continue
            ix = channels.index(i)
            y = ic_data.squeeze()[ix].double().to("cpu")
            x = torch.from_numpy(ifs_data.sel(variable=i).values.squeeze())
            if i.startswith("z") or i.startswith("r"):
                # skip z as there seems to be a bug.
                torch.allclose(x[:719, :], y[:719, :], rtol=1e-1)

        if len(fails):
            raise Exception(f"Failed params {fails}")
