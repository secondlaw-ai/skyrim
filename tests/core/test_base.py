from dataclasses import dataclass
from skyrim.libs.ic import IfsDatasource
from skyrim.core.models.base import GlobalModel


# TODO: build a timeloop
@dataclass
class BoringModel:
    in_channel_names: list
    out_channel_names: list
    time_step: int = 6


class BoringGlobalModel(GlobalModel):
    model_name = "boring"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return BoringModel(in_channel_names=["u10"], out_channel_names=["u10"])


def test_get_initial_conditions():
    # wip, test channels.
    model = BoringGlobalModel(ic_source="ifs")
    assert isinstance(model.data_source, IfsDatasource)
