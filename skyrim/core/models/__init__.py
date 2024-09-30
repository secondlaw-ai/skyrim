from .pangu import PanguModel
from .fourcastnet import FourcastnetModel
from .fourcastnet_v2 import FourcastnetV2Model
from .dlwp import DLWPModel
from .graphcast import GraphcastModel
from .fuxi import FuxiModel
from .fengwu import FengwuModel

MODELS = {
    "pangu": PanguModel,
    "fourcastnet": FourcastnetModel,
    "fourcastnet_v2": FourcastnetV2Model,
    "dlwp": DLWPModel,
    "graphcast": GraphcastModel,
    "fuxi": FuxiModel,
    "fengwu": FengwuModel,
}
