from .pangu import PanguModel, PanguPrediction
from .fourcastnet import FourcastnetModel, FourcastnetPrediction
from .fourcastnet_v2 import FourcastnetV2Model, FourcastnetV2Prediction
from .dlwp import DLWPModel, DLWPPrediction
from .graphcast import GraphcastModel, GraphcastPrediction


MODEL_FACTORY = {
    "pangu": (PanguModel, PanguPrediction),
    "fourcastnet": (FourcastnetModel, FourcastnetPrediction),
    "fourcastnet_v2": (FourcastnetV2Model, FourcastnetV2Prediction),
    "dlwp": (DLWPModel, DLWPPrediction),
    "graphcast": (GraphcastModel, GraphcastPrediction),
}
