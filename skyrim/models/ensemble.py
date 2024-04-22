import datetime
import torch
from loguru import logger
import xarray as xr
from . import MODEL_FACTORY


class GlobalEnsemble:
    def __init__(self, model_names):
        # validate model names against a predefined factory dictionary
        if not all(name in MODEL_FACTORY for name in model_names):
            missing_models = [name for name in model_names if name not in MODEL_FACTORY]
            raise ValueError(
                f"Models {missing_models} are not available in MODEL_FACTORY."
            )
        self.model_names = model_names
        self._model = None  
        self.common_channels = None

    def _load_model(self, model_name):
        """Load the specified model into GPU memory."""
        logger.debug(f"Loading {model_name} model.")
        self._model = MODEL_FACTORY[model_name][0]()
        self._model.model.to("cuda")
        if self.common_channels is None:
            self.common_channels = set(self._model.out_channel_names)
        else:
            self.common_channels.intersection_update(self._model.out_channel_names)

    def _release_model(self):
        """Release the current model from GPU memory and clear it."""
        model_name = self._model.__class__.__name__
        logger.debug(f"Releasing {model_name} model.")
        self._model.model.to("cpu")
        torch.cuda.empty_cache()
        del self._model
        self._model = None

    def _ensemble_predictions(self, predictions):
        """Average predictions along shared channels."""
        # TODO: check these xr.arrays' memory usage
        
        logger.debug("Ensembling predictions along shared channels.")
        filtered_preds = [
            pred.sel(channel=list(self.common_channels)) for pred in predictions
        ]
        if not filtered_preds:
            raise ValueError(
                "No predictions to average or no common channels available."
            )

        # Convert list of xarray.DataArray to a single DataArray for averaging
        combined = xr.concat(filtered_preds, dim="model")
        averaged = combined.mean(dim="model")
        return averaged

    def predict_one_step(self, start_time: datetime.datetime, save: bool = False):
        """Subclasses should implement this method."""
        raise NotImplementedError

    def rollout(
        self, start_time: datetime.datetime, n_steps: int = 3, save: bool = True
    ):
        """Perform a rollout for all models, aggregating predictions and managing resources."""
        # average the prediction in final_predictions list along the shared channels

        output_paths = []
        predictions = []  # keeps the final step predictions for each model

        for model_name in self.model_names:
            self._load_model(model_name)
            try:
                pred, paths = self._model.rollout(start_time, n_steps, save)
                output_paths.extend(paths)
                predictions.append(pred)
            finally:
                self._release_model()

        # Average the predictions along shared channels
        averaged_prediction = self._ensemble_predictions(predictions)
        return averaged_prediction, output_paths
