import datetime
import torch
from loguru import logger
import xarray as xr
from .base import GlobalPrediction
from ...common import OUTPUT_DIR
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
        self.common_channels = None
        self._model = None

    @property
    def time_step(self):
        # TODO: fix this hardcoded value :)
        return datetime.timedelta(hours=6)

    def __repr__(self) -> str:
        return f"GlobalEnsemble({self.model_names})"

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
        # TODO: check if this works with graphcast with jax backend

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
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        save: bool = True,
        save_config: dict = {},
    ):
        """Perform a rollout for all models, aggregating predictions and managing resources."""
        # TODO: seperate model predictions should be deleted after final ens calculation?

        output_paths = []  # keeps the paths of the individual model predictions
        predictions = []  # keeps the final step predictions for each model

        for model_name in self.model_names:
            self._load_model(model_name)
            try:
                pred, paths = self._model.rollout(
                    start_time=start_time,
                    n_steps=n_steps,
                    save=save,
                    output_dir=output_dir,
                )
                output_paths.extend(paths)
                predictions.append(pred)
            finally:
                self._release_model()

        # Average the predictions along shared channels
        averaged_prediction = self._ensemble_predictions(predictions)

        if save:
            logger.debug("Caculating and saving ensemble predictions.")
            ens_output_paths = self._save_ensembled_outputs(
                output_paths, n_steps, output_dir
            )
        return averaged_prediction, ens_output_paths

    def _save_ensembled_outputs(self, output_paths, n_steps, output_dir):
        ens_prefix = "_".join(sorted(self.model_names))
        ens_directory = output_dir / ens_prefix
        ens_directory.mkdir(exist_ok=True)  # Ensure directory exists
        ens_output_paths = []
        for s in range(n_steps):
            step_paths = output_paths[s::n_steps]
            _, source, start_time, end_time = step_paths[0].stem.split("__")

            # Combine data arrays into a single dataset for ensemble
            preds = [xr.open_dataarray(p) for p in step_paths]
            ens_pred = self._ensemble_predictions(preds)

            file_path = (
                ens_directory / f"{ens_prefix}__{source}__{start_time}__{end_time}.nc"
            )
            ens_output_paths.append(file_path)
            ens_pred.to_netcdf(file_path)
        return ens_output_paths


class GlobalEnsemblePrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
