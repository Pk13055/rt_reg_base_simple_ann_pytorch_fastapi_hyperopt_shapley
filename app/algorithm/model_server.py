import numpy as np, pandas as pd
import os
from shap import Explainer
import json
import sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.regressor as regressor


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "regressionBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5

    def _get_preprocessor(self):
        self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        self.model = regressor.load_model(self.model_path)
        return self.model

    def predict(self, data):

        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X = proc_data["X"].values.astype(np.float)
        # make predictions
        preds = model.predict(pred_X)
        # inverse transform the predictions to original scale
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)
        # return te prediction df with the id and prediction fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = np.round(preds, 4)

        return preds_df

    def _get_predictions(self, X):
        model = self._get_model()
        preds = model.predict(X)
        return preds

    def _get_scaled_predictions(self, X):
        model = self._get_model()
        preprocessor = self._get_preprocessor()
        preds = model.predict(X)
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)
        return preds

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))
        # ------------------------------------------------------------------------------
        # original class predictions

        model = self._get_model()
        X_columns = list(proc_data["X"].columns)
        pred_X = proc_data["X"].values.astype(np.float)
        ids = proc_data["ids"]

        # pred_values = model.predict(pred_X)
        pred_values = self._get_scaled_predictions(pred_X)

        # ------------------------------------------------------------------------------
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        # create the shapley explainer
        mask = np.zeros_like(pred_X)
        explainer = Explainer(self._get_scaled_predictions, mask, seed=1)
        # Get local explanations
        shap_values = explainer(pred_X)
        print(shap_values)

        # ------------------------------------------------------------------------------
        # create pd dataframe of explanation scores
        N = pred_X.shape[0]
        explanations = []
        for i in range(N):
            samle_expl_dict = {}
            samle_expl_dict[self.id_field_name] = ids[i]
            samle_expl_dict["predictions"] = pred_values[i]
            samle_expl_dict["baseline"] = shap_values.base_values[i]

            feature_impacts = {}
            for f_num, feature in enumerate(shap_values.feature_names):
                feature_impacts[X_columns[f_num]] = round(
                    shap_values.values[i][f_num], 4
                )

            samle_expl_dict["feature_impacts"] = feature_impacts
            explanations.append(samle_expl_dict)

        # ------------------------------------------------------
        """
        To plot the shapley values:
        you can only plot one sample at a time. 
        if you want to plot all samples. create a loop and use the index (sample_idx)
        """
        # sample_idx = 4
        # shap_values.base_values = shap_values.base_values[sample_idx]
        # shap_values.values = shap_values.values[sample_idx]
        # shap_values.data = shap_values.data[sample_idx]
        # shap.plots.waterfall(shap_values)
        # ------------------------------------------------------
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
