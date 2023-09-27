"""Module containing an interface to trained PyTorch model."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from dice_ml.constants import ModelTypes
from dice_ml.utils.exception import SystemException
from dice_ml.model_interfaces.base_model import BaseModel


class LgbmModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='', dtypes=None, func=None, kw_args=None):
        """Init method

        :param model: trained LGBM Model.
        :param model_path: path to trained model.
        :param backend: 
        :param dtypes: used to manually set a pandas DataFrame column types before a get_output() prediction is made in the {"model": "lgbm_model.LgbmModel"}
        :param func: function transformation required for ML model. If func is None, then func will be the identity function.
        :param kw_args: Dictionary of additional keyword arguments to pass to func. DiCE's data_interface is appended to the
                        dictionary of kw_args, by default.
        """

        super().__init__(model, model_path, backend, dtypes, func, kw_args)

    def load_model(self):
        if self.model_path != '':
            self.model = lgb.Booster(model_file="data/hle_model_lgbm.json")

            if self.model_type == ModelTypes.Classifier:
                self.model.params['objective'] = 'classification'
            elif self.model_type == ModelTypes.Regressor:
                self.model.params['objective'] = 'regression'
            
    def get_output(self, input_instance, model_score=True,
                   transform_data=False, out_tensor=False):
        """returns prediction probabilities

        :param input_tensor: test input.
        :param transform_data: boolean to indicate if data transformation is required.
        """
        input_instance = self.transformer.transform(input_instance)

        # Fix type errors by assigning the the original data's DataFrame Types
        if isinstance(input_instance, pd.DataFrame) and self.dtypes is not None:
            input_instance = input_instance.astype(self.dtypes)

        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.model.predict_proba(input_instance)
            else:
                return self.model.predict(input_instance)
        else:
            return self.model.predict(input_instance)

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        temp_input = np.transpose(np.array([np.random.uniform(0, 1) for i in range(inp_size)]).reshape(-1, 1))
        return self.get_output(temp_input).shape[1]

    def get_num_output_nodes2(self, input_instance):
        if self.model_type == ModelTypes.Regressor:
            raise SystemException('Number of output nodes not supported for regression')
        return self.get_output(input_instance).shape[1]
