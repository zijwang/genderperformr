import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from .model_new import GenderPerformrModel as NewGenderPerformrModel
from .model import GenderPerformrModel
from .utils import *
from .dataset import GenderPerformrDataset


class GenderPerformr:
    def __init__(self, is_new_model=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 is_paralleled = False):
        """
        :param is_new_model: whether to use the newly trained model with torch 0.4 API and better partitioned dataset
                    (in supplementary material), or the model used for the main paper (default)
        :param device: either `torch.device("cuda")` (default if cuda is available)
                        or `torch.device("cpu")` (default if cuda is not available)
        :param is_paralleled: whether to use torch.nn.DataParallel or not (default). Use only when predicting large amount of data
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if is_new_model:
            model_path = os.path.join(os.path.dirname(__file__), "data", "model_new.pth.tar")
        else:
            model_path = os.path.join(os.path.dirname(__file__), "data", "model.pth.tar")

        model_dict = torch.load(model_path, map_location=self.device)
        model_params_dict = parse_model_parameters(model_dict["params"])

        if is_new_model:
            self.model = NewGenderPerformrModel(device=self.device, **model_params_dict)
        else:
            self.model = GenderPerformrModel(device=self.device, **model_params_dict)
        self.model.load_state_dict(model_dict["state_dict"])

        if is_paralleled:
            if self.device.type == "cpu":
                print("Data parallel is not available with cpus")
            else:
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, username, **kwargs):
        """
            Predict the gender of a single username or a list of usernames

            :param username: str or list of str
            :return: a tuple of raw probability in [0, 1]
            and the label of the prediction denoted in "M" (Male), "F" (Female), "N" (Neutral) or empty string for all others
            or a tuple of two lists of raw probabilities and labels if the input is a list of str
        """

        if isinstance(username, str):
            return self._predict_single(username)
        elif isinstance(username, list):
            return self._predict_list(username, **kwargs)
        else:
            assert ("Type %s not support" % type(username))

    def _predict_single(self, username: str):
        """
        Predict the gender of a single username
        It is not recommended to use this API to predict multiple usernames (instead, use `predict_list`)
        :param username: the username to be predicted
        :return: a tuple of a raw probability in [0, 1]
        and the label of the prediction denoted in "M" (Male), "F" (Female), "N" (Neutral) or empty string for all others
        """
        data = [i.to(self.device) for i in prep_tensor(prep_name(username))]
        with torch.no_grad():
            pred = self.model.forward(data)
        raw_prediction = np.exp(pred.detach().cpu().numpy()[0][1])
        return raw_prediction, get_prediction(raw_prediction)

    def _predict_list(self, username_list: list, batch_size=256, num_workers=0):
        """
        Predict the gender of a list of usernames
        :param username_list: a list of usernames to be predicted
        :param batch_size: an int suggesting the batch size for dataloader (default: 128)
        :param num_workers: 0 or a positive int number suggesting number of workers for DataLoader (default: 0)
        :return: a tuple of a list of raw probabilities in [0, 1]
        and a list of labels of the prediction denoted in "M" (Male), "F" (Female), "N" (Neutral) or empty string for all others
        """
        _dataset = GenderPerformrDataset(username_list)
        _dataloader = DataLoader(dataset=_dataset,
                                 batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        raw_predictions = np.array([])
        with torch.no_grad():
            for _data in _dataloader:
                _data = [i.to(self.device) for i in _data]
                pred = self.model.forward(_data)
                raw_predictions = np.append(raw_predictions, np.exp(pred.detach().cpu().numpy())[:, 1])
        return raw_predictions, get_prediction_list(raw_predictions)
