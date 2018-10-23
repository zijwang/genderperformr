import unidecode
from .data.consts import *
import torch
import numpy as np


def parse_model_parameters(params: str) -> dict:
    args = {}
    params_list = params.split("-")
    assert len(params_list) >= 10
    args["batch_size"] = int(params_list[0])
    args["is_bidirection"] = int(params_list[2])
    args["emb_out_size"] = int(params_list[3])
    args["lstm_hidden_size"] = int(params_list[4])
    args["lstm_layers"] = int(params_list[5])
    args["lstm_dropout"] = float(params_list[6])
    args["lstm_out_size"] = int(params_list[7])
    args["opt_type"] = params_list[8]
    args["opt_lr"] = float(params_list[9])
    return args


def prep_name(username: str) -> list:
    """
        username should always be in ascii. anything other than those in ascii will be filtered by unidecode
    """
    tensor = [EMB.get(i, EMB["<other>"]) for i in unidecode.unidecode_expect_ascii(username).replace("[?]", "")]
    if len(tensor) == 0:
        tensor = [EMB["<other>"]]
    return tensor


def prep_tensor(username_list: list):
    if len(username_list) > USERNAME_LEN:
        return torch.LongTensor([username_list[:USERNAME_LEN]]), torch.LongTensor([USERNAME_LEN])
    else:
        username_len = len(username_list)
        return torch.cat([torch.LongTensor(username_list),
                          torch.zeros(USERNAME_LEN - username_len, dtype=torch.long)]).unsqueeze(0), \
               torch.LongTensor([len(username_list)])


def get_prediction(prob: float) -> str:
    return "M" if prob < 0.1 else "F" if prob > 0.9 else "N" if 0.45 < prob < 0.55 else ""


def get_prediction_list(prob_list: np.ndarray) -> list:
    return [get_prediction(prob) for prob in prob_list]
