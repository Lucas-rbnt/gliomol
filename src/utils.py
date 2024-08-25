# Standard libraries
from typing import Any, Dict, List

# Third-party libraries
import pandas as pd
import torch.nn as nn
import torch
import monai

def make_data_dict(dataframe: 'pd.DataFrame', tumor: bool = False) -> List[Dict[str, Any]]:
    if tumor:
        data_dict = [{'label': dataframe.label.values.tolist()[i], 'id': dataframe.id.values.tolist()[i], 'image': torch.load(dataframe.tumor.values.tolist()[i])} for i in range(len(dataframe))]
    else:
        data_dict = [{'label': dataframe.label.values.tolist()[i], 'id': dataframe.id.values.tolist()[i], 'image': dataframe.path.values.tolist()[i]} for i in range(len(dataframe))]
    return data_dict

def froze_encoder(model: nn.Module) -> nn.Module:
    for param in list(model.encoder.parameters())[:33]:
        param.requires_grad = False
    return model

def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key.replace('module.', '')] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def set_seed(seed: int = 42):
    import numpy as np
    import random
    monai.utils.set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
                               