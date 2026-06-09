from abc import ABC, abstractmethod
from torch import nn



class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()
        self.layer_gpu_id = []
        self.blcok_io_shapes = []
        self.model_options_str = ""
        self.layers = nn.Sequential()
        self.create_layers(*args, **kwargs)

    @property
    def model_options(self):
        return self.model_options_str

    @property
    @abstractmethod
    def model_name(self):
        return str()

    @abstractmethod
    def create_layers(self, *args, **kwargs):
        pass

    def set_model_options(self, model_options: str):
        self.model_options = model_options

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
