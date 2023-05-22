# coding=utf-8
__authors__ = "Takuma Shibahara"
__copyright__ = "Hitachi, Ltd."
__license__ = "Proprietary software"
__maintainer__ = ""
__email__ = ""

from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl


class PWL(pl.LightningModule):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 in_features: int = 100,
                 dropout_input: float = 0.0,
                 dropout_inner: float = 0.5,
                 reg_l1: float = 1.0e-15,
                 reg_l2: float = 1.0e-15,
                 lr: float = 0.01):

        super(PWL, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.in_features = in_features
        self.dropout_input = dropout_input
        self.dropout_inner = dropout_inner
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.lr = lr

        self.criterion = nn.NLLLoss()

        units = OrderedDict()
        for k in range(self.num_layers - 1):
            if k == 0:
                units["drop_" + str(k + 1)] = nn.Dropout(self.dropout_input)
                units["inner_" + str(k + 1)] = nn.Linear(self.input_dim, self.in_features)
                units["act_" + str(k + 1)] = nn.ReLU()
            else:
                units["drop_" + str(k + 1)] = nn.Dropout(self.dropout_inner)
                units["inner_" + str(k + 1)] = nn.Linear(self.in_features, self.in_features)
                units["act_" + str(k + 1)] = nn.ReLU()

        units["drop_end"] = nn.Dropout(self.dropout_inner)
        units["inner_end"] = nn.Linear(self.in_features, self.input_dim)

        self.rho_unit = nn.Sequential(units)  # NOTE: Various NNs, including RNNs and CNNs, can be used to compose the reallocation vector.

        units = OrderedDict()
        units["drop"] = nn.Dropout(self.dropout_input)
        units["inner"] = nn.Linear(self.input_dim, self.output_dim)
        self.output_linear = nn.Sequential(units)

    def nn_init(self):
        self.param_dict = OrderedDict()
        for name, param in self.named_parameters():
            if re.search(r"weight", name):
                nn.init.xavier_uniform_(param)
            elif re.search(r"bias", name):
                nn.init.zeros_(param)

            self.param_dict[name] = param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rho = self.rho_unit(x)
        return torch.clamp(self.output_linear(self.rho * x), min=-1.0, max=1.0)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            return self.output_linear.inner.weight.unsqueeze(1) * self.rho_unit(x).unsqueeze(0)  # NOTE: Eqn. (4)

    def regulizer(self) -> torch.Tensor:
        if 0 < self.reg_l1 or 0 < self.reg_l2:
            param = self.output_linear.inner.weight.unsqueeze(1) * self.rho.unsqueeze(0)
            penalty = self.reg_l1 * param.abs().sum() + self.reg_l2 * param.pow(2).sum().sqrt()
        else:
            penalty = 0.0

        return penalty

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            return F.softmax(self.forward(x), dim=-1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        yh = self.forward(x)
        return self.criterion(torch.log_softmax(yh, dim=-1), y.long()) + self.regulizer()

    def on_training_epoch_end(self, train_step_outputs):
        loss = torch.stack([val["loss"] for val in train_step_outputs], dim=0).mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.98)
