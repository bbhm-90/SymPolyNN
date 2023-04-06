from typing import (
    List,
    Tuple
)
from abc import ABC, abstractmethod

from itertools import combinations
import numpy as np
import torch

class MyActivation(torch.nn.Module):
    def __init__(self, actFun='relu') -> None:
        super(MyActivation, self).__init__()
        if actFun == "relu":
            self.actFun = torch.nn.ReLU()
        elif actFun == "relu_leaky":
            self.actFun = torch.nn.LeakyReLU()
        elif actFun == 'elu':
            self.actFun = torch.nn.ELU()
        elif actFun == 'silu':
            self.actFun = torch.nn.SiLU()
        elif actFun == 'selu':
            self.actFun = torch.nn.SELU()
        elif actFun == 'tanh':
            self.actFun = torch.nn.Tanh()
        elif actFun == 'iden':
            self.actFun = lambda x: x
        else:
            raise NotImplementedError(actFun)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.actFun(x)

class GaussianFourierFeatureMapping(torch.nn.Module):
    def __init__(self, 
        input_dim:int, 
        output_dim:int, 
        sigma:float
        ) -> None:
        super(GaussianFourierFeatureMapping, self).__init__()
        B = torch.randn(size=(input_dim, output_dim)) * sigma * np.pi * 2.
        self.B = torch.nn.Parameter(B).requires_grad_(False)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (b, d)
        assert x.ndim == 2
        x_proj = x @ self.B
        return torch.concat([torch.cos(x_proj), torch.sin(x_proj)], axis=-1)

class MyMLP(torch.nn.Module):
    def __init__(
        self,
        layers:List[int],
        activations:List[MyActivation],
        last_bias=False,
        positional_encoding=False,
        params = {}
        ) -> None:
        """
            layers: [1, 5, 5, 1]
        """
        super(MyMLP, self).__init__()
        assert len(layers) >= 3, len(layers)
        assert len(activations) == len(layers) - 1, (len(activations), len(layers))
        net = []
        layers_ = [i for i in layers]
        if positional_encoding and params['sigma_pos_enc'] > 0.:
            net.append(
                GaussianFourierFeatureMapping(layers[0], layers[1], sigma=params['sigma_pos_enc'])
            )
            layers_[0] = layers_[1]*2
        lb = True
        for i in range(len(activations)):
            if i == len(activations):
                lb = last_bias
            net.append(
                torch.nn.Linear(layers_[i], layers_[i+1], bias=lb),
            )
            net.append(activations[i])
        self.net = torch.nn.Sequential(*net)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def train_regression(self, 
        x:torch.Tensor, 
        y:torch.Tensor, 
        opt:torch.optim.Optimizer
        ) -> float:
        """
            train one batch data
        """
        loss_func = torch.nn.MSELoss()
        y_ = self.forward(x)
        assert y_.shape == y.shape, (y_.shape, y.shape)
        loss = loss_func(y_, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

class BaseAdditive(ABC, torch.nn.Module):
    def __init__(self,
        dim,
        layers_after_first:List[int],
        activations:List[MyActivation],
        last_bias=False,
        positional_encoding=False,
        params = {}, 
        ) -> None:
        super().__init__()
        self.dim = dim
        self.weights_1st_order = torch.nn.Parameter(torch.ones(dim, 1) / dim)
        
        self.dim_pairs:torch.Tensor = None
        self._set_dim_pairs()
        self.weights_2nd_order = torch.nn.Parameter(torch.zeros(self.dim_pairs.shape[0], 1))
        
        if params['final_bias'] == 1:
            self.final_bias = torch.nn.Parameter(torch.zeros(1))
        elif params['final_bias'] == 0:
            self.final_bias = torch.zeros(1).requires_grad_(False)
        else:
            raise NotImplementedError(params['final_bias'])
        
        self.models_1st_order:torch.nn.ModuleList = []
        self.models_2nd_order:torch.nn.ModuleList = []
        self.set_models(layers_after_first, activations, last_bias, positional_encoding, params)
    
    @abstractmethod
    def _set_dim_pairs(self):
        pass
    
    @abstractmethod
    def set_models(self):
        pass
    
    def set_nn_weights(self, value, model:torch.nn.ModuleList) -> None:
        for layer in model:
            for nn in layer.net:
                if isinstance(nn, torch.nn.Linear):
                    nn.weight.data.fill_(value)

    def reset_weights_1st_order(self, 
        vals:np.ndarray) -> None:
        assert vals.shape == (self.dim, 1)
        self.weights_1st_order = torch.nn.Parameter(torch.from_numpy(vals).float())
    
    def reset_weights_2nd_order(self, 
        vals:np.ndarray) -> None:
        assert vals.shape == (len(self.dim_pairs), 1)
        self.weights_2nd_order = torch.nn.Parameter(torch.from_numpy(vals).float())

    def get_shape_functions_1st_order(self, 
        x:torch.Tensor
        ) -> torch.Tensor:
        assert x.shape[1] == self.dim, (x.shape[1], self.dim)
        y = []
        for i in range(len(self.models_1st_order)):
            y.append(self.models_1st_order[i](x[:, i:i+1]))
        y = torch.concat(y, axis=1)
        assert y.shape[1] == self.dim
        return y

    def get_prediction_1st_order(self, 
        shape_functions_1st_order:torch.Tensor
        ) -> torch.Tensor:
        assert shape_functions_1st_order.shape[1] == self.weights_1st_order.shape[0]
        assert self.weights_1st_order.ndim == 2
        y = shape_functions_1st_order @ self.weights_1st_order + self.final_bias
        assert y.shape[1] == 1
        return y

    @abstractmethod
    def get_shape_functions_2nd_order(self, 
        x:torch.Tensor
        ) -> torch.Tensor:
        pass

    def get_prediction_2nd_order(self, 
        shape_functions_2nd_order:torch.Tensor
        ) -> torch.Tensor:
        assert self.weights_2nd_order.ndim == 2
        out = shape_functions_2nd_order @ self.weights_2nd_order
        assert out.shape[1] == 1
        return out

    @abstractmethod
    def forward(self, 
        x:torch.Tensor
        ) -> torch.Tensor:
        pass

    def make_grad_zero(self, model) -> None:
        for param in model:
            if param.grad is not None:
                param.grad.data.zero_()

    def train_first_order_only(self, 
        x:torch.Tensor, 
        y:torch.Tensor, 
        opt: torch.optim.Optimizer,
        freeze_second_order_weights = True,
        **kwargs
        ) -> Tuple[float, float]:
        criterion_pred = torch.nn.MSELoss()

        if freeze_second_order_weights:
            self.weights_2nd_order.requires_grad = False
        for param in self.models_2nd_order:
            param.requires_grad = False

        self.weights_1st_order.requires_grad = True
        for param in self.models_1st_order:
            param.requires_grad = True

        y_, _, _ = self.forward(x)
        # TODO: this would be inconsistent
        # shape_func1 = self.get_shape_functions_1st_order(x)
        # y_ = self.get_prediction_1st_order(shape_func1)
        assert y.shape == y_.shape, (y.shape, y_.shape)
        penalty_first_order = kwargs['penalty_first_order']
        loss_pred = criterion_pred(y_, y)
        loss_sparse_1st = torch.abs(self.weights_1st_order).mean()
        opt.zero_grad()
        loss_tot = loss_pred + penalty_first_order * loss_sparse_1st
        loss_tot.backward()
        self.make_grad_zero(self.models_2nd_order)
        if freeze_second_order_weights:
            self.make_grad_zero(self.weights_2nd_order)
        opt.step()
        return loss_pred.item(), loss_sparse_1st.item()

    def train_second_order_only(self, 
        x:torch.Tensor, 
        y:torch.Tensor, 
        opt:torch.optim.Optimizer, 
        penalty_second_order_weights=0.,
        freez_first_order_weights = True
        ) -> Tuple[float, float]:
        criterion_pred = torch.nn.MSELoss()

        for param in self.models_1st_order.parameters():
            param.requires_grad = False
        if freez_first_order_weights:
            self.weights_1st_order.requires_grad = False

        for param in self.models_2nd_order.parameters():
            param.requires_grad = True
        self.weights_2nd_order.requires_grad = True


        y_, _, _ = self.forward(x)
        assert y.shape == y_.shape, (y.shape, y_.shape)
        loss_pred = criterion_pred(y_, y)
        loss = loss_pred
        loss_sparsity = -1.
        if penalty_second_order_weights>0.:
            loss_sparsity = torch.abs(self.weights_2nd_order).mean()
            loss += (loss_sparsity * penalty_second_order_weights)
        opt.zero_grad()
        loss.backward()
        self.make_grad_zero(self.models_1st_order)
        if freez_first_order_weights:
            self.make_grad_zero(self.weights_1st_order)
        opt.step()
        return loss_pred.item(), loss_sparsity.item()

    def train_end_to_end(self, 
        x:torch.Tensor, 
        y:torch.Tensor, 
        opt:torch.optim.Optimizer, 
        penalty_first_order=0., 
        penalty_second_order=0.,
        penalty_data=1.,
        tol_sparsity = 1e-4
        ) -> Tuple[float, float, float]:
        # criterion_pred = torch.nn.MSELoss()
        for param in self.models_1st_order.parameters():
            param.requires_grad = True
        self.weights_1st_order.requires_grad = True
        if self.models_2nd_order:
            for param in self.models_2nd_order.parameters():
                param.requires_grad = True
        self.weights_2nd_order.requires_grad = True

        y_, _, _ = self.forward(x)
        assert y.shape == y_.shape, (y.shape, y_.shape)
        
        loss_pred = (y_ - y).pow(2).mul(penalty_data).mean()
        loss = loss_pred
        loss_sparsity_first_order = torch.ones(1) * -1
        if penalty_first_order > 0.:
            # loss_sparsity_first_order = (torch.abs(self.weights_1st_order) - tol_sparsity).relu().sum()
            loss_sparsity_first_order = torch.abs(self.weights_1st_order).mean()
            loss += (loss_sparsity_first_order *penalty_first_order)
        loss_sparsity_second_order = torch.ones(1) * -1
        if penalty_second_order > 0.:
            # loss_sparsity_second_order = (torch.abs(self.weights_2nd_order) - tol_sparsity).relu().sum()
            loss_sparsity_second_order = torch.abs(self.weights_2nd_order).mean()
            loss += (loss_sparsity_second_order * penalty_second_order)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss_pred.item(), loss_sparsity_first_order.item(), loss_sparsity_second_order.item()


class PolynomialHigherOrder(BaseAdditive):
    def __init__(self, 
        dim,
        layers_after_first:List[int],
        activations:List[MyActivation],
        last_bias=False,
        positional_encoding=False,
        params = {}, 
        ) -> None:
        super().__init__(
            dim,layers_after_first,activations,last_bias, positional_encoding, params
        )
    
    def _set_dim_pairs(self):
        self.dim_pairs = list(combinations(range(self.dim), 2))
        self.dim_pairs = self.dim_pairs + [[i, i] for i in range(self.dim)]
        self.dim_pairs = torch.tensor(self.dim_pairs, dtype=int).contiguous()

    def set_models(self,
        layers_after_first:List[int],
        activations:List[MyActivation],
        last_bias=False,
        positional_encoding=False,
        params = {},
        ) -> None:
        models= []
        for _ in range(self.dim):
            models.append(
                MyMLP(layers=[1] + layers_after_first, 
                        activations=activations,
                        last_bias=last_bias,
                        positional_encoding=positional_encoding,
                        params=params,
                        )
            )
        self.models_1st_order = torch.nn.ModuleList(models)

    def get_shape_functions_2nd_order(self, 
        shape_functions_1st_order:torch.Tensor
        ) -> torch.Tensor:
        assert shape_functions_1st_order.shape[1] == self.dim, (shape_functions_1st_order.shape, self.dim)
        out = shape_functions_1st_order[:, self.dim_pairs]
        out = out[:, :, 0] * out[:, :, 1]
        assert out.shape[1] == self.dim_pairs.shape[0], (out.shape, self.dim_pairs.shape[0])
        return out
    
    def forward(self, 
        x:torch.Tensor
        ) -> torch.Tensor:
        assert x.shape[1] == self.dim, (x.shape[1], self.dim)
        shape_fun1 = self.get_shape_functions_1st_order(x)
        shape_fun2 = self.get_shape_functions_2nd_order(shape_fun1)
        out = self.get_prediction_1st_order(shape_fun1) + self.get_prediction_2nd_order(shape_fun2)
        return out, shape_fun1, shape_fun2