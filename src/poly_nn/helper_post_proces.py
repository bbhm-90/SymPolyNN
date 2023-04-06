import os
import joblib
import json
import numpy as np
import pandas as pd
import torch

from src.poly_nn.model import (
    MyActivation, 
    PolynomialHigherOrder,
    MyMLP
)

from src.poly_nn.helper import (
    read_data,
    check_args,
)

pjoin = os.path.join

class PostProcessor():
    def __init__(self, result_dir:str, data=None) -> None:
        args_add = pjoin(result_dir, 'args.json')
        with open(args_add, 'r') as f:
            args = json.load(f)
        check_args(args)
        self.outDir = args['outDir']
        randomSeed = args['randomSeed']
        randomSeed = args['randomSeed']
        self.addetive_class = args['addetive_class']
        dataAdd = args['dataAdd']
        dataCord = args['dataCord']
        layers_after_first = args['layers_after_first']
        activations = args['activations']
        positional_encoding = args['positional_encoding']
        sigma_pos_enc = args['sigma_pos_enc']
        final_bias = args['final_bias']

        torch.manual_seed(randomSeed)
        np.random.seed(randomSeed)
        if dataCord == "cartesian":
            input_fields = ['sigma1', 'sigma2', 'sigma3']
        elif dataCord == "cylindrical":
            input_fields = ['p', 'rho', 'theta']
        else:
            raise NotImplementedError(dataCord)
        
        if data is None:
            self.xraw, self.yraw = read_data(dataAdd, input_fields)
        else:
            self.xraw, self.yraw = data[:, :-1], data[:, -1].reshape(-1, 1)
        if self.addetive_class in {"baseLO", "PolynomialHO"}:
            addetiveClass = PolynomialHigherOrder
        elif self.addetive_class == "singleMLP":
            addetiveClass = MyMLP
        else:
            raise NotImplementedError(self.addetive_class)
        if self.addetive_class == "singleMLP":
            self.model = addetiveClass(layers=[self.xraw.shape[1]] + layers_after_first,
                activations=[MyActivation(act) for act in activations],
                positional_encoding=positional_encoding,
                params={'sigma_pos_enc':sigma_pos_enc, 'final_bias':final_bias}
            )
        else:
            self.model = addetiveClass(
                dim=self.xraw.shape[1], 
                layers_after_first=layers_after_first,
                activations=[MyActivation(act) for act in activations],
                positional_encoding=positional_encoding,
                params={'sigma_pos_enc':sigma_pos_enc, 'final_bias':final_bias}
            )
        self.xscaler = joblib.load(pjoin(result_dir, 'xscaler.joblib'))
        self.yscaler = joblib.load(pjoin(result_dir, 'yscaler.joblib'))

        xscaled = self.xscaler.transform(self.xraw)
        yscaled = self.yscaler.transform(self.yraw)

        self.xscaled = torch.from_numpy(xscaled).float()
        self.yscaled = torch.from_numpy(yscaled).float()
        self.model.load_state_dict(torch.load(pjoin(self.outDir, "model.pth")))
        self.model.eval()
        traced_model = torch.jit.trace(self.model, self.xscaled[:3, :])
        traced_model.save(pjoin(self.outDir, "model.ptjit"))
        self.weights_1st_order = None
        self.weights_2nd_order = None
        if hasattr(self.model, "weights_2nd_order"):
            self.weights_2nd_order = self.model.weights_2nd_order.detach().numpy().flatten()
        if hasattr(self.model, "weights_1st_order"):
            self.weights_1st_order = self.model.weights_1st_order.detach().numpy().flatten()
        self.shap_func_dir:str = None

    @torch.no_grad()
    def get_1st_shape_func(self, xscaled:torch.Tensor)->np.ndarray:
        if self.weights_1st_order is None:
            return
        _, out, _ = self.model(xscaled)
        return out.detach().numpy()
    
    @torch.no_grad()
    def get_2nd_shape_func(self, xscaled:torch.Tensor)->np.ndarray:
        if self.weights_2nd_order is None:
            return
        _, _, out = self.model(xscaled)
        return out.detach().numpy()
    
    @torch.no_grad()
    def get_prediction(self, xscaled:torch.Tensor) -> np.ndarray:
        if self.weights_1st_order is None:
            out = self.model(xscaled)
        else:
            out, _, _ = self.model(xscaled)
        return out.detach().numpy()

    def get_inverse_transform_y(self, yscaled:np.ndarray) -> np.ndarray:
        assert yscaled.shape[1] == 1
        return self.yscaler.inverse_transform(yscaled)

    def save_shape_func(self,
        x:np.ndarray,
        shape_functions_1st_order:np.ndarray=None, 
        ):
        self.shap_func_dir = pjoin(self.outDir, "shape_func")
        if not os.path.exists(self.shap_func_dir):
            os.makedirs(self.shap_func_dir)
        if shape_functions_1st_order is not None:
            for i in range(shape_functions_1st_order.shape[1]):
                pd.DataFrame(
                    {
                        "x":x[:, i],
                        "f":shape_functions_1st_order[:, i]
                    }
                ).to_csv(
                    pjoin(self.shap_func_dir, f"f_{i}_x_{i}.csv"),
                    index=False,
                    header=False,
                )

    def get_regular_samples(self, Nsamples=200, delta_ratio=0.05) -> torch.Tensor:
        xmin = self.xscaled.min(dim=0)[0]
        xmax = self.xscaled.max(dim=0)[0]
        delta = (xmax - xmin) * delta_ratio

        xscaled_new = []
        Nsamples = 200
        for i in range(self.xscaled.shape[1]):
            xscaled_new.append(
                torch.linspace(xmin[i]+delta[i], xmax[i]-delta[i], Nsamples)
            )
        xscaled_new = torch.stack(xscaled_new, dim=1)
        assert xscaled_new.shape[1] == self.xscaled.shape[1]
        return xscaled_new
    
    def extract_shape_functions(self, need_save=True):
        xscaled_new = self.get_regular_samples()
        shape_functions_1st_order = self.get_1st_shape_func(xscaled_new)
        shape_functions_2nd_order = self.get_2nd_shape_func(xscaled_new)
        if need_save:
            self.save_shape_func(xscaled_new, shape_functions_1st_order)
        return xscaled_new, shape_functions_1st_order, shape_functions_2nd_order
    
    @torch.no_grad()
    def save_func_weights(self):
        with torch.no_grad():
            tmp = {}
            tmp['weights_1st_order'] = self.model.weights_1st_order.detach().numpy().flatten().tolist()
            tmp['weights_2nd_order'] = self.model.weights_2nd_order.detach().numpy().flatten().tolist()
            tmp['dim_pairs'] = self.model.dim_pairs.detach().numpy().tolist()
        with open(pjoin(self.outDir, 'func_weights.json'), 'w') as f:
            json.dump(tmp, f, indent=2)
        print("--- saved model")