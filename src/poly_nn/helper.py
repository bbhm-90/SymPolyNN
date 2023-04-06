from typing import (
    Iterator,
    Tuple,
    Union,
    List
)
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch

def get_feature_multiplication(
    x:np.ndarray, 
    index_pairs:Union[Iterator[Tuple[int, int]], torch.Tensor]
    ) -> torch.Tensor:
    out = []
    for i, j in index_pairs:
        out.append(
            x[:, i:i+1] * x[:, j:j+1]
        )
    out = np.concatenate(out, axis=1)
    assert out.shape[1] == len(index_pairs)
    return out

def check_args(args:Union[ArgumentParser, dict]):
    if isinstance(args, dict):
        if "positional_encoding" in args.keys():
            assert args['sigma_pos_enc'] >= 0., args['sigma_pos_enc']
            return
    if args.positional_encoding:
       assert args.sigma_pos_enc >= 0., args.sigma_pos_enc
       return

def read_data(
        data_add:str, 
        input_fields:List[str]
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
        input_fields: ['p','rho','theta']
                      ['sigma1','sigma2','sigma3']
                      ["sigma_h", "sigma_vm", "L", "v"]
    """
    xRaw = pd.read_csv(data_add)
    x = xRaw[input_fields].to_numpy()
    if len(input_fields) == 0:
        xRaw = xRaw.to_numpy()
        x = xRaw[:, :-1]
    y = xRaw.to_numpy()[:, -1:]
    return x, y

def get_equal_yx_line(xmin, xmax):
    x = np.zeros((2, 2))
    x[0, :] = [xmin, xmin]
    x[1, :] = [xmax, xmax]
    return x

def find_root(model, x0:torch.Tensor, lr=1e-2, tol=1e-5, maxItr=10, verbose=False):
    assert x0.ndim == 2
    assert x0.shape[0] == 1
    x = x0.detach().float().requires_grad_(True)
    itr = 0
    while itr < maxItr:
        y = model(x)
        if y.abs() < tol:
            break
        dy = torch.autograd.grad(y, x)[0]
        with torch.no_grad():
            x -= (y.item() / dy) * lr
        if verbose:
            print(itr, y.item())
        itr += 1
    if y.abs() > tol:
        return None
    return x.detach().numpy()