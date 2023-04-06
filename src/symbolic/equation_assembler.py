from typing import List
import os
import joblib
import numpy as np
from sympy import symbols, sympify
from sympy.parsing.sympy_parser import parse_expr
pjoin = os.path.join

class SymbolicComponent():
    def __init__(self, conf:dict) -> None:
        equation:str = conf['equation']
        self.input_dims:int = conf['input_dims']
        self.xscaler_path:str = conf['xscaler_path'] # sklearn class
        self.yscaler_path:str = conf['yscaler_path']
        self.xscaler = None
        self.yscaler = None
        self.set_scaler()
        # self.syms_x = symbols(
        #     (" ").join([f"x{i}" for i in range(len(self.input_dims))])
        # )
        self.syms_x = [symbols(f"x{i}") for i in range(len(self.input_dims))]
        self.sym_eq = sympify(parse_expr(equation))
        # self.predictor = lambdify(self.syms_x, self.sym_eq, "numpy")
        # self.__subs_row = lambda row: self.sym_eq.subs([(x, row[i]) for i, x in enumerate(self.syms_x)])

    def __subs_row(self, row):
        return self.sym_eq.subs([(x, row[i]) for i, x in enumerate(self.syms_x)])


    def set_scaler(self):
        self.xscaler = joblib.load(self.xscaler_path)
        self.yscaler = joblib.load(self.yscaler_path)
    
    def predict(self, x:np.ndarray):
        assert x.shape[1] == len(self.input_dims)
        assert x.shape[1] >= 1
        x_ = self.xscaler.transform(x)
        y_ = np.apply_along_axis(self.__subs_row, 1, x_)
        assert y_.shape[0] == x.shape[0]
        if y_.ndim ==  1:
            y_ = y_.reshape(-1, 1)
        y = self.yscaler.inverse_transform(y_)
        return y

class SymbolicYeildSurfacePolynimialHO():
    def __init__(self, conf) -> None:
        assert conf["form_type"] == "PolynomialHO"
        self.x_num_dim = conf['x_num_dim']
        self.func_weights = np.array(conf['func_weights']).reshape(-1, 1)
        self.ho_dim_pairs = conf['ho_dim_pairs']
        self.symb_funcs:List[SymbolicComponent] = [SymbolicComponent(conf['symb_funcs'][i]) for i in range(self.x_num_dim)]
        self.xscaler_path:str = conf['xscaler_path'] # sklearn class
        self.yscaler_path:str = conf['yscaler_path']
        self.final_bias:float = conf['final_bias']
        assert isinstance(self.final_bias, float)
        self.set_scaler()

    def set_scaler(self):
        self.xscaler = joblib.load(self.xscaler_path)
        self.yscaler = joblib.load(self.yscaler_path)
    
    def predict_from_basis(self, all_basis:np.ndarray):
        assert all_basis.shape[1] == self.func_weights.shape[0]
        y_ = all_basis @ self.func_weights + self.final_bias
        y = self.yscaler.inverse_transform(y_)
        return y
    
    def get_ho_basis(self, lo_basis:np.ndarray):
        assert lo_basis.shape[1] == self.x_num_dim
        ho_basis = []
        for i, j in self.ho_dim_pairs:
            ho_basis.append(lo_basis[:, i] * lo_basis[:, j])
        ho_basis = np.stack(ho_basis, axis=1)
        assert ho_basis.shape[1] == len(self.ho_dim_pairs), ho_basis.shape
        return ho_basis

    def get_lo_basis(self, x:np.ndarray):
        assert x.shape[1] == self.x_num_dim
        assert x.shape[1] >= 1, self.x.shape
        x_ = self.xscaler.transform(x)
        lo_basis = []
        for i in range(self.x_num_dim):
            tmp = self.symb_funcs[i].predict(x_[:, i:i+1])
            assert tmp.shape[1] >= 1
            lo_basis.append(tmp)
        lo_basis = np.concatenate(lo_basis, axis=1)
        assert lo_basis.shape[1] == self.x_num_dim
        assert lo_basis.ndim == 2
        return lo_basis

    def predict(self, x:np.ndarray):
        lo_basis = self.get_lo_basis(x)
        ho_basis = self.get_ho_basis(lo_basis)
        all_basis = np.concatenate([lo_basis,ho_basis], axis=1)
        return self.predict_from_basis(all_basis)