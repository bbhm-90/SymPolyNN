import time
import os
import json
import joblib
import numpy as np
from sympy import sympify
from pysr import PySRRegressor
from src.poly_nn.helper_parser import get_scaler

pjoin = os.path.join

def fit_scaler(x, scaler_type):
    assert scaler_type in {"standard", "minmax"}
    scaler = get_scaler(scaler_type)
    scaler.fit(x)
    return scaler

class MyPySR(PySRRegressor):
    def __init__(self, model_conf_add:str, output_dirpath:str=None) -> None:
        assert model_conf_add.split(".")[-1] == "json", model_conf_add
        with open(model_conf_add, 'r') as f:
            self.conf = json.load(f)
        if output_dirpath is not None:
            self.conf["tempdir"] = output_dirpath
        if "scaler_type" in self.conf:
            self.scaler_type = self.conf["scaler_type"]
            del self.conf["scaler_type"]

        super().__init__(**self.conf)
        self.xscaler = None
        self.yscaler = None
    
    def load_from_file(self, trained_model_add:str):
        assert trained_model_add.split('.')[-1] == "pkl", trained_model_add
        model = self.from_file(trained_model_add)
        model.warm_start = True
        return model
    
    def set_data_scaler(self, x:np.ndarray, y:np.ndarray):
        if self.scaler_type == "":
            return x, y
        self.xscaler = fit_scaler(x, self.scaler_type)
        self.yscaler = fit_scaler(y, self.scaler_type)

    def load_xscaler(self, add:str,):
        assert add.endswith(".joblib")
        self.xscaler = joblib.load(add)

    def load_yscaler(self, add:str,):
        assert add.endswith(".joblib")
        self.yscaler = joblib.load(add)

    def load_scaler(self, addX:str, addY:str):
        self.load_xscaler(addX)
        self.load_yscaler(addY)

    def fit_model(self, x, y):
        assert x.shape[0] == y.shape[0], (x.shape, y.shape)
        assert x.shape[1] >= 1, x.shape
        assert y.shape[1] >= 1, y.shape
        self.set_data_scaler(x, y)
        x_, y_ = x, y
        if self.xscaler is not None:
            x_ = self.xscaler.transform(x)
            y_ = self.yscaler.transform(y)
        t0= time.time()
        self.fit(x_, y_)
        elapsed_time = time.time() - t0
        if self.xscaler is not None:
            joblib.dump(self.xscaler, pjoin(self.tempdir_, "xscaler.joblib"))
        if self.yscaler is not None:
            joblib.dump(self.yscaler, pjoin(self.tempdir_, "yscaler.joblib"))
        with open(pjoin(self.tempdir_, 'elapsed_time.txt'), 'w') as f:
            f.write(f"{elapsed_time}")
        self.save_config()
        
        tmp = self.equation_file_.split(".csv")[0]
        csvPathDes = pjoin(self.tempdir_, self.equation_file_)
        
        pklPathCur = tmp + ".pkl"
        pklPathDes = pjoin(self.tempdir_, pklPathCur)

        bkupPathCur = tmp + ".csv.bkup"
        bkupPathDes = pjoin(self.tempdir_, bkupPathCur)

        os.system(f"mv {pklPathCur} {pklPathDes}")
        os.system(f"rm {pklPathCur}")

        os.system(f"mv {bkupPathCur} {bkupPathDes}")
        os.system(f"rm {bkupPathCur}")

        os.system(f"mv {self.equation_file_} {csvPathDes}")
        os.system(f"rm {self.equation_file_}")

        print(self.equation_file_)
        return elapsed_time

    def save_config(self, indent=4) -> None:
        tmp = {i:j for i, j in self.conf.items()}
        tmp.update({'scaler_type':self.scaler_type})
        with open(pjoin(self.tempdir_, "config.json"), 'w') as f:
            json.dump(tmp, f, indent=indent)

    def get_sympy_expression(self, index=None, num_decimals=None):
        # # print(clsym.model.equations_)
        if num_decimals is None:
            return sympify(self.sympy(index))
        # print(symp_round(self.sympy(index), num_decimals))
        expr = self.sympy(index)
        expr = expr.xreplace({num: round(num, num_decimals) for num in expr.atoms() if num.is_Number})
        return sympify(expr)
    
    def get_predict(self, x, index=None):
        x_ = self.xscaler.transform(x)
        y_ = self.predict(x_, index=index).reshape(-1, 1)
        return self.yscaler.inverse_transform(y_)
    
    def get_all_equations(self):
        ## called after training
        return self.equations_