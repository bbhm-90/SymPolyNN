import torch
import joblib
import numpy as np

model_MLP = torch.load("example/flower_shape/MLP/model_single_MLP.pth")
f_INPUT_scaler  = joblib.load("example/flower_shape/MLP/xscaler.pkl")
f_OUTPUT_scaler = joblib.load("example/flower_shape/MLP/yscaler.pkl")

def f_MLP(p, rho, theta):

    RT = np.array([p, rho, theta]).reshape(1,3)
    RT = f_INPUT_scaler.transform(RT)
    RT = torch.tensor(RT, dtype=torch.float)
    
    f = model_MLP(RT)
    f = f_OUTPUT_scaler.inverse_transform(f.detach().numpy())
    f = f[0]

    return f[0]