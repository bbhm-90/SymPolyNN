import torch
import joblib
import numpy as np

model_NAM = torch.jit.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/model.ptjit")
f_INPUT_scaler  = joblib.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/xscaler.joblib")
f_OUTPUT_scaler = joblib.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/yscaler.joblib")

def f_NAM(p, rho, theta):

    RT = np.array([p, rho, theta]).reshape(1,3)
    RT = f_INPUT_scaler.transform(RT)
    RT = torch.tensor(RT, dtype=torch.float)
    
    f = model_NAM(RT)
    f = f[0]
    f_numpy = f.detach().numpy()[0,0]
    f_numpy = f_OUTPUT_scaler.inverse_transform(f_numpy.reshape(-1,1))

    return f_numpy[0,0]