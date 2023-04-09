import numpy as np

def Voigt2Tensor(voigt, flag='strain'):

    if flag.lower() == 'strain':
        if np.shape(voigt)[0] == 3:
            Tensor = np.zeros((2,2))
            Tensor[0,0] = voigt[0]
            Tensor[0,1] = voigt[2]*0.5
            Tensor[1,0] = voigt[2]*0.5
            Tensor[1,1] = voigt[1]
        elif np.shape(voigt)[0] == 6:
            Tensor = np.zeros((3,3))
            Tensor[0,0] = voigt[0]
            Tensor[1,1] = voigt[1]
            Tensor[2,2] = voigt[2]
            Tensor[0,1] = voigt[3]*0.5
            Tensor[1,0] = voigt[3]*0.5
            Tensor[1,2] = voigt[4]*0.5
            Tensor[2,1] = voigt[4]*0.5
            Tensor[0,2] = voigt[5]*0.5
            Tensor[2,0] = voigt[5]*0.5
        else:
            assert False, "Check Fem.Dimension"

    elif flag.lower() == 'stress':
        if np.shape(voigt)[0] == 3:
            Tensor = np.zeros((2,2))
            Tensor[0,0] = voigt[0]
            Tensor[0,1] = voigt[2]
            Tensor[1,0] = voigt[2]
            Tensor[1,1] = voigt[1]
        elif np.shape(voigt)[0] == 6:
            Tensor = np.zeros((3,3))
            Tensor[0,0] = voigt[0]
            Tensor[1,1] = voigt[1]
            Tensor[2,2] = voigt[2]
            Tensor[0,1] = voigt[3]
            Tensor[1,0] = voigt[3]
            Tensor[1,2] = voigt[4]
            Tensor[2,1] = voigt[4]
            Tensor[0,2] = voigt[5]
            Tensor[2,0] = voigt[5]

    else:
        assert False, "Check flag in Voigt2Tensor()"

    return Tensor

def Tensor2Voigt(Tensor, flag='strain'):        
    if np.shape(Tensor)[0] == 2:
        voigt = np.zeros((3))
        if flag == 'strain':
            voigt[0] = Tensor[0,0]
            voigt[1] = Tensor[1,1]
            voigt[2] = Tensor[0,1] + Tensor[1,0]
        elif flag == 'stress':
            voigt[0] = Tensor[0,0]
            voigt[1] = Tensor[1,1]
            voigt[2] = Tensor[0,1]
        else:
            assert False, "Check flag in Tensor2Voigt()"

    elif np.shape(Tensor)[0] == 3:
        voigt = np.zeros((6))
        if flag == 'strain':
            voigt[0] = Tensor[0,0]
            voigt[1] = Tensor[1,1]
            voigt[2] = Tensor[2,2]
            voigt[3] = Tensor[0,1]+Tensor[1,0]
            voigt[4] = Tensor[1,2]+Tensor[2,1]
            voigt[5] = Tensor[2,0]+Tensor[0,2]
        elif flag == 'stress':
            voigt[0] = Tensor[0,0]
            voigt[1] = Tensor[1,1]
            voigt[2] = Tensor[2,2]
            voigt[3] = Tensor[0,1]*0.5
            voigt[4] = Tensor[1,2]*0.5
            voigt[5] = Tensor[2,0]*0.5
        else:
            assert False, "Check flag in Tensor2Voigt()"
    else:
        assert False, "Check Fem.Dimension"
    return voigt

def FthTensor2Voigt(Ce):

    ddsdde = np.zeros((6,6))
    for ii in range(3):
        for jj in range(3):
            ddsdde[ii,jj] = Ce[ii,ii,jj,jj]

    ddsdde[0,3] = Ce[0,0,0,1]
    ddsdde[0,4] = Ce[0,0,1,2]
    ddsdde[0,5] = Ce[0,0,2,0]
    ddsdde[1,3] = Ce[1,1,0,1]
    ddsdde[1,4] = Ce[1,1,1,2]
    ddsdde[1,5] = Ce[1,1,2,0]
    ddsdde[2,3] = Ce[2,2,0,1]
    ddsdde[2,4] = Ce[2,2,1,2]
    ddsdde[2,5] = Ce[2,2,2,0]
    ddsdde[3,0] = Ce[0,1,0,0]
    ddsdde[3,1] = Ce[0,1,1,1]
    ddsdde[3,2] = Ce[0,1,2,2]
    ddsdde[4,0] = Ce[1,2,0,0]
    ddsdde[4,1] = Ce[1,2,1,1]
    ddsdde[4,2] = Ce[1,2,2,2]
    ddsdde[5,0] = Ce[2,0,0,0]
    ddsdde[5,1] = Ce[2,0,1,1]
    ddsdde[5,2] = Ce[2,0,2,2]
    ddsdde[3,3] = Ce[0,1,0,1]
    ddsdde[3,4] = Ce[0,1,1,2]
    ddsdde[3,5] = Ce[0,1,2,0]
    ddsdde[4,3] = Ce[1,2,0,1]
    ddsdde[4,4] = Ce[1,2,1,2]
    ddsdde[4,5] = Ce[1,2,2,0]
    ddsdde[5,3] = Ce[2,0,0,1]
    ddsdde[5,4] = Ce[2,0,1,2]
    ddsdde[5,5] = Ce[2,0,2,0]

    return ddsdde

# tensor outer product: C_ijkl = A_ij B_kl
def tensor_oMult(A, B):
    assert(A.shape == B.shape)
    nDim = A.shape[0]
    res = np.zeros((nDim,nDim,nDim,nDim))
    for i in range(nDim):
        for j in range(nDim):
            for k in range(nDim):
                for l in range(nDim):
                    res[i,j,k,l] = A[i,j] * B[k,l]
    return res

# tensor oPlus operation: C_ijkl = A_jl B_ik
def tensor_oPlus(A, B):
    assert(A.shape == B.shape)
    nDim = A.shape[0]
    res = np.zeros((nDim,nDim,nDim,nDim))
    for i in range(nDim):
        for j in range(nDim):
            for k in range(nDim):
                for l in range(nDim):
                    res[i,j,k,l] = A[j,l] * B[i,k]
    return res

# tensor oMinus operation: C_ijkl = A_il B_jk
def tensor_oMinus(A, B):
    assert(A.shape == B.shape)
    nDim = A.shape[0]
    res = np.zeros((nDim,nDim,nDim,nDim))
    for i in range(nDim):
        for j in range(nDim):
            for k in range(nDim):
                for l in range(nDim):
                    res[i,j,k,l] = A[i,l] * B[j,k]
    return res

# 4th order identity tensor II
def identity_4(nDim):
    I = np.eye(nDim)
    res = np.zeros((nDim,nDim,nDim,nDim))
    for i in range(nDim):
        for j in range(nDim):
            for k in range(nDim):
                for l in range(nDim):
                    res[i,j,k,l] = (I[i,l] * I[j,k] + I[i,k] * I[j,l]) / 2.
    return res