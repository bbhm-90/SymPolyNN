import numpy as np
from numpy import ix_
from include.FemBulk import *
from include.util.tensor_operations import *
from include.util.eigen_sorting import *
import importlib

class GPstate:
    eps_e: np.array
    eps_p: np.array
    deps: np.array
    lamda: float
    stress: np.array

def ElementSetUp(Fem, Node, Element):
    Element.B_matrix = []
    Element.Jacc  = []
    Element.Area  = []
    Element.GPstrain_e = []
    Element.GPstrain_p = []
    Element.GPlamda = []
    Element.GPstress = []

    for Edof in Element.Connectivity:
        NodeTarget = []
        for dof in Edof:
            NodeTarget.append(Node.Coord[Node.Id.index(dof)])
        Init_tensor = []
        Init_tensor1 = []
        Init_tensor2 = []
        Init_tensor3 = []
        B_matrix = []
        Jacc_Elem = []
        Area = 0.

        for gp in Fem.GP:
            if Fem.Dimension == 2:
                Jacc, B = Strain(Fem, NodeTarget, gp[0], gp[1])
            elif Fem.Dimension ==3:
                Jacc, B = Strain(Fem, NodeTarget, gp[0], gp[1], gp[2])
            else:
                assert False, "Check Fem.Dimension"

            Area = Area + Jacc * gp[-1]
            B_matrix.append(B)
            Jacc_Elem.append(Jacc)
            if Fem.Dimension ==2:
                tmp_tensor = np.zeros((3))
                tmp_tensor1 = np.zeros((3))
                tmp_tensor1[0] = Fem.HSP
                tmp_tensor1[1] = Fem.HSP
            elif Fem.Dimension ==3:
                tmp_tensor = np.zeros((6))
                tmp_tensor1 = np.zeros((6))
                tmp_tensor1[0] = Fem.HSP
                tmp_tensor1[1] = Fem.HSP
                tmp_tensor1[2] = Fem.HSP
            else:
                assert False, "Check Fem.Dimension"
            Init_tensor.append(np.copy(tmp_tensor))
            Init_tensor3.append(np.copy(tmp_tensor))
            Init_tensor1.append(0.0)
            Init_tensor2.append(tmp_tensor1)

        Element.B_matrix.append(B_matrix)
        Element.Jacc.append(Jacc_Elem)
        Element.Area.append(Area)
        Element.GPstrain_e.append(Init_tensor)
        Element.GPlamda.append(Init_tensor1)
        Element.GPstress.append(Init_tensor2)
        Element.GPstrain_p.append(Init_tensor3)

    return

def ConstructStiffness(Fem, Node, Element, ConstitutiveModel):
    G_Edof = Fem.G_Edof
    Dim    = Fem.Dimension

    Element.Stiffness  =[]
    Node.F_int = np.zeros_like(Node.F_int)
    for ind1, Edof in enumerate(Element.Connectivity):
        ENdof = ElementalNodeDof(Fem, Node, Edof)
        E_K = np.zeros([G_Edof, G_Edof])
        B    = Element.B_matrix[ind1]
        Jacc = Element.Jacc[ind1]
        GPstress= Element.GPstress[ind1]
        GPeps_e = Element.GPstrain_e[ind1]
        GPeps_p = Element.GPstrain_p[ind1]
        GPlamda = Element.GPlamda[ind1]
        GPF_int = np.zeros((len(ENdof)))

        for ind, gp in enumerate(Fem.GP):
            B_GP    = B[ind]
            Jacc_GP = Jacc[ind]
            GPdata  =  GPstate()
            GPdata.eps_e  = GPeps_e[ind]
            GPdata.eps_p  = GPeps_p[ind]
            GPdata.lamda  = GPlamda[ind]
            GPdata.deps   = B_GP @ Node.du[ENdof]
            GPdata.stress = GPstress[ind]
            
            ConstitutiveModel.ReturnMapping(GPdata)

            GPeps_e[ind]  = GPdata.eps_e
            GPeps_p[ind]  = GPdata.eps_p
            GPstress[ind] = GPdata.stress
            GPlamda[ind]  = GPdata.lamda

            GP_K = B_GP.T @ GPdata.D @ B_GP * Jacc_GP * gp[-1]
            GPF_int += B_GP.T @ GPstress[ind] * Jacc_GP * gp[-1]

            E_K += GP_K

        Element.Stiffness.append(E_K)
        Node.F_int[ENdof] += GPF_int
    return


def Solve(Node, Global_K):
    IndexBCN  = list(np.where(np.isnan(Node.BC_E))[0])
    IndexBCE  = list(np.where(np.invert(np.isnan(Node.BC_E)))[0])

    Node.u1 = np.copy(Node.u)

    Sliced_K11 = Global_K[ix_(IndexBCN, IndexBCN)]

    F_total = Node.F_ext[IndexBCN] - Node.F_int[IndexBCN]

    Node.du[IndexBCN] = np.linalg.solve(Sliced_K11, F_total)
    Node.u1[IndexBCN] = Node.u[IndexBCN] + Node.du[IndexBCN]
    Node.du[IndexBCE] = Node.u1[IndexBCE] - Node.u[IndexBCE]

    Node.u = np.copy(Node.u1)

    return np.linalg.norm(F_total)**2/(1.+np.linalg.norm(Node.F_int)**2)


class ConstitutiveLaw():
    def __init__(self, Fem):
        self.Dim  = Fem.Dimension
        self.K  = Fem.K
        self.mu = Fem.mu
        self.lam = Fem.lamda
        self.E  = Fem.E
        self.nu = Fem.nu
        pm = importlib.import_module(Fem.PlasticModel)
        self.PlasticityModel = pm.MyPlasticity(Fem)
        self.MatProp = Fem.MatProp
        self.HSP = Fem.HSP
        self.GlobalNRStep = 0

    def ReturnMapping(self, GPstate):
        if self.Dim == 2:
            self.ReturnMappingEigenSpace2D(GPstate)
        elif self.Dim == 3:
            self.ReturnMappingEigenSpace3D(GPstate)
        else:
            assert False, "Check Fem.Dimension"
        return

    def ReturnMappingEigenSpace2D(self, GPstate):
        # Define elastic parameters
        E   = self.E
        nu  = self.nu

        D_e = E/(1.+nu)/(1.-2.*nu) * np.array([[1.-nu,    nu, 0.], \
                                               [nu,    1.-nu, 0.], \
                                               [0.,       0., (1.-2.*nu)/2.]]) # plane strain

        # Define derivatives (elastic)
        K  = E / (3.*(1.-2.*nu))
        mu = E / (2.*(1.+nu))

        aa = K + (4/3)*mu
        bb = K - (2/3)*mu

        dsig1depse1 = aa
        dsig1depse2 = bb
        dsig2depse1 = bb
        dsig2depse2 = aa
         
        # Read previous elastic and plastic strains
        eps_e_n = Voigt2Tensor(GPstate.eps_e, 'strain')
        eps_p_n = Voigt2Tensor(GPstate.eps_p, 'strain')
        lamda   = GPstate.lamda

        deps = Voigt2Tensor(GPstate.deps, 'strain')

        # [1] Compute trial elastic strain
        eps_e_tr = eps_e_n + deps
        eps_e_tr_principal_mag, eps_e_tr_principal_vec = np.linalg.eigh(eps_e_tr)
        eps_e_tr_principal_mag, eps_e_tr_principal_vec = SortEig2D(eps_e_tr_principal_mag, eps_e_tr_principal_vec)

        eps_e_tr1 = eps_e_tr_principal_mag[0]
        eps_e_tr2 = eps_e_tr_principal_mag[1]

        n1 = eps_e_tr_principal_vec[:,0]
        n2 = eps_e_tr_principal_vec[:,1]

        # [2] Compute trial stress
        sigma1_tr = aa*eps_e_tr_principal_mag[0] + bb*eps_e_tr_principal_mag[1] + self.HSP
        sigma2_tr = bb*eps_e_tr_principal_mag[0] + aa*eps_e_tr_principal_mag[1] + self.HSP
        
        # [3] Check yielding
        YF = self.PlasticityModel.f(sigma1_tr, sigma2_tr, self.HSP, lamda) # yield function

        if YF <= 0. or self.GlobalNRStep == 0:
            # Update stress & strain
            sigma = Voigt2Tensor(np.dot(D_e, GPstate.eps_e + GPstate.deps), 'stress')
            sigma[0,0] = sigma[0,0] + self.HSP
            sigma[1,1] = sigma[1,1] + self.HSP

            eps_e = eps_e_n + deps
            eps_p = eps_p_n
            eps   = eps_e + eps_p

            # Update tangent
            D = D_e

        else:
            # Initialize variables
            eps_e_principal_mag, eps_e_principal_vec = np.linalg.eigh(eps_e_n)
            eps_e_principal_mag, eps_e_principal_vec = SortEig2D(eps_e_principal_mag, eps_e_principal_vec)

            eps_e1 = eps_e_principal_mag[0]
            eps_e2 = eps_e_principal_mag[1]
            dlamda  = 0

            x = np.zeros(3) # target variables
            x[0] = eps_e1
            x[1] = eps_e2
            x[2] = dlamda

            # Newton-Raphson iteration (return mapping)
            for ii in range(20):
                # Initialize residual and jacobian
                res = np.zeros(3)
                jac = np.zeros((3,3))

                # Current strain
                eps_e1_current = x[0]
                eps_e2_current = x[1]

                # Current stress
                sigma1_current = aa*eps_e1_current + bb*eps_e2_current + self.HSP
                sigma2_current = bb*eps_e1_current + aa*eps_e2_current + self.HSP

                # Current lamda
                lamda_current = lamda + x[2]

                # Update derivatives
                # >> First order derivatives
                dfdsig1, dfdsig2, _, _ = self.PlasticityModel.df(sigma1_current, sigma2_current, self.HSP, lamda_current)

                # >> Second order derivatives
                d2fdsig1dsig1, d2fdsig2dsig2, _, d2fdsig1dsig2, _, _ \
                    = self.PlasticityModel.df2(sigma1_current, sigma2_current, self.HSP)
                
                # Update residual
                res[0] = x[0] - eps_e_tr1 + x[2]*dfdsig1
                res[1] = x[1] - eps_e_tr2 + x[2]*dfdsig2
                res[2] = self.PlasticityModel.f(sigma1_current, sigma2_current, self.HSP, lamda_current)

                # Update Jacobian ***
                jac[0,0] = 1 + x[2]*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1)
                jac[0,1] =     x[2]*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2)
                jac[0,2] = dfdsig1

                jac[1,0] =     x[2]*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1)
                jac[1,1] = 1 + x[2]*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2)
                jac[1,2] = dfdsig2

                jac[2,0] = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1
                jac[2,1] = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2
                jac[2,2] = 0.0

                # Solve system of equations
                dx = np.linalg.solve(jac, -res) # increment of target variables

                # Update x
                x = x + dx

                # Compute error
                err = np.linalg.norm(dx)

                if err < 1e-7:
                    break

            # Update strain
            # >> total strain
            eps = eps_e_n + eps_p_n + deps

            # >> elastic strain
            eps_e = x[0]*np.tensordot(n1,n1,axes=0) + x[1]*np.tensordot(n2,n2,axes=0)

            # >> plastic strain
            eps_p = eps - eps_e
            lamda = lamda + x[2]

            # Update stress
            sigma_vec = np.dot(D_e, Tensor2Voigt(eps_e, 'strain'))
            sigma = Voigt2Tensor(sigma_vec, 'stress')
            sigma[0,0] = sigma[0,0] + self.HSP
            sigma[1,1] = sigma[1,1] + self.HSP

            # Update tangent
            eps_e_tmp = np.zeros((3,3))
            eps_e_tmp[0:2,0:2] = eps_e

            deveps = eps_e_tmp - (1./3.)*np.trace(eps_e_tmp)*np.eye(3)
            n      = deveps / np.linalg.norm(deveps,'fro')
            nxn    = tensor_oMult(n,n)
            D_alg  = K*tensor_oMult(np.eye(3),np.eye(3)) + 2.0*mu*(identity_4(3)-(1./3.)*tensor_oMult(np.eye(3),np.eye(3)) - nxn)
            D      = FthTensor2Voigt(D_alg)
            D      = D[ix_([0,1,3], [0,1,3])]

        # [4] Update Gauss point variables
        GPstate.eps_e  = Tensor2Voigt(eps_e, 'strain')
        GPstate.eps_p  = Tensor2Voigt(eps_p, 'strain')
        GPstate.lamda  = lamda
        GPstate.stress = Tensor2Voigt(sigma, 'stress')
        GPstate.D      = D

        return

    def ReturnMappingEigenSpace3D(self, GPstate):
        # Define elastic parameters
        E   = self.E
        nu  = self.nu
        lam = self.lam

        # Define derivatives (elastic)
        K  = E / (3.*(1.-2.*nu))
        mu = E / (2.*(1.+nu))

        aa = K + (4/3)*mu
        bb = K - (2/3)*mu

        dsig1depse1 = aa
        dsig1depse2 = bb
        dsig1depse3 = bb
        dsig2depse1 = bb
        dsig2depse2 = aa
        dsig2depse3 = bb
        dsig3depse1 = bb
        dsig3depse2 = bb
        dsig3depse3 = aa

        D_e = lam*tensor_oMult(np.eye(3),np.eye(3)) + 2.*mu*identity_4(3)
        D = FthTensor2Voigt(D_e)
         
        # Read previous elastic and plastic strains
        eps_e_n = Voigt2Tensor(GPstate.eps_e, 'strain')
        eps_p_n = Voigt2Tensor(GPstate.eps_p, 'strain')
        lamda   = GPstate.lamda

        deps = Voigt2Tensor(GPstate.deps, 'strain')

        # [1] Compute trial elastic strain
        eps_e_tr = eps_e_n + deps
        eps_e_tr_principal_mag, eps_e_tr_principal_vec = np.linalg.eigh(eps_e_tr)
        eps_e_tr_principal_mag, eps_e_tr_principal_vec = SortEig(eps_e_tr_principal_mag, eps_e_tr_principal_vec)

        eps_e_tr1 = eps_e_tr_principal_mag[0]
        eps_e_tr2 = eps_e_tr_principal_mag[1]
        eps_e_tr3 = eps_e_tr_principal_mag[2]

        n1 = eps_e_tr_principal_vec[:,0]
        n2 = eps_e_tr_principal_vec[:,1]
        n3 = eps_e_tr_principal_vec[:,2]

        # [2] Compute trial stress
        sigma1_tr = aa*eps_e_tr1 + bb*eps_e_tr2 + bb*eps_e_tr3 + self.HSP
        sigma2_tr = bb*eps_e_tr1 + aa*eps_e_tr2 + bb*eps_e_tr3 + self.HSP
        sigma3_tr = bb*eps_e_tr1 + bb*eps_e_tr2 + aa*eps_e_tr3 + self.HSP

        sigma_tr = sigma1_tr*np.tensordot(n1,n1,axes=0) + sigma2_tr*np.tensordot(n2,n2,axes=0) + sigma3_tr*np.tensordot(n3,n3,axes=0)

        # [3] Check yielding
        YF = self.PlasticityModel.f(sigma1_tr, sigma2_tr, sigma3_tr, lamda) # yield function

        if YF <= 0. or self.GlobalNRStep == 0:
            # Update stress & strain
            sigma = sigma_tr

            eps_e = eps_e_n + deps
            eps_p = eps_p_n
            eps   = eps_e + eps_p

        else:
            # Initialize variables
            eps_e_principal_mag, eps_e_principal_vec = np.linalg.eigh(eps_e_n)
            eps_e_principal_mag, eps_e_principal_vec = SortEig(eps_e_principal_mag, eps_e_principal_vec)

            eps_e1 = eps_e_principal_mag[0]
            eps_e2 = eps_e_principal_mag[1]
            eps_e3 = eps_e_principal_mag[2]
            dlamda  = 0

            x = np.zeros(4) # target variables
            x[0] = eps_e1
            x[1] = eps_e2
            x[2] = eps_e3
            x[3] = dlamda

            # Newton-Raphson iteration (return mapping)
            for ii in range(20):
                # Initialize residual and jacobian
                res = np.zeros(4)
                jac = np.zeros((4,4))

                # Current strain
                eps_e1_current = x[0]
                eps_e2_current = x[1]
                eps_e3_current = x[2]

                # Current stress
                sigma1_current = aa*eps_e1_current + bb*eps_e2_current + bb*eps_e3_current + self.HSP
                sigma2_current = bb*eps_e1_current + aa*eps_e2_current + bb*eps_e3_current + self.HSP
                sigma3_current = bb*eps_e1_current + bb*eps_e2_current + aa*eps_e3_current + self.HSP

                # Current lamda
                lamda_current = lamda + x[3]

                # Update derivatives
                # >> First order derivatives
                dfdsig1, dfdsig2, dfdsig3, dfdlamda \
                    = self.PlasticityModel.df(sigma1_current, sigma2_current, sigma3_current, lamda_current)

                # >> Second order derivatives
                d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1 \
                    = self.PlasticityModel.df2(sigma1_current, sigma2_current, sigma3_current)
                
                # Update residual
                res[0] = x[0] - eps_e_tr1 + x[3]*dfdsig1
                res[1] = x[1] - eps_e_tr2 + x[3]*dfdsig2
                res[2] = x[2] - eps_e_tr3 + x[3]*dfdsig3
                res[3] = self.PlasticityModel.f(sigma1_current, sigma2_current, sigma3_current, lamda_current)

                # Update Jacobian ***
                jac[0,0] = 1 + x[3]*(d2fdsig1dsig1*dsig1depse1 + d2fdsig1dsig2*dsig2depse1 + d2fdsig3dsig1*dsig3depse1)
                jac[0,1] =     x[3]*(d2fdsig1dsig1*dsig1depse2 + d2fdsig1dsig2*dsig2depse2 + d2fdsig3dsig1*dsig3depse2)
                jac[0,2] =     x[3]*(d2fdsig1dsig1*dsig1depse3 + d2fdsig1dsig2*dsig2depse3 + d2fdsig3dsig1*dsig3depse3)
                jac[0,3] = dfdsig1

                jac[1,0] =     x[3]*(d2fdsig1dsig2*dsig1depse1 + d2fdsig2dsig2*dsig2depse1 + d2fdsig2dsig3*dsig3depse1)
                jac[1,1] = 1 + x[3]*(d2fdsig1dsig2*dsig1depse2 + d2fdsig2dsig2*dsig2depse2 + d2fdsig2dsig3*dsig3depse2)
                jac[1,2] =     x[3]*(d2fdsig1dsig2*dsig1depse3 + d2fdsig2dsig2*dsig2depse3 + d2fdsig2dsig3*dsig3depse3)
                jac[1,3] = dfdsig2

                jac[2,0] =     x[3]*(d2fdsig3dsig1*dsig1depse1 + d2fdsig2dsig3*dsig2depse1 + d2fdsig3dsig3*dsig3depse1)
                jac[2,1] =     x[3]*(d2fdsig3dsig1*dsig1depse2 + d2fdsig2dsig3*dsig2depse2 + d2fdsig3dsig3*dsig3depse2)
                jac[2,2] = 1 + x[3]*(d2fdsig3dsig1*dsig1depse3 + d2fdsig2dsig3*dsig2depse3 + d2fdsig3dsig3*dsig3depse3)
                jac[2,3] = dfdsig3

                jac[3,0] = dfdsig1*dsig1depse1 + dfdsig2*dsig2depse1 + dfdsig3*dsig3depse1
                jac[3,1] = dfdsig1*dsig1depse2 + dfdsig2*dsig2depse2 + dfdsig3*dsig3depse2
                jac[3,2] = dfdsig1*dsig1depse3 + dfdsig2*dsig2depse3 + dfdsig3*dsig3depse3
                jac[3,3] = 0.0

                # Solve system of equations
                dx = np.linalg.solve(jac, -res) # increment of target variables

                # Update x
                x = x + dx

                # Compute error
                err = np.linalg.norm(dx)

                if err < 1e-7:
                    break

            # Update strain
            # >> total strain
            eps = eps_e_n + eps_p_n + deps

            # >> elastic strain
            eps_e = x[0]*np.tensordot(n1,n1,axes=0) + x[1]*np.tensordot(n2,n2,axes=0) + x[2]*np.tensordot(n3,n3,axes=0)

            # >> plastic strain
            eps_p = eps - eps_e
            lamda = lamda + x[3]

            # Update stress
            sigma1 = aa*x[0] + bb*x[1] + bb*x[2] + self.HSP
            sigma2 = bb*x[0] + aa*x[1] + bb*x[2] + self.HSP
            sigma3 = bb*x[0] + bb*x[1] + aa*x[2] + self.HSP
            sigma  = sigma1*np.tensordot(n1,n1,axes=0) + sigma2*np.tensordot(n2,n2,axes=0) + sigma3*np.tensordot(n3,n3,axes=0)

            # Update tangent
            deveps = eps_e - (1./3.)*np.trace(eps_e)*np.eye(3)
            n      = deveps / np.linalg.norm(deveps,'fro')
            nxn    = tensor_oMult(n,n)
            D_alg  = K*tensor_oMult(np.eye(3),np.eye(3)) + 2.0*mu*(identity_4(3)-(1./3.)*tensor_oMult(np.eye(3),np.eye(3)) - nxn)
            D      = FthTensor2Voigt(D_alg)

        # [4] Update Gauss point variables
        GPstate.eps_e  = Tensor2Voigt(eps_e, 'strain')
        GPstate.eps_p  = Tensor2Voigt(eps_p, 'strain')
        GPstate.lamda  = lamda
        GPstate.stress = Tensor2Voigt(sigma, 'stress')
        GPstate.D      = D

        return