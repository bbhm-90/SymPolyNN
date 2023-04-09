import autograd.numpy as np
from autograd import elementwise_grad as egrad

class MyPlasticity():
    def __init__(self, Fem):
        self.E  = Fem.E
        self.nu = Fem.nu
        self.phi = np.radians(Fem.MatProp[2])

        self.get_dfdsig1 = egrad(self.f, 0)
        self.get_dfdsig2 = egrad(self.f, 1)
        self.get_dfdsig3 = egrad(self.f, 2)

    def f(self, sigma1, sigma2, sigma3, lamda):

        sigma1 = -sigma1
        sigma2 = -sigma2
        sigma3 = -sigma3
        
        I1 = sigma1 + sigma2 + sigma3
        I2 = sigma1*sigma2 + sigma2*sigma3 + sigma3*sigma1
        I3 = sigma1*sigma2*sigma3

        beta = (9 - np.sin(self.phi)**2)/(1 - np.sin(self.phi)**2)

        f = np.abs(I1*I2)**(1./3) - np.abs(beta*I3)**(1./3)

        return f

    def df(self, sigma1, sigma2, sigma3, lamda):

        dfdsig1  = self.get_dfdsig1(sigma1, sigma2, sigma3, lamda)
        dfdsig2  = self.get_dfdsig2(sigma1, sigma2, sigma3, lamda)
        dfdsig3  = self.get_dfdsig3(sigma1, sigma2, sigma3, lamda)
        dfdlamda = 0.0

        norm = np.sqrt(dfdsig1**2 + dfdsig2**2 + dfdsig3**2)
        dfdsig1 = dfdsig1 / norm
        dfdsig2 = dfdsig2 / norm
        dfdsig3 = dfdsig3 / norm

        return dfdsig1, dfdsig2, dfdsig3, dfdlamda

    def df2(self, sigma1, sigma2, sigma3):

        dist = 1e-3
        dfdsig1, dfdsig2, dfdsig3, _ = self.df(sigma1, sigma2, sigma3, 0.0)

        dfdsig1_s1dist, dfdsig2_s1dist, dfdsig3_s1dist, _ = self.df(sigma1+dist, sigma2, sigma3, 0.0)
        dfdsig1_s2dist, dfdsig2_s2dist, dfdsig3_s2dist, _ = self.df(sigma1, sigma2+dist, sigma3, 0.0)
        dfdsig1_s3dist, dfdsig2_s3dist, dfdsig3_s3dist, _ = self.df(sigma1, sigma2, sigma3+dist, 0.0)

        d2fdsig1dsig1 = (dfdsig1_s1dist - dfdsig1) / dist
        d2fdsig2dsig2 = (dfdsig2_s2dist - dfdsig2) / dist
        d2fdsig3dsig3 = (dfdsig3_s3dist - dfdsig3) / dist

        d2fdsig1dsig2 = (dfdsig1_s2dist - dfdsig1) / dist
        d2fdsig2dsig3 = (dfdsig2_s3dist - dfdsig2) / dist
        d2fdsig3dsig1 = (dfdsig3_s1dist - dfdsig3) / dist

        return d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1
