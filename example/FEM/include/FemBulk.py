import numpy as np
from numpy import ix_

def GetGPSR(Fem):
    GP_SR = []
    Dim = Fem.Dimension
    if Dim == 2:
        if Fem.ElmentType == "q4":
            for gp in Fem.GP:
                N, _ = ShapeFunction(Fem,1./gp[0], 1./gp[1])
                GP_SR.append(N)
            GP_SR = np.array(GP_SR)
        elif Fem.ElmentType == "t3":
            N = [[1.],
                 [1.],
                 [1.]]
            GP_SR = np.array(N)
        else:
            print("Element type: ", Fem.ElmentType)
            assert False, "Element type not supported"
    elif Dim ==3:
        if Fem.ElmentType == "hex8":
            for gp in Fem.GP:
                N, _ = ShapeFunction(Fem,1./gp[0], 1./gp[1], 1./gp[2])
                GP_SR.append(N)
            GP_SR = np.array(GP_SR)
        elif Fem.ElmentType == "tet4":
            N = [[1.],
                 [1.],
                 [1.],
                 [1.]]
            GP_SR = np.array(N)
        else:
            assert False, "Element type not supported"
    else:
        assert False, "Dimension error"
    return GP_SR

def CalculatePlasticMultiplierAtNode(Fem, Node, Element):
    G_Edof = Fem.G_Edof
    Dim    = Fem.Dimension
    LocalNElem  = np.zeros([Node.NNode])
    Node.PlasticMultiplier = np.zeros((Node.NNode))

    GP_SR= GetGPSR(Fem)

    for ind1, Edof in enumerate(Element.Connectivity):
        ENdof = ElementalNodeDof(Fem, Node, Edof)
        Ndof  = NodeDof(Node, Edof)

        ElementGPlamda1 = np.array(Element.GPlamda[ind1])

        Node.PlasticMultiplier[Ndof] += GP_SR @ ElementGPlamda1
        LocalNElem[Ndof] += 1

    Node.PlasticMultiplier = Node.PlasticMultiplier / LocalNElem

def CalculateVonMisesStressAtNode(Fem, Node):
    if Fem.Dimension == 2:
        x  = Node.stress[:,0]
        y  = Node.stress[:,1]
        z  = np.zeros((Node.NNode))+Fem.HSP
        xy = Node.stress[:,2]
        yz = np.zeros((Node.NNode))
        zx = np.zeros((Node.NNode))
    elif Fem.Dimension == 3:
        x  = Node.stress[:,0]
        y  = Node.stress[:,1]
        z  = Node.stress[:,2]
        xy = Node.stress[:,3]
        yz = Node.stress[:,4]
        zx = Node.stress[:,5]
    else:
        assert 0, "Dimension error"

    Node.sigmaVM = np.sqrt(0.5*((x-y)**2 + (y-z)**2 + (z-x)**2 + 6.0*(xy**2 + yz**2 + zx**2)))
    return

def CalculateStrainAtNode(Fem, Node, Element):
    G_Edof = Fem.G_Edof
    Dim    = Fem.Dimension
    LocalNElem  = np.zeros([Node.NNode])
    if Dim == 2:
        Node.strain_e = np.zeros((Node.NNode,3))
        Node.strain_p = np.zeros((Node.NNode,3))
        Dim2 = 3
    elif Dim ==3:
        Node.strain_e = np.zeros((Node.NNode,6))
        Node.strain_p = np.zeros((Node.NNode,6))
        Dim2 = 6
    else:
        assert 0, "Dimension error"

    GP_SR = GetGPSR(Fem)

    for ind1, Edof in enumerate(Element.Connectivity):

        ENdof = ElementalNodeDof(Fem, Node, Edof)
        Ndof  = NodeDof(Node, Edof)

        ElementGPstrain_e = np.array(Element.GPstrain_e[ind1])
        ElementGPstrain_p = np.array(Element.GPstrain_p[ind1])
        Node.strain_e[Ndof,:] += GP_SR @ ElementGPstrain_e
        Node.strain_p[Ndof,:] += GP_SR @ ElementGPstrain_p
        LocalNElem[Ndof] += 1

    for ii in range(Dim2):
        Node.strain_e[:,ii] = Node.strain_e[:,ii]/LocalNElem
        Node.strain_p[:,ii] = Node.strain_p[:,ii]/LocalNElem
    return

def CalculateStressAtNode(Fem, Node, Element):
    G_Edof = Fem.G_Edof
    Dim    = Fem.Dimension
    if Fem.Dimension == 2:
        Node.stress = np.zeros([Node.NNode,3])
        Dim2 = 3
    elif Fem.Dimension == 3:
        Node.stress = np.zeros([Node.NNode,6])
        Dim2 = 6
    else:
        assert 0, "Dimension error"
    LocalNElem  = np.zeros([Node.NNode])

    GP_SR = GetGPSR(Fem)

    for ind1, Edof in enumerate(Element.Connectivity):

        ENdof = ElementalNodeDof(Fem, Node, Edof)
        Ndof  = NodeDof(Node, Edof)

        ElementGPstress = np.array(Element.GPstress[ind1])
        Node.stress[Ndof,:] += GP_SR @ ElementGPstress
        LocalNElem[Ndof] += 1

    for ii in range(Dim2):
        Node.stress[:,ii] = Node.stress[:,ii]/LocalNElem

def Assembleage(Fem, Node, Element):

    Global_K = np.zeros([Node.NNode * Fem.Dimension, Node.NNode * Fem.Dimension])
    for EID, Edof in enumerate(Element.Connectivity):

        ENdof = ElementalNodeDof(Fem, Node, Edof)
        Ldof = np.arange(len(ENdof))
        Global_K[ix_(ENdof,ENdof)] += Element.Stiffness[EID][ix_(Ldof,Ldof)]

    return Global_K

def BC_SetUpAtStep(Fem, Node):
    IndexBCN = list(np.where(np.isnan(Node.BC_E))[0])
    IndexBCE = list(np.where(np.invert(np.isnan(Node.BC_E)))[0])

    Node.F_ext            = np.zeros_like(Node.BC_N)
    Node.F_ext[IndexBCN]  = Node.BC_N_init[IndexBCN]
    Node.F_ext[IndexBCN] += Node.BC_N[IndexBCN] * Fem.step

    Node.du = np.zeros_like(Node.du)
    if Fem.step != 0:
        Node.du[IndexBCE] = Node.BC_E[IndexBCE]
    Node.u[IndexBCE] += Node.du[IndexBCE]
    return

def ElementalNodeDof(Fem, Node, Connectivity):
    tmp = NodeDof(Node, Connectivity)
    if Fem.Dimension == 2:
        tmp1 = np.array(tmp)*2
        tmp2 = np.array(tmp)*2 + 1
        c = np.empty((tmp1.size + tmp2.size), dtype =tmp1.dtype)
        c[0::2] = tmp1
        c[1::2] = tmp2
    elif Fem.Dimension == 3:
        tmp1 = np.array(tmp)*3
        tmp2 = np.array(tmp)*3 + 1
        tmp3 = np.array(tmp)*3 + 2
        c = np.empty((tmp1.size + tmp2.size + tmp3.size), dtype =tmp1.dtype)
        c[0::3] = tmp1
        c[1::3] = tmp2
        c[2::3] = tmp3
    else:
        assert(0,"Dimension error")
    return c

def NodeDof(Node, Connectivity):
    Ndof1 =[]
    for ii in Connectivity:
        Ndof1.append(Node.Id.index(ii))
    return Ndof1

def ShapeFunction(Fem, s,t,u=0.):
    if Fem.ElmentType == "q4":
        N_matrix = np.array([( s-1.)*( t-1.)/4.,\
                             (-s-1.)*( t-1.)/4.,\
                             (-s-1.)*(-t-1.)/4.,\
                             ( s-1.)*(-t-1.)/4.])

        dN   = np.array([[t-1., 1.-t, 1.+t, -1.-t],
                         [s-1., -1.-s, 1.+s, 1.-s]])
        dN *= 0.25

    elif Fem.ElmentType == 't3':
        N_matrix = np.array([s,\
                             t,\
                             1.-s-t])

        dN   = np.array([[1., 0., -1.],
                         [0., 1., -1.]])

    elif Fem.ElmentType == 'hex8':
        N_matrix = np.zeros((8))
        N_matrix[0] = (1.-s)*(1.-t)*(1.+u)
        N_matrix[1] = (1.-s)*(1.-t)*(1.-u)
        N_matrix[2] = (1.-s)*(1.+t)*(1.-u)
        N_matrix[3] = (1.-s)*(1.+t)*(1.+u)
        N_matrix[4] = (1.+s)*(1.-t)*(1.+u)
        N_matrix[5] = (1.+s)*(1.-t)*(1.-u)
        N_matrix[6] = (1.+s)*(1.+t)*(1.-u)
        N_matrix[7] = (1.+s)*(1.+t)*(1.+u)

        N_matrix *= 0.125

        dN = np.array([[-(1.-t)*(1.+u), -(1.-t)*(1.-u), -(1.+t)*(1.-u), -(1.+t)*(1.+u),  (1.-t)*(1.+u),  (1.-t)*(1.-u),  (1.+t)*(1.-u),  (1.+t)*(1.+u)],
                       [-(1.-s)*(1.+u), -(1.-s)*(1.-u),  (1.-s)*(1.-u),  (1.-s)*(1.+u), -(1.+s)*(1.+u), -(1.+s)*(1.-u),  (1.+s)*(1.-u),  (1.+s)*(1.+u)],
                       [ (1.-s)*(1.-t), -(1.-s)*(1.-t), -(1.-s)*(1.+t),  (1.-s)*(1.+t),  (1.+s)*(1.-t), -(1.+s)*(1.-t), -(1.+s)*(1.+t),  (1.+s)*(1.+t)]])
        dN *= 0.125

    elif Fem.ElmentType == 'tet4':
        N_matrix = np.array([s,\
                             t,\
                             u,\
                             1.-s-t-u])

        dN   = np.array([[1., 0., 0.,-1.],
                         [0., 1., 0.,-1.],
                         [0., 0., 1.,-1.]])

    else:
        assert 0,"Element type not supported"

    return N_matrix, dN 

def Strain(Fem, NodeTarget,s,t,u=0.):
    G_Edof = Fem.G_Edof
    Dim    = Fem.Dimension
    _, dN = ShapeFunction(Fem,s,t,u)
    Jacobian = np.matmul(dN, NodeTarget)

    if Dim == 2:
        B1= np.array([[1., 0., 0., 0.],
                      [0., 0., 0., 1],
                      [0., 1., 1., 0]])

        B2 = np.zeros([4,4])

        B2[2:4,2:4] = np.linalg.inv(Jacobian)
        B2[0:2,0:2] = np.linalg.inv(Jacobian)

        B3 = np.zeros([4,G_Edof])

        for ind in range(int(G_Edof/Dim)):
            B3[0:2,2*ind]   = dN[:,ind]
            B3[2:4,2*ind+1] = dN[:,ind]
    else:
        B1= np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 1.],
                      [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 1., 0.],
                      [0., 0., 1., 0., 0., 0., 1., 0., 0.]])

        B2 = np.zeros([9,9])

        B2[0:3,0:3] = np.linalg.inv(Jacobian)
        B2[3:6,3:6] = np.linalg.inv(Jacobian)
        B2[6:9,6:9] = np.linalg.inv(Jacobian)

        B3 = np.zeros([9,G_Edof])

        for ind in range(int(G_Edof/Dim)):
            B3[0    :Dim*1,Dim*ind]   = dN[:,ind]
            B3[Dim*1:Dim*2,Dim*ind+1] = dN[:,ind]
            B3[Dim*2:Dim*3,Dim*ind+2] = dN[:,ind]

        
    B_matrix= B1 @ B2 @ B3
    Jacc = np.linalg.det(Jacobian)

    return Jacc, B_matrix