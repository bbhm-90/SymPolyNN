import numpy as np

def SortEig2D(eigenvalue, eigenvector):
    n1 = np.copy(eigenvector[:,0])
    n2 = np.copy(eigenvector[:,1])

    error_list = []
    error_list.append(np.linalg.norm(n1 - np.array([1.,0.])))
    error_list.append(np.linalg.norm(n1 - np.array([-1.,0.])))
    error_list.append(np.linalg.norm(n2 - np.array([1.,0.])))
    error_list.append(np.linalg.norm(n2 - np.array([-1.,0.])))

    D = np.zeros((2))
    V = np.zeros((2,2))
    ind1 = np.argmin(np.array(error_list))
    if ind1 == 0 or ind1 == 1:
        V[:,0] = n1
        V[:,1] = n2
        D[0]   = eigenvalue[0]
        D[1]   = eigenvalue[1]
    else:
        V[:,0] = n2
        V[:,1] = n1
        D[0]   = eigenvalue[1]
        D[1]   = eigenvalue[0]
    
    return D, V

def SortEig(eigenvalue, eigenvector, alpha_prev=0., beta_prev=0., gamma_prev=0.):
    n1 = eigenvector[:,0]
    n2 = eigenvector[:,1]
    n3 = np.cross(n1,n2)

    vc_prev = rotation_matrix(alpha_prev, beta_prev, gamma_prev)

    if np.linalg.norm(np.cross(n1,n2)- n3) > 1e-4:
        print(eigenvector)
        assert False, "Check right hand rule"

    combination_vl   = []
    combination_vc   = []
    ErrorMeasureList = []

    combination_vl.append([0,1,2])
    combination_vl.append([0,2,1])
    combination_vl.append([1,0,2])
    combination_vl.append([1,2,0])
    combination_vl.append([2,0,1])
    combination_vl.append([2,1,0])

    combination_vc.append([1,1,1])
    combination_vc.append([1,-1,-1])
    combination_vc.append([-1,1,-1])
    combination_vc.append([-1,-1,1])

    vc      = np.zeros((3,3))
    vc[:,0] = n1
    vc[:,1] = n2
    vc[:,2] = n3
    vl      = np.copy(eigenvalue)
    vc_tmp  = np.zeros_like(eigenvector)

    for jj in combination_vl:
        for combi in combination_vc:
            
            vc_tmp = np.copy(vc.T[jj].T)
            vc_tmp[:,2] = np.cross(vc_tmp[:,0],vc_tmp[:,1])

            vc_tmp[:,0] *= combi[0]
            vc_tmp[:,1] *= combi[1]
            vc_tmp[:,2] *= combi[2]

            err = np.linalg.norm(vc_prev - vc_tmp)
            ErrorMeasureList.append(err)

    ind1 = np.argmin(np.array(ErrorMeasureList))
    ind2 = divmod(ind1 ,4)[0]
    ind3 = ind1 % 4
    D = np.copy(vl[combination_vl[ind2]])
    V = np.copy(vc.T[combination_vl[ind2]].T)
    V[:,2] = np.cross(V[:,0], V[:,1])

    combi   = combination_vc[ind3]
    V[:,0] *= combi[0]
    V[:,1] *= combi[1]
    V[:,2] *= combi[2]

    return D, V

def rotation_matrix(theta1, theta2, theta3):
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)

    Ra = np.array([[1., 0., 0.],
                   [0., c1,-s1],
                   [0., s1, c1]])

    Rb = np.array([[ c2, 0., s2],
                   [ 0., 1., 0.],
                   [-s2, 0., c2]])

    Rc = np.array([[c3,-s3, 0.],
                   [s3, c3, 0.],
                   [0., 0., 1.]])

    matrix = Ra @ Rb @ Rc

    return matrix