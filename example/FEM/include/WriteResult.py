import numpy as np
import meshio

def WritePVD(Fem):
    name = Fem.result+'/'+Fem.title+'.pvd'
    fid1 = open(name, "w")
    fid1.write("<?xml version=\"1.0\"?>\n")
    fid1.write("<VTKFile type=\"Collection\" version=\"0.1\">\n")
    fid1.write("  <Collection>\n")
    for ii in range(Fem.totalstep):
        tmp = (ii+1)/Fem.totalstep
        fid1.write("    <DataSet timestep=\""+str(tmp)+"\" part=\"0\" file=\""+Fem.title+"_"+str(ii+1)+".vtu\" />\n")
    fid1.write("  </Collection>\n")
    fid1.write("</VTKFile>\n")
    fid1.close()

def WriteVTU(Fem, Node, Element, att=None):
    points = Node.Coord

    if Fem.ElmentType == 'hex8':
        cells = [("hexahedron", (np.array(Element.Connectivity)-1).tolist())]

    elif Fem.ElmentType == 'tet4':
        cells = [("tetra", (np.array(Element.Connectivity)-1).tolist())]

    elif Fem.ElmentType == 't3':
        cells = [("triangle", (np.array(Element.Connectivity)-1).tolist())]

    elif Fem.ElmentType == 'q4':
        cells = [("quad", (np.array(Element.Connectivity)-1).tolist())]
    else:
        print("Fem.ElmentType",Fem.ElmentType)
        assert 0, "Element type not supported"

    if Fem.Dimension == 3:
        mesh = meshio.Mesh(
            points,
            cells,
            point_data={"von Mises stress": Node.sigmaVM,
                        "Accumulated plastic strain": Node.PlasticMultiplier,
                        "u_1": Node.u[0::3],
                        "u_2": Node.u[1::3],
                        "u_3": Node.u[2::3],
                        "eps_11": Node.strain_e[:,0]+Node.strain_p[:,0],
                        "eps_22": Node.strain_e[:,1]+Node.strain_p[:,1],
                        "eps_33": Node.strain_e[:,2]+Node.strain_p[:,2],
                        "eps_12": Node.strain_e[:,3]+Node.strain_p[:,3],
                        "eps_23": Node.strain_e[:,4]+Node.strain_p[:,4],
                        "eps_31": Node.strain_e[:,5]+Node.strain_p[:,5],
                        "sigma_11": Node.stress[:,0],
                        "sigma_22": Node.stress[:,1],
                        "sigma_33": Node.stress[:,2],
                        "sigma_12": Node.stress[:,3],
                        "sigma_23": Node.stress[:,4],
                        "sigma_31": Node.stress[:,5]})
    elif Fem.Dimension == 2:
        mesh = meshio.Mesh(
            points,
            cells,
            point_data={"von Mises stress": Node.sigmaVM,
                        "Accumulated plastic strain": Node.PlasticMultiplier,
                        "u_1": Node.u[0::2],
                        "u_2": Node.u[1::2],
                        "eps_11": Node.strain_e[:,0]+Node.strain_p[:,0],
                        "eps_22": Node.strain_e[:,1]+Node.strain_p[:,1],
                        "eps_12": Node.strain_e[:,2]+Node.strain_p[:,2],
                        "sigma_11": Node.stress[:,0],
                        "sigma_22": Node.stress[:,1],
                        "sigma_12": Node.stress[:,2]})

    mesh.write(Fem.result+'/'+Fem.title+'_'+str(Fem.step)+".vtu")
    return