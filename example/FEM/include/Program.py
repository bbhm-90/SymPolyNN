import numpy as np
import os.path
from dataclasses import dataclass

def ReadInput(input_name, Fem):
    print('#readinput')
    Node = NodeAttribute()
    Element = ElementAttribute()

    if not os.path.exists(input_name):
        ErrorMessage("Check the name of *.INPUT")

    file1 = open(input_name,'r')
    line = file1.readline().strip()

    if (line == "*Title"):
        line = file1.readline().strip()
        Fem.title = line
        Fem.LogGen('./example/FEM/log/' +Fem.title + '.dat')
        Fem.LogWrite('Input name: ' + input_name + '\n\n')
        Fem.LogWrite('#Readinput')
        Fem.PrintCommand('*Title',1)
        Fem.PrintCommand(Fem.title,2)
    else:
        print("First content in *.INPUT must be *Title")

    line = file1.readline()
    line = file1.readline().strip()
    while line:
        if (line == "*ResultDirectory"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            Fem.result = line
            Fem.PrintCommand(Fem.result, 2)
            if not os.path.exists(Fem.result):
                ErrorMessage("Check the result directory - *ResultDirectory in *.INPUT")
        
        elif (line == "*Mesh"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            NodeName = line + ".NODE"
            if not os.path.exists(NodeName):
                print(NodeName)
                ErrorMessage("Check the mesh name *Mesh in *.INPUT")
            Fem.PrintCommand(NodeName,2)
            fileNode = open(NodeName,'r')
            linetmp = fileNode.readline().strip()
            Node.NNode = int(linetmp)
            for ind in range(Node.NNode):    
                linetmp = fileNode.readline().strip()
                tmp = linetmp.replace(',', '\t')
                tmp = tmp.split('\t')
                Node.Id.append(int(tmp.pop(0)))
                Node.Coord.append(list(map(float, tmp)))
            Fem.Dimension = len(Node.Coord[0])
            
            ElemName = line + ".ELEM"
            if not os.path.exists(ElemName):
                print(ElemName)
                ErrorMessage("Check the mesh name *Mesh in *.INPUT")
            Fem.PrintCommand(ElemName,2)
            fileNode = open(ElemName,'r')
            linetmp = fileNode.readline().strip()
            Element.NElem = int(linetmp)
            for ind in range(Element.NElem):    
                linetmp = fileNode.readline().strip()
                tmp = linetmp.replace(',', '\t')
                tmp = tmp.split('\t')
                Element.Id.append(int(tmp.pop(0)))
                tmp = list(map(int, tmp))
                Element.Connectivity.append(tmp)

            if len(tmp) == 4 and Fem.Dimension == 2:
                Fem.ElmentType = 'q4'
                Fem.GP = [[-1./np.sqrt(3), -1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3), -1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3),  1./np.sqrt(3), 1.],
                          [-1./np.sqrt(3),  1./np.sqrt(3), 1.]]
            elif len(tmp) == 3:
                Fem.ElmentType = 't3'
                Fem.GP = [[1./3., 1./3., 1./2.]]
            elif len(tmp) == 8:
                Fem.ElmentType = 'hex8'
                Fem.GP = [[-1./np.sqrt(3.), -1./np.sqrt(3.),  1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3.), -1./np.sqrt(3.),  1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3.),  1./np.sqrt(3.),  1./np.sqrt(3), 1.],
                          [-1./np.sqrt(3.),  1./np.sqrt(3.),  1./np.sqrt(3), 1.],
                          [-1./np.sqrt(3.), -1./np.sqrt(3.), -1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3.), -1./np.sqrt(3.), -1./np.sqrt(3), 1.],
                          [ 1./np.sqrt(3.),  1./np.sqrt(3.), -1./np.sqrt(3), 1.],
                          [-1./np.sqrt(3.),  1./np.sqrt(3.), -1./np.sqrt(3), 1.]]
            elif len(tmp) == 4 and Fem.Dimension == 3:
                Fem.ElmentType = 'tet4'
                Fem.GP = [[1./4., 1./4., 1./4., 1./6.]]
            else:
                assert 0, "Element type not supported"

        elif (line == "*PlasticModel"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            Fem.PrintCommand(line,2)
            Fem.PlasticModel = line

        elif (line == "*ConstitutiveLaw"):
            Fem.PrintCommand(line,1)
            ReadConstitutiveLaw(file1, Fem)

        elif (line == "*InitialPressure"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            Fem.InitialStress = 1
            Fem.HSP = float(line)
            Fem.PrintCommand(str(Fem.HSP),2)

        elif (line == "*LoadingStep"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            Fem.totalstep = int(line)
            Fem.PrintCommand(str(Fem.totalstep),2)
            
        elif (line == "*BoundaryCondition" or line == "*BC"):
            Fem.PrintCommand(line,1)
            exit(1)

        elif (line == "*BCFile"):
            Fem.PrintCommand(line,1)
            Fem.BCFile = True
            line = file1.readline().strip()
            Fem.PrintCommand(line,2)
            Fem.BCFileDirectory = line

        elif (line == "*TimeIntegration"):
            Fem.PrintCommand(line,1)
            line = file1.readline().strip()
            Fem.timeintegration = line
            line = file1.readline().strip()
            line = line.replace(" ", "")
            line = line.replace(',', '\t')
            tmp = line.split('\t')
            Fem.totalstep  = int(tmp[0])
            Fem.totaltime  = float(tmp[1])
            Fem.dt         = (Fem.totaltime/float(Fem.totalstep))
            Fem.PrintCommand("Total step : " + str(Fem.totalstep ),2)
            Fem.PrintCommand("Total time : " + str(Fem.totaltime),2)
            Fem.PrintCommand("dt         : " + str(Fem.dt),2)
            
        else:
            print(line)
            print("Check input file")
            exit(1)

        line = file1.readline().strip()
        count = 0
        while True:
            if(line == ""):
                line = file1.readline().strip()
                count +=1
            else:
                break
            if count == 5:
                break

    Node.BC_E       = np.empty([Node.NNode*Fem.Dimension])
    Node.BC_E[:]    = np.NaN
    Node.u          = np.zeros([Node.NNode*Fem.Dimension])
    Node.u1         = np.zeros([Node.NNode*Fem.Dimension])
    Node.du         = np.zeros([Node.NNode*Fem.Dimension])
    Node.BC_N       = np.zeros([Node.NNode*Fem.Dimension])
    Node.BC_N_init  = np.zeros([Node.NNode*Fem.Dimension])
    Node.F_int      = np.zeros([Node.NNode*Fem.Dimension])
    Node.F_ext      = np.zeros([Node.NNode*Fem.Dimension])
    Node.stress  = []
    Node.sigmaVM = np.zeros([Node.NNode])
    Fem.G_Edof = Fem.Dimension * len(Element.Connectivity[0])
    file1.close
    return Node, Element

class NodeAttribute:
    Coord = []
    u: np.array
    u1: np.array
    du: np.array
    BC_E: np.array
    BC_N: np.array
    BC_N_init: np.array
    F_int: np.array    # internal force
    F_ext: np.array    # external force
    stress: np.array   # nodal Cauchy stress
    strain_e: np.array # nodal elastic strain
    strain_p: np.array # nodal plastic strain
    sigmaVM:np.array   # nodal von Mises stress
    NNode: int
    Id = []
    
class ElementAttribute:
    Connectivity = []
    Stiffness = []
    B_matrix  = []
    NElem : int
    Id = []         # element id
    Area = []
    GPstrain_e = [] # elastic strain
    GPstrain_p = [] # plastic strain
    GPstrain = []
    GPstress = []
    Jacc = []
    
class Model:
    title: str = ''
    ElmentType: str = ''
    result: str = ''
    PlasticModel: str = ''
    analysis: str = ''
    dt: float = 0.
    InitialStress: int = 0
    totalstep: int = 1
    step: int = 0
    totaltime: float = 0.
    Dimension: int = 2
    BCFileDirectory: str=''
    BCFile: bool=False
    HSP: float = 0.
    EBC = []
    NBC = []
    GP = []

    E: float             # Elastic modulus 
    nu: float            # poisson ratio
    lamda: float         # 1st lame parameter
    mu: float            # 2nd lame parameter
    K: float
    nu: float

    def LogGen(self, logname):
        self.file_log = open(logname,'w')
    
    def LogWrite(self, command):
        self.file_log.write(command)
        
    def LogClose(self):
        self.file_log.close
        
    def show(self):
        print("============= Model description =============")
        print("Model")
        print("\ttitle   : ",self.title)
        print("\tresult  : ",self.result,"\n")
        print("Solid")
        print("\tE    = ",self.E,"\tElastic modulus")
        print("\tv    = ",self.nu,"\t\tPoisson ratio")
        print("\tlamda= ",self.lamda,"\t\t1st Lame parameter")
        print("\tmu   = ",self.mu,"\t\t2nd Lame parameter \n")
        print("============= Model description =============")
    def PrintCommand(self, command, Nindent):
        command.strip('\n')
        for i in range(0,Nindent):
            command = '\t' + command
        print(command)
        self.LogWrite(command+'\n')
        return

def ErrorMessage(command):
    command ="\t Error : " + command
    print(command)
    assert 0,command
    exit(1)
    return

def ReadConstitutiveLaw(file1, Fem):
    line = file1.readline().strip()
    Fem.PrintCommand(line, 2)
    while line:
        if(line.lower() == 'solid'):
            line = file1.readline().strip()
            Fem.PrintCommand(line, 2)
            if(line.lower() == 'elastoplasticity'):
                line = file1.readline().strip()
                line = line.replace(',', '\t')
                tmp = line.split('\t')
                Fem.E  = float(tmp[0])
                Fem.nu = float(tmp[1])
                Fem.MatProp  = []
                for ii in tmp:
                    Fem.MatProp.append(float(ii))
                for ii, param in enumerate(Fem.MatProp):
                    Fem.PrintCommand("Parameter "+str(ii+1)+": " + str(param),3)
                line = file1.readline().strip()
                Fem.PrintCommand("(1st and 2nd parameters are: E and v)",3)

                E  = Fem.E
                nu = Fem.nu
                Fem.lamda = E*nu / ((1.0+nu)*(1.0-2.0*nu))
                Fem.K     = E / (3*(1.0-2.0*nu))
                Fem.mu    = E / (2.0*(1.0+nu))
            else:
                print("Check *ConstitutiveLaw in *.INPUT") 
                exit(1)
        else:
            print(line)
            print("Check *ConstitutiveLaw in *.INPUT") 
            exit(1)
    return