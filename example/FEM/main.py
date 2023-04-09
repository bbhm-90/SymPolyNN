import sys
sys.path.append("./example/FEM/include")

import numpy as np
import time as ct
import Program
import WriteResult
import ElastoPlasticity as Plasticity
import FemBulk as Bulk
import importlib

from MeshHandle import *


# Setup max. iter & tol. for FE simulation
max_itr = 20
toler   = 1e-7
tic = ct.time()
assert len(sys.argv) == 2, "How to run FE simulation:\n\t python ./examples/FEM/main.py ./examples/FEM/input/PlateWithHole2D_benchmark.INPUT"


# Model generation
Fem = Program.Model()
input_name = sys.argv[1]


# Read input
Node, Element = Program.ReadInput(input_name, Fem)
print(" ")
print("============================================================================")
print("Finite element analysis (higher-order-NAM-based plasticity)")
print("============================================================================")
ct.sleep(2)


# FE analysis
WriteResult.WritePVD(Fem)
Plasticity.ElementSetUp(Fem, Node, Element)
ConstitutiveModel = Plasticity.ConstitutiveLaw(Fem)


# Setup output
if Fem.Dimension == 2:
    WriteAttributes = np.zeros((Fem.totalstep,10))
elif Fem.Dimension == 3:
    WriteAttributes = np.zeros((Fem.totalstep,17))


# Assign boundary conditions
if Fem.BCFile == True:
    tmp = importlib.import_module(Fem.BCFileDirectory)
    tmp.ApplyBC(Fem, Node, Element)
else:
    assert False, "B.C. not properly applied"


# Global Newton-Raphson iteration
for step in range(Fem.totalstep):
    Fem.step = step +1
    Bulk.BC_SetUpAtStep(Fem, Node)

    Fem.PrintCommand("Loading step: "+str(Fem.step),0)

    for ii in range(max_itr):
        ConstitutiveModel.GlobalNRStep = ii
        Plasticity.ConstructStiffness(Fem, Node, Element, ConstitutiveModel)
        Global_K = Bulk.Assembleage(Fem, Node, Element)
        res = Plasticity.Solve(Node, Global_K)
        Fem.PrintCommand("  err. >> "+str(res),0)
        if res < toler:
            break
    print("----------------------------------------------------------------------------")

    Bulk.CalculateStrainAtNode(Fem, Node, Element)
    Bulk.CalculateStressAtNode(Fem, Node, Element)
    Bulk.CalculateVonMisesStressAtNode(Fem, Node)
    Bulk.CalculatePlasticMultiplierAtNode(Fem, Node, Element)

    WriteResult.WriteVTU(Fem, Node, Element)


# End of program
toc = ct.time() - tic
Fem.PrintCommand(' ',0)
Fem.PrintCommand("Elapsed CPU time: "+str(toc)+"[sec]",0)
Fem.PrintCommand("Log file saved at: ./examples/FEM/log/" +Fem.title + ".dat",0)
Fem.LogClose()