# How to run FE simulation

1) Mesh generation
 > Generate mesh either by using Gmsh (*.msh) or ABAQUS (*.inp).
 > If your mesh is generated via ABAQUS, convert *.inp to *.msh (see MeshGen_readme.txt)
 > Convert *.msh to *.NODE and *.ELEM files by executing generate_mesh.py in ./example/FEM/MeshGen
 > After executing the Python script, your mesh files are saved in ./example/FEM/Mesh
 > e.g., ./example/FEM/Mesh/PerforatedPlate2d

2) Specify boundary conditions
 > Boundary conditions are handled via Python scripts in ./example/FEM/include/BC
 > e.g., BC_PlateWithHole2D.py

3) FE input file
 > All necessary informations including material properties should be specified in *.INPUT file.
 > Specify: Title, ConstitutiveLaw, LoadingStep, BCFile, InitialPressure, ResultDirectory, Mesh, PlasticModel.
 > e.g., PlateWithHole2D_ho_symb.INPUT

4) Run FE analysis
 > e.g., python ./example/FEM/main.py ./example/FEM/input/PlateWithHole2D_ho_symb.INPUT