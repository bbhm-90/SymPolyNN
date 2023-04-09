import os
import pygmsh
import meshio
import numpy as np

# file name
file_name = "Column2d"

def WriteNodeAndElementFromMsh(directory, Node, Element, Dim=2):

    NNode = Node.shape[0]
    NElem = Element.shape[0]
    fid1 = open(directory+'.NODE', "w")
    fid1.write("%s\n" % (len(Node)))

    for ii, node in enumerate(Node):
        if Dim == 2:
            fid1.write("%s\t%s\t%s\n" %(ii+1, node[0], node[1]))
        elif Dim == 3:
            fid1.write("%s\t%s\t%s\t%s\n" %(ii+1, node[0], node[1], node[2]))
        else:
            assert false, "Check the 4th input 'Dim'"


    fid1.close()

    fid2 = open(directory+'.ELEM', "w")
    fid2.write("%s\n" % (len(Element)))
    for ii, Connectivity in enumerate(Element):
        fid2.write("%s\t" % (ii+1))
        for jj in Connectivity:
            fid2.write("%s\t" % (jj+1))
        fid2.write("\n")
    fid2.close()

    return

# Read mesh
msh = meshio.read("./"+file_name+".msh")
print(msh)

Node = msh.points
Element = msh.cells[0].data

# Write .NODE and .ELEM
isExist = os.path.exists("../Mesh/"+file_name)
if not isExist:
  os.makedirs("../Mesh/"+file_name)

WriteNodeAndElementFromMsh("../Mesh/"+file_name+"/"+file_name, Node, Element)