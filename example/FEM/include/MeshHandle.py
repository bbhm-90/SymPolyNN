import numpy as np

class EdgeAttribute:
    AdjacFacet = []
    AdjacNode = []

class FacetAttribute:
    AdjacElem = []
    AdjacNode = []

def GenerateFacet(Fem, Element):
    Facet = FacetAttribute()
    if Fem.Dimension == 2:
        for kk in range(Element.NElem):
            elem = Element.Connectivity[kk]
            elem = elem.copy()
            elem.append(elem[0])
            for ii in range(len(elem)-1):
                flag = -1
                for jj, EN in enumerate(Facet.AdjacNode):
                    if EN[0] == elem[ii+1] and EN[1] == elem[ii]:
                        flag = jj
                        break

                if flag == -1:
                    Facet.AdjacNode.append([elem[ii], elem[ii+1]])
                    Facet.AdjacElem.append([Element.Id[kk],-1])
                else:
                    Facet.AdjacElem[flag][1] = Element.Id[kk]

    elif Fem.Dimension==3:
        for kk in range(Element.NElem):
            elem = Element.Connectivity[kk]
            elem = np.array(elem.copy())
            if Fem.ElmentType == 'hex8':
                facet_list = [[4,5,6,7],
                              [5,1,2,6],
                              [1,0,3,2],
                              [0,4,7,3],
                              [7,6,2,3],
                              [0,1,5,4]]
            else:
                assert False, "Fem.ElmentType not ready"
            for LFacet in facet_list:
                GFacet = list(elem[LFacet])
                flag = -1
                circular_shift = []
                circular_shift.append(GFacet.copy())
                for jj in range(len(GFacet)-1):
                    GFacet.append(GFacet.pop(0))
                    tmp = GFacet.copy()
                    circular_shift.append(tmp)

                print("GFacet\t",np.array(GFacet)-1)
                for jj, EN in enumerate(Facet.AdjacNode):
                    tmp = list(reversed(EN))
                    if tmp in circular_shift:
                        flag = jj
                        break

                if flag == -1:
                    Facet.AdjacNode.append(circular_shift[0])
                    Facet.AdjacElem.append([Element.Id[kk],-1])
                else:
                    Facet.AdjacElem[flag][1] = Element.Id[kk]
            print("")

    return Facet