/import numpy as np
from matplotlib import pyplot as plt
import cv2

WHITE = 0

neighbors = {
        "n_1": [ 0,-1],
        "n_2": [-1,-1],
        "n_3": [-1, 0],
        "n_4": [-1, 1],
        "n_5": [ 0, 1],
        "n_6": [ 1, 1],
        "n_7": [ 1, 0],
        "n_8": [ 1,-1]
    }

def rotClockwise(axis, nei):
    cont_n = 1
    while nei != axis + neighbors["n_"+str(cont_n)]:
        cont_n+=1
    return cont_n, axis + neighbors["n_"+str(cont_n)]

def frontierFinder(b_0, matrix):
    height = np.shape(matrix)[0]
    width  = np.shape(matrix)[1]

    while matrix[b_0[0], b_0[1]]==WHITE:
        if b_0[1]!= width-1:
            b_0+=[0,1]
        else:
            if b_0[0]==height-1:
                return
            b_0+=[1,-width+1]

    c = b_0 + neighbors["n_1"]

    # procura o primeiro vizinho não branco para ser o c
    cont_n = 1
    while matrix[c[0], c[1]] != WHITE:
        c = b_0 + neighbors["n_"+str(cont_n)]
        cont_n+=1

    b = c
    c = b_0 + neighbors["n_"+str(((cont_n-1)+9)%9)]# aqui não lembro como arruma
    frontier = [b_0, b] + frontierExplorator(b, c, matrix, b_0, frontier)

    res = np.zeros((height, width))
    for f in frontier:
        res[f[0], f[1]] = 1
    plt.imshow(res)
    plt.show()
    
    
def frontierExplorator(b, c, matrix, b_0, frontier):
    if b == b_0:
        return frontier
   
    cont_n, nk =  rotClockwise(b, c)

    while matrix[nk[0], nk[1]] != WHITE:
        nk = b + neighbors["n_"+str(cont_n)]
        cont_n+=1
    c = b + neighbors["n_"+str(((cont_n-1)+9)%9)] # aqui não lembro como arruma
    b = nk
    frontier += [b]
    return frontierExplorator(b, c, matrix, b_0, frontier)
    
def main():
    b_0 = (0, 1)
    c_0 = (0, 0)

    b_1 = 0
    c_1 = 0

    b = 0
    c = 0

    neighbors = {
        "n_1": [-1, 0],
        "n_2": [-1,-1],
        "n_3": [ 0,-1],
        "n_4": [ 1,-1],
        "n_5": [ 1, 0],
        "n_6": [ 1, 1],
        "n_7": [ 0, 1],
        "n_8": [-1, 1]
    }

    teste = np.array(
        [
            [0,0,0,0,0,0,0],
            [0,0,1,1,1,1,0],
            [0,1,0,0,1,0,0],
            [0,0,1,0,1,0,0],
            [0,1,0,0,1,0,0],
            [0,1,1,1,1,0,0],
            [0,0,0,0,0,0,0]
        ]
    )

    height = np.shape(teste)[0]
    width = np.shape(teste)[1]

    b_0 = np.array([0, 0]) # era pra ser 0,1
    c_0 = np.array([0, 0])

    b_1 = 0
    c_1 = 0

    b = 0
    c = 0

    frontierFinder(b_0, teste)
         
    #print(b_0)
    c_0 = b_0 - [0,1]
    #print(c_0)
    plt.imshow(teste)
    #plt.show()
    #print(np.shape(teste))

if __name__ == '__main__': main()
10 + 