import numpy as np
import math as math
from pqueue import PqueueHeap

"""
Helper functions that both the speedy and slow voronoi classes use
"""

def calcDistance(pt1,pt2):
    distance = np.linalg.norm(pt1-pt2)
    return distance

def calcSigmaStar(p,v1,v2):
    ans = (np.matmul((v1-p),np.transpose(v1-v2))/(np.linalg.norm(v1-v2))**2)
    return ans

def calcWeight(p,v1,v2):
    ans = np.sqrt((np.linalg.norm(p-v1))**2 - ((np.matmul((v1-p),np.transpose(v1-v2))**2)/(np.linalg.norm(v1-v2))**2))
    return ans

def calcCost(v1,v2,D):
    k1 = 1.0
    k2 = 1.0e5
    if D != 0:
        cost = k1*np.linalg.norm(v1-v2)+k2/D
    else:
        cost = k1*np.linalg.norm(v1-v2)
    return cost


def dijkstraSearch(V,E,J):
    nodes = [[] for _ in V]

    for i in range(len(E)):
        src = E[i][0]
        dest = E[i][1]
        weight = J[i]
        nodes[src].append((dest, weight))
        nodes[dest].append((src, weight))

    dist = [math.inf for _ in V]
    prevNode = [None for _ in nodes]

    start = len(nodes) - 2

    queue = PqueueHeap()
    for i in range(len(nodes)):
        queue.insert(i, math.inf)

    dist[start] = 0
    queue.decreaseKey(start, 0)

    cur = queue.deleteMin()
    while cur != None:
        curDist = dist[cur]
        edges = nodes[cur]
        for (dest, weight) in edges:
            newDist = curDist + weight
            if newDist < dist[dest]:
                prevNode[dest] = cur
                dist[dest] = newDist
                queue.decreaseKey(dest, newDist)
        cur = queue.deleteMin()


    if dist[-1] == math.inf:
        print('Impossible!')
        return None

    path = [len(nodes) - 1]
    prev = prevNode[len(nodes) - 1]
    while prev != None:
        path.insert(0, prev)
        prev = prevNode[prev]

    return(path)
