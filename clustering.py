import numpy as np
import matplotlib.pyplot as plt
import math


IMAGE_H, IMAGE_W = 768, 1024

LW = 200


def getNearestPoint(point, row):

    d = float('inf')
    nearestPoint = None
    angle = None
    for p in row:
        dist = math.hypot(point[0] - p[0], point[1] - p[1])
        if dist < d and (p[0] != point[0] or p[1] != point[1]):
            d = dist
            nearestPoint = p
            angle = math.atan2(point[1]-p[1], point[0]-p[0])

    return angle, d, nearestPoint


ANGLE_TRESHOLD = math.pi/8
DISTANCE_TRESHOLD = 300


def cluster(p):
    cl = []
    n = len(p)
    print(p)
    for r in range(0, n):
        row = p[n-r-1]
        for i in range(0, len(row)):
            pi = row[i]
            temp = [pi]
            m = 1
            while True:
                if (n-r-m) < 0:
                    break
                # get the next pont from the row above the current row
                a, d, pj = getNearestPoint(pi, p[n-r-m])
                if a != None and d != None and a <= ANGLE_TRESHOLD and d < DISTANCE_TRESHOLD:
                    temp += [pj]
                    pi = pj
                m += 1

            cl.append(temp)
    cl.sort(key=lambda a: len(a))
    x = np.mean([point[0] for point in cl[-1:][0]])
    plt.scatter([point[0] for row in p for point in row], [
        point[1] for row in p for point in row])
    plt.plot([x, x], [0, IMAGE_H])

    i = 0
    linesX=[]
    while True:
        if i*LW + x%LW > IMAGE_W:
            break
        # plt.plot([i*LW + x%LW, i*LW + x%LW], [0, IMAGE_H],
        #          linestyle='dotted', color='r')
        linesX.append(i*LW + x%LW)
        i+=1
    # plt.title("After Cluster")
    # plt.show()


    points = [point for row in p for point in row]
    clusters = []

    for lineX in linesX:
        temp = []
        for point in points:
            if abs(point[0]-lineX) < LW/2:
                temp.append(point)
        clusters.append(temp)
        
    clusters[list(map(lambda a:int(a),linesX)).index(int(x))] = cl[-1:][0]
    return list(filter(lambda l:len(l)>=3,clusters))
