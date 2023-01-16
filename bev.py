import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

IMAGE_H, IMAGE_W = 768, 1024

# ROI
tl = [470, 400]
tr = [540, 400]
br = [300, 540]
bl = [666, 540]
roiParams = np.float32([tl, tr, br, bl])

# image params
imgTl = [0, 0]
imgTr = [IMAGE_W, 0]
imgBr = [0, IMAGE_H]
imgBl = [IMAGE_W, IMAGE_H]
imgParams = np.float32([imgTl, imgTr, imgBr, imgBl])


def convertToBev(coordinates):

    # entryImg = copy.deepcopy(READ_IMG)
    # for row in coordinates:
    #     for point in row:
    #         cv2.circle(entryImg, (int(point[1]), int(point[0])),
    #                    2, (0, 0, 255, 255), cv2.FILLED)
    # plt.imshow(cv2.cvtColor(entryImg, cv2.COLOR_BGR2RGB))
    # plt.title("EntryPictureBirdsEyeView")
    # plt.show()

    M = cv2.getPerspectiveTransform(
        roiParams, imgParams)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(
        imgParams, roiParams)  # Inverse transformation

    points = np.float32([[point[1], point[0]]
                        for row in coordinates for point in row])
    points = np.expand_dims(points, axis=1)
    matrix = cv2.perspectiveTransform(points, M)
    # warped_img = cv2.warpPerspective(
    #     READ_IMG, M, (IMAGE_W, IMAGE_H))  # Image warping
    # for row in matrix:
    #     for point in row:
    #         cv2.circle(warped_img, (int(point[0]), int(point[1])),
    #                    5, (0, 0, 255, 255), cv2.FILLED)
    # plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
    # plt.title("EndBirdsEyeView")
    # plt.show()

    newCoord = []
    points = [point.tolist() for row in points for point in row]
    for row in coordinates:
        temp = []
        for point in row:
            newPoint=matrix[points.index([point[1]*1.0,point[0]*1.0])][0]
            if newPoint[0]>0 and newPoint[1]>0 and newPoint[1] < IMAGE_H and newPoint[0] < IMAGE_W:
                temp.append([newPoint[0],IMAGE_H-newPoint[1]])
        if temp!=[]:
            newCoord.append(temp)
    return newCoord, Minv
