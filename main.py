from utils import predict_patches, resize_img, draw_points, curve_fitting, rescale_coordinates
from PIL import Image
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
from clustering import cluster
from bev import convertToBev
import os

IMAGE_H, IMAGE_W = 768, 1024
old_dims = [768, 1024]
new_dims = [375, 1242]

def swapXY(list):
    return [[point[1], point[0]] for point in list]


warnings.filterwarnings("ignore", category=FutureWarning)

path_lane = r"evaluation"
for ph in os.listdir(path_lane):
    img_path = path_lane+"\\"+ph
    im = cv2.imread(path_lane + "\\" + ph)
    im, no_pad_im, res_f= resize_img(img_path)
    coordinates = predict_patches(img_path)
    # plt.scatter([point[1] for row in coordinates for point in row], [
    #     point[0] for row in coordinates for point in row])
    # plt.show()

    bevPoints, minv = convertToBev(coordinates)
    # plt.scatter([point[0] for row in bevPoints for point in row], [
    #     point[1] for row in bevPoints for point in row])
    # plt.title('AfterBev')
    # plt.show()

    cl = cluster(bevPoints)
    # print(cl)
    l = list(map(lambda clusterPart: np.float32(
        [[point[0], 768-point[1]] for point in clusterPart]), cl))
    l = list(map(lambda list: np.expand_dims(list, axis=1), l))
    l = list(map(lambda list: cv2.perspectiveTransform(list, minv), l))

    # img = draw_points([[point[1], point[0]]
    #                    for clusterPart in l for row in clusterPart for point in row], im)
    # cv2.imshow("drawn", img)
    # cv2.waitKey(0)

    print([[[int(point[1]), int(point[0])] for row in clusterPart for point in row]
           for clusterPart in l])
    if cl != []:
        img = curve_fitting([[[int(point[0]), int(point[1])] for row in clusterPart for point in row]
                             for clusterPart in l], im)
        cv2.imshow("drawn", img)
        cv2.waitKey(0)

    oldCoord = [[[int(point[0]), int(point[1])] for row in clusterPart for point in row]
             for clusterPart in l]
    pad = [old_dims[0] - no_pad_im[0], old_dims[1] - no_pad_im[1]]
    res_c = rescale_coordinates(old_dims, new_dims,oldCoord, res_f, pad)
    with open(f"output\\{ph[:2]}_lane{ph[2:-4]}.txt", "w") as file:
        l = res_c
        for clusterPart in l:
            temp = []
            for point in clusterPart:
                temp.append(str(point[0]))
                temp.append(str(point[1]))
            file.write(' '.join(temp) +"\n")
    print(f'End:{ph}')

