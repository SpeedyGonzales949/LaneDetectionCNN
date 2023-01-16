import math
import os

import cv2
import logging

import numpy as np

from sklearn.cluster import KMeans

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms


logger = logging.getLogger("x")
NEW_SIZE = (1024, 768)
BLANK = np.zeros([16, 64, 3], dtype=np.uint8)
PATH = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\5th image patches"
C = 0


def resize_img(img, max_size=NEW_SIZE, preserve_ratio=True,
               switch_channels=False, padding=True,
               verbose=False, is_path=True):
    """
    Input:
    - img - image (if 'is_path' is true, this is actually the path to the image, else it is a webcam stream photo)
    - max_size - length of the longest side
    - preserve_ratio - whether to preserve ratio between longer and shorter side
      (False might lead to stretched / squished data, padding is recommended)
    - switch_channels - switch BGR to RGB
    - padding - whether to resize with padding (recommended)
    - is_path - whether 'img' is actually the path to img
    Output:
    - new_img - resized image
    - no_pad_shape - shape without padding (for coordinate rescaling)
    """

    # read image, extract shape
    if is_path:
        img = cv2.imread(img)

    h, w = img.shape[:2]

    # calculate new dimensions
    if not preserve_ratio:
        # dimensions without aspect ratio (not recommended)
        new_dims = (max_size[0], max_size[1])
    else:
        # dimensions with aspect ratio preserved (recommended)

        resize_factor = max_size[1] / h
        p_w, p_h = int(w * resize_factor), int(h * resize_factor)
        if p_w > max_size[0] or p_h > max_size[1]:
            resize_factor = max_size[0] / w

        if verbose:
            logger.info(f'resize factor:{resize_factor}')

        new_dims = (int(w * resize_factor), int(h * resize_factor))
        # print(new_dims)

        if verbose:
            logger.info(f'new dims: {new_dims}')

    if verbose:
        logger.info(f'old dims: {w}x{h}\nnew dims: {new_dims[0]}x{new_dims[1]}')

    # resize image
    new_img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

    # save old shape (for rescaling later)
    no_pad_shape = new_img.shape

    # apply padding
    if padding:
        new_w, new_h = no_pad_shape[:2]
        pad_h = math.ceil((max_size[0] - new_h) / 2)
        pad_w = math.ceil((max_size[1] - new_w) / 2)

        new_img = cv2.copyMakeBorder(new_img, pad_w, pad_w,
                                     pad_h, pad_h, cv2.BORDER_CONSTANT)

        new_img = new_img[0:max_size[1], 0:max_size[0]]

    # BGR to RGB, channels first
    if switch_channels:
        new_img = new_img[:, :, ::-1].transpose(2, 0, 1).copy()

    return new_img, no_pad_shape, resize_factor


# img1 = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\5th image.jpg"
# resized_img1 = resize_img(img1)[0]


# patches of 16 h by 64 w from image of shape (768, 1024, 3)
# produce 48 patches h and 16 patches w
def save_patches(img, is_path=False, directory=PATH, c=C):
    os.chdir(directory)

    if is_path:
        img = cv2.imread(img)

    for i in range(48):
        for j in range(16):
            patch = img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64]
            if not np.array_equal(patch, BLANK):
                c += 1
                cv2.imwrite(str(c) + ".jpg", patch)


class LaneClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 6, 3)
        self.conv3 = nn.Conv2d(6, 4, 3)
        self.fc1 = nn.Linear(4*3*27, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def predict(patch):
    transform = transforms.Compose(
        [transforms.ToTensor()
         ])
    input = transform(patch)
    input = input.unsqueeze(0)
    with torch.no_grad():
        laneClassifier = LaneClassifier()
        laneClassifier.load_state_dict(
            torch.load("model_classification_tutorial1.pt"))
        laneClassifier.eval()
        output = laneClassifier(input)

        if torch.diff(output).abs() > 0.5:
            return torch.argmax(output).item()
        else:
            return 0


def predict_patches(img):
    coordinates = []
    lane_ct = 0
    resized_img = resize_img(img)[0]
    for i in range(48):
        temp = []
        for j in range(16):
            patch = resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64]
            if predict(patch) == 1:
                lane_ct += 1
                # cv2.imshow("lane", patch)
                # cv2.waitKey(0)
                bw_patch = convert_bw(patch)
                middle_x_patch, middle_y_patch = get_middle_bw(bw_patch)
                resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64] = cv2.circle(
                    patch, (middle_y_patch, middle_x_patch), radius=1, color=(0, 0, 255), thickness=-1)
                # resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64] = cv2.copyMakeBorder(patch,1,1,1,1,cv2.BORDER_CONSTANT,value=[0, 255, 0])
                temp.append(
                    [i*16 + middle_x_patch, j*64 + middle_y_patch])
        if temp != []:
            coordinates.append(temp)
    # cv2.imshow("lane_pts", resized_img)
    # cv2.waitKey(0)
    return coordinates


def cut_border(img, l=0, r=0, t=0, b=0):
    h, w, _ = img.shape
    h_pad, w_pad = np.zeros((h, 3), dtype=np.uint8), np.zeros(
        [3, w], dtype=np.uint8)
    if sum(img[:, 0].flatten()) <= 64:
        l += 1
        return cut_border(img[:, 1:], l, r, t, b)
    elif sum(img[:, -1].flatten()) <= 64:
        r += 1
        return cut_border(img[:, :-1])
    elif np.array_equal(w_pad, img[0, :]):
        return cut_border(img[1:, :])
    elif np.array_equal(w_pad, img[-1, :]):
        return cut_border(img[:-1, :])
    else:
        return img, l, r, t, b


def convert_bw(img):
    orig_h, owig_w, _ = img.shape
    img, l, r, t, b = cut_border(img)
    no_border_h, no_border_w, _ = img.shape
    copy_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1] * img.shape[0], 3))
    kmeans = KMeans(n_clusters=2)
    s = kmeans.fit(img)
    labels = kmeans.labels_
    labels = list(labels)
    centroid = kmeans.cluster_centers_
    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)

    centroid_sum = [sum(c) for c in centroid]
    centroid_sum, centroid = zip(
        *sorted(zip(centroid_sum, centroid), reverse=True))

    new_img = np.zeros([no_border_h, no_border_w, 3], dtype=np.uint8)
    h, w, _ = new_img.shape
    for i in range(h):
        for j in range(w):
            if abs(sum(copy_img[i, j] - centroid[0])) < abs(sum(copy_img[i, j] - centroid[1])):
                new_img[i, j] = [255, 255, 255]
            else:
                new_img[i, j] = [0, 0, 0]
    # maybe remove noise?
    gray_lane = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    (thresh, bw_lane) = cv2.threshold(gray_lane, 127, 255, cv2.THRESH_BINARY)
    return cv2.copyMakeBorder(bw_lane, t, b, l, r, cv2.BORDER_CONSTANT)


def get_middle_bw(img):
    mean_x, mean_y, occur = 0, 0, 0
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] != 0:
                mean_x += i
                mean_y += j
                occur += 1

    mean_x = round(mean_x / occur)
    mean_y = round(mean_y / occur)
    return mean_x, mean_y


def draw_points(coordinates, image):
    for x, y in coordinates:
        image = cv2.circle(image, (int(y), int(x)), radius=4,
                           color=(0, 0, 255), thickness=-1)
    return image


def curve_fitting(coordinates, img):
    lanes = []
    for lane in coordinates:
        x_list, y_list = [], []

        if lane:
            for x, y in lane:
                x_list.append(x)
                y_list.append(y)

            minimum, maximum = min(x_list), max(x_list)
            x_range = list(range(minimum, maximum))

            z = np.polyfit(x_list, y_list, 2)
            trendpoly = np.poly1d(z)

            lane = ([[int(trendpoly(x_range)[i]), x_range[i]]
                    for i in range(len(x_range))])

        lanes.append(lane)
    for lane in lanes:
        img = draw_points(lane, img)
    return img

# old_dims = [768, 1024]
# new_dims = [375, 1242]
#
# img_um_40_res, no_pad_um_40, res_f = resize_img(img_um_40)
#
# pad = [old_dims[0] - no_pad_um_40[0], old_dims[1] - no_pad_um_40[1]]
#
# res_c = rescale_coordinates(old_dims, new_dims, um_40, res_f, pad)
def rescale_coordinates(old_shape, new_shape, coords, resize_factor, pad=None, verbose=False):
    """
    Rescale (bounding box) coordinates from resized to original image.
    Input:
    - old_shape - shape of resized image
    - new_shape - shape of original image
    - coords - coordinate tensor (4 points for each coordinate)
    - pad - padding applied upon resizing (optional)
    Output:
    - coords - rescaled coordinates
    """

    resize_factor = 1 / resize_factor

    if verbose: logging.info(f'scaling from {old_shape} to {new_shape}')

    h_old, w_old = old_shape
    h_new, w_new = new_shape

    pad_left = pad[1]
    pad_bottom = pad[0]

    if pad_left % 2 == 1:
        pad_left += 1

    pad_left //= 2
    pad_bottom //= 2

    resized_coorinates = []
    # Iterate on coordinates
    for lane in coords:
        new_l = []
        for i, coord in enumerate(lane):
            # get coordinate points
            print(coord)
            x, y = coord

            # Calculate new coordinate points
            new_x = (x - pad_left) * resize_factor
            new_y = (y - pad_bottom) * resize_factor

            # Check if coordinates are inside the image
            if new_x < 0: new_x = 0
            if new_y < 0: new_y = 0
            if new_x > w_new: new_x = w_new
            if new_y > h_new: new_y = h_new

            new_l.append([round(new_x), round(new_y)])
        resized_coorinates.append(new_l)
    return resized_coorinates