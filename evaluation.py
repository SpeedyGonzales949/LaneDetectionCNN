import logging
import math
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

logger = logging.getLogger("x")
NEW_SIZE = (1024, 768)
BLANK = np.zeros([16, 64, 3], dtype=np.uint8)
PATH = r""
C = 0


def resize_img(img, max_size=NEW_SIZE, preserve_ratio=True,
               switch_channels=False, padding=True,
               verbose=False, is_path=False):
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


# patches of 16 h by 64 w from image of shape (768, 1024, 3)
# produce 48 patches h and 16 patches w
def save_patches(img, is_path=True, directory=PATH, c=C):
    os.chdir(directory)

    if is_path:
        img = cv2.imread(img)

    img, _ = resize_img(img)

    for i in range(48):
        for j in range(16):
            patch = img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64]
            if not np.array_equal(patch, BLANK):
                c += 1
                cv2.imwrite(str(c) + ".jpg", patch)


def predict(patch): pass


def predict_patches(img):
    lane_ct = 0
    black_patch = np.zeros((16, 64, 3))
    resized_img = resize_img(img)[0]
    for i in range(48):
        for j in range(16):
            patch = resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64]
            if not np.array_equal(black_patch, patch) and predict(patch) == 1:
                lane_ct += 1
                # cv2.imshow("lane", patch)
                # cv2.waitKey(0)
                bw_patch = convert_bw(patch)
                middle_x_patch, middle_y_patch = get_middle_bw(bw_patch)
                resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64] = cv2.circle(patch,
                                                                                   (middle_y_patch, middle_x_patch),
                                                                                   radius=1, color=(0, 0, 255),
                                                                                   thickness=-1)
                resized_img[i * 16:(i + 1) * 16, j * 64:(j + 1) * 64] = cv2.copyMakeBorder(patch, 1, 1, 1, 1,
                                                                                           cv2.BORDER_CONSTANT,
                                                                                           value=[0, 255, 0])

    print(lane_ct)


def cut_border(img, l=0, r=0, t=0, b=0):
    h, w, _ = img.shape
    h_pad, w_pad = np.zeros((h, 3), dtype=np.uint8), np.zeros([3, w], dtype=np.uint8)
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
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(img)
    labels = kmeans.labels_
    labels = list(labels)
    centroid = kmeans.cluster_centers_
    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)

    centroid_sum = [sum(c) for c in centroid]
    centroid_sum, centroid = zip(*sorted(zip(centroid_sum, centroid), reverse=True))

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

    if occur != 0:
        mean_x = round(mean_x / occur)
        mean_y = round(mean_y / occur)
        return mean_x, mean_y
    else:
        return h // 2, w // 2


def draw_points(coordinates, image):
    im = np.copy(image)
    for lane in coordinates:
        if lane:
            for y, x in lane:
                im = cv2.circle(im, (y, x), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imshow("drawn", im)
    cv2.waitKey(0)


def write_gt(lane1, lane2, filename):
    # check if lane 1 is left and lane2 is right
    if lane2:
        if lane1[0][1] > lane2[0][1]:
            lane1, lane2 = lane2, lane1

    points_first = []
    for i in range(0, len(lane1), 19):
        points_first.append(lane1[i])

    if points_first[-1] != lane1[-1]:
        points_first.append(lane1[-1])

    with open(filename[:-4] + '.txt', 'a') as f:
        for x, y in points_first:
            f.write(str(x) + " " + str(y) + " ")

    if lane2:
        points_second = []
        for i in range(0, len(lane2), 19):
            points_second.append(lane2[i])

        if points_second[-1] != lane2[-1]:
            points_second.append(lane2[-1])

        with open(filename[:-4] + '.txt', 'a') as f:
            f.write("\n")
            for x, y in points_second:
                f.write(str(x) + " " + str(y) + " ")


def get_gt(file, pred=False):
    lanes = []
    with open(file, "r") as f:
        for line in f:
            lane = []
            line = line.split()
            for i in range(0, len(line) - 1, 2):
                if pred:
                    lane.append([int(line[i + 1]), int(line[i])])
                else:
                    lane.append([int(line[i]), int(line[i + 1])])
            lanes.append(lane)

    # if len(lanes) == 1:
    #     lanes.append([])
    # else:
    #     if lanes[0][0][1] > lanes[1][0][1]:
    #         lanes[0], lanes[1] = lanes[1], lanes[0]

    return lanes


def draw_lane():
    path_segm = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\gt_image_2"
    path_im = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\image_2"

    for segm in os.listdir(path_segm):
        img = cv2.imread(path_segm + "/" + segm, 0)

        # for threshold
        # colours = img.flatten()
        # print(f"set {set(colours)}")

        _, bw_img = cv2.threshold(img, 104, 255, cv2.THRESH_BINARY)
        orig_im = cv2.imread(path_im + "/" + str(segm[0:2] + segm[7:]))

        img_num = int(segm[8:-4])
        lane = []
        h, w = bw_img.shape
        for i in range(h):
            for j in range(w - 1):
                # lanes on both sides
                # if 24 <= img_num <= 40 or 61 <= img_num <= 69 or 93 <= img_num <= 94:
                if bw_img[i][j] != bw_img[i][j + 1]:
                    lane.append([i, j])
                # lane on left side only
                # else:
                #     if bw_img[i][j] < bw_img[i][j + 1]:
                #         lane.append([i, j])

        # clustering for noise
        clustering = DBSCAN(eps=3, min_samples=3).fit(lane)
        lane = [lane[i] for i in range(len(lane)) if clustering.labels_[i] != -1]

        # clustering for left-right lane where present
        clustering = DBSCAN(eps=10, min_samples=3).fit(lane)

        lane_first = [lane[i] for i in range(len(lane)) if clustering.labels_[i] == 0]
        lane_second = [lane[i] for i in range(len(lane)) if clustering.labels_[i] == 1]
        final_lanes = [lane_first, lane_second]

        # GT for dataset => the longest lane has 196 px => 10 points => each lane has len/19 points including first and last
        # write_gt(lane_first, lane_second, segm)

        curve_fitting(final_lanes, orig_im)


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

            lane = ([[int(trendpoly(x_range)[i]), x_range[i]] for i in range(len(x_range))])

        lanes.append(lane)
    draw_points(lanes, img)


def curve_pred(coordinate, minimum, maximum):
    x_list, y_list = [], []

    for x, y in coordinate:
        x_list.append(x)
        y_list.append(y)

    x_range = list(range(minimum, maximum))

    z = np.polyfit(x_list, y_list, 2)
    trendpoly = np.poly1d(z)

    return [[int(trendpoly(x_range)[i]), x_range[i]] for i in range(len(x_range))]


GT_PT = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\gt_txt_2lane"
PRED_PT = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\pred_txt"
PHOTO_PT = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\image_2"


def hausdorff_distance(set1, set2):
    distance = 0
    for point1 in set1:
        distances = [np.linalg.norm(np.subtract(point1, point2)) for point2 in set2]
        if distances:
            distance = max(distance, min(distances))
        else:
            distance = max(distance)
    return distance


def evaluate_line_match(pred_lines, gt_lines):
    tp = 0
    fp = 0
    fn = 0
    yp = 0
    i = 0
    for pred_line in pred_lines:
        xp = 0
        i += 1
        for gt_line in gt_lines:
            if hausdorff_distance(pred_line, gt_line) < 15:
                tp += 1
                xp = 1
                break
            else:
                fp += 1

        if xp == 1:
            fp = 0
        else:
            yp += 1

    if tp > 2:
        yyp = len(gt_lines) - tp - yp

    print(tp)

    if fn < 0:
        fn = 0

    return tp, yp, fn
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    # return f1_score


def evaluate():
    tp = 0
    fp = 0
    fn = 0

    for file in os.listdir(GT_PT):
        gt_file = GT_PT + "\\" + file
        gt = get_gt(gt_file)
        if gt:
            while not gt[-1]:
                gt.pop()

        pred_file = PRED_PT + "\\" + file
        pred = get_gt(pred_file, pred=True)
        if pred:
            while not pred[-1]:
                pred.pop()

        image = cv2.imread(PHOTO_PT + "\\" + file[:2] + file[7:-4] + ".png")
        curve_fitting(gt, image)
        curve_fitting(pred, image)

        ttp, tfp, tfn = evaluate_line_match(pred, gt)
        tp += ttp
        fp += tfp
        fn += tfn

    k = sum([tp, fp, fn])
    print(tp / k, fp / k, fn / k)
    print(sum([tp / k, fp / k, fn / k]))


# draw_lane()
evaluate()


# im_00 = cv2.imread(r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\image_2\um_000000.png")
# um_00 = [[[415, 504], [420, 487], [424, 451], [421, 443]], [[531, 410], [533, 440], [551, 455]]]
#
# old_dims = [768, 1024]
# new_dims = [375, 1242]
#
# img_um_00_res, no_pad_um_00, res_f = resize_img(im_00)
#
# pad = [old_dims[0] - no_pad_um_00[0], old_dims[1] - no_pad_um_00[1]]
#
# res_c = rescale_coordinates(old_dims, new_dims, um_00, res_f, pad)
#
#
#
# draw_points(res_c, im_00)


# ----------------------------------------------------------------

# path = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\image_2\um_000000.png"
# img_um_40 = cv2.imread(path)
#
# um_40_gt = get_gt(
#     r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\kitti\training\gt_txt\um_lane_000000.txt")
# um_40 = [[[636, 516], [627, 504], [610, 488], [573, 450], [564, 440], [549, 424], [534, 408], [521, 395], [473, 378],
#           [472, 342]], []]

# ----------------------------------

# um_40_xy = []
# for lane_u in um_40:
#     l = []
#     if lane_u:
#         for i in range(len(lane_u)):
#             l.append([lane_u[i][1], lane_u[i][0]])
#     um_40_xy.append(l)
# um_40 = um_40_xy

# curve_fitting(um_40_gt, img_um_40)

# ---------------------------------------------------------------

# old_dims = [768, 1024]
# new_dims = [375, 1242]
#
# img_um_40_res, no_pad_um_40, res_f = resize_img(img_um_40)
#
# pad = [old_dims[0] - no_pad_um_40[0], old_dims[1] - no_pad_um_40[1]]
#
# res_c = rescale_coordinates(old_dims, new_dims, um_40, res_f, pad)
#
# draw_points(res_c, img_um_40)
#
# um_40_xy = []
# for lane_u in um_40_gt:
#     l = []
#     if lane_u:
#         for i in range(len(lane_u)):
#             l.append([lane_u[i][1], lane_u[i][0]])
#     um_40_xy.append(l)
# draw_points(um_40_xy, img_um_40)

# curve_fitting(res_c, img_um_40)


# curve_fitting(um_40, img_um_40_res)
# ---------------------SHUFFLE BACKGROUND AND CLIP TO LANES LENGTH---------------------------------
# import random
#
# path_lane = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\dataset\lane"
# path_background = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\dataset\background"
#
# path_lane_x = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\dataset augm\lane"
# path_background_x = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\dataset augm\background"
#
# lane = []
# background = []
#
# for ph in os.listdir(path_lane):
#     lane.append(ph)
#
# for ph in os.listdir(path_background):
#     background.append(ph)
#
# random.shuffle(background)
# background = background[:len(lane)]
#
# print(lane[0])
#
# os.chdir(path_lane_x)
# for i in range(len(lane)):
#     im_l = cv2.imread(path_lane + "\\" + lane[i])
#
#     cv2.imwrite(path_lane_x + "\\" + lane[i], im_l)
#
# os.chdir(path_background_x)
# for i in range(len(lane)):
#     im_b = cv2.imread(path_background + "\\" + background[i])
#
#     cv2.imwrite(path_background_x + "\\" + background[i], im_b)


# --------------------------------------------------------------------------------
# def get_discrete_points(orig_coords): pass
#
#
# draw_lane()
#
# img = cv2.imread(r"C:\Users\Florian Moga\Desktop\kitti\training\image_2\um_000040.png")
# im, _ = resize_img(img)
# cv2.imshow("rs", im)
# cv2.waitKey(0)


# ---------------------------------------CREATE AUGM DATASET---------------------------------
# lane = []
# lane_p = r"C:\Users\Florian Moga\Desktop\Facultate\An3\CVDL\dataset and samples\dataset\lane"
# for ph in os.listdir(lane_p):
#     lane.append(ph)
#
# os.chdir(lane_p)
# for i in range(len(lane)):
#     # if lane[i][0] == "n" or lane[i][0] == "v" or lane[i][0] == "h":
#     #     os.remove(lane_p + '\\' + lane[i])
#
#     im_l = cv2.imread(lane_p + "\\" + lane[i])
#
#     im_hflip = np.flip(im_l, axis=0)
#
#     cv2.imwrite(lane_p + "\\h" + lane[i], im_hflip)
#
#     im_vflip = np.flip(im_l, axis=1)
#
#     cv2.imwrite(lane_p + "\\v" + lane[i], im_vflip)
#
#     noise = np.random.normal(loc=0, scale=1, size=im_l.shape)
#     im_noise = np.clip((im_l[..., ::-1] / 255.0 + noise * 0.1), 0, 1)
#     im_noise *= 255
#
#     cv2.imwrite(lane_p + "\\n" + lane[i], im_noise.astype(np.uint8))
