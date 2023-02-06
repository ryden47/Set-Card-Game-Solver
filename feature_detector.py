from __future__ import print_function
import os
from IMG_manipulations import *
from sewar.full_ref import mse


def identify_number(card, scope_by=0.1, iterations=5):
    image = card.image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 220)  # original: edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count, prev_length = 0, 0
    epsilon = 15   # sensitivity of diffrence between contours length
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if prev_length == 0:
            prev_length = cv2.arcLength(c, closed=True)
            count += 1
        else:
            # print(abs(cv2.arcLength(c, closed=True) - prev_length))
            if abs(cv2.arcLength(c, closed=True) - prev_length) <= epsilon:
                count += 1
            else:
                break
    return count


def identify_shape(card):
    card.shape_rgb = extract_shape(card.image, rgb=True)
    _, cnt = rgb2BWcnt(card.shape_rgb)
    gray = cv2.cvtColor(card.shape_rgb, cv2.COLOR_RGB2GRAY)
    shape = np.zeros_like(gray)
    pts = np.array([p for p, _ in np.ndenumerate(gray)])
    for (x, y) in pts:
        pt = tuple([int(round(x)), int(round(y))])
        shape[pt] = cv2.pointPolygonTest(cnt, (pt[1], pt[0]), False)
    # FOCUS ON THE SHAPE -----------------------------------------
    out = np.zeros_like(gray)
    out[shape == 1] = gray[shape == 1]
    out_rgb = np.zeros_like(card.shape_rgb)
    out_rgb[shape == 1, :] = card.shape_rgb[shape == 1, :]
    card.shape_focus = out_rgb
    # ------------------------------------------------------------
    best_shapeName, best_err = None, float('inf')
    res, total = [], 0
    for file in os.listdir("./train/shape/"):
        filepath = "./train/shape/" + file
        train_load = cv2.imread(filepath, 0)
        shape_cp = cv2.resize(shape, train_load.shape[::-1])
        err = mse(train_load, shape_cp)
        res.append([train_load, err])
        total += err
        if err < best_err:
            best_err = err
            best_shapeName = file[:file.index("_")]
    return best_shapeName


def identify_shading(card, epsilon=0.04):
    gray = cv2.cvtColor(card.shape_rgb, cv2.COLOR_RGB2GRAY)
    best_shadingName, best_err = None, float('inf')
    for file in os.listdir("./train/shading/"):
        if card.shape in file:  # look only at images with matching shape
            filepath = "./train/shading/" + file
            shading_name = file[:file.index("-")]
            train_load = cv2.imread(filepath, 0)    # grayscale
            gray_resized = cv2.resize(gray, train_load.shape[::-1])
            err = mse(train_load/255, gray_resized/255)
            if err < best_err:
                best_err = err
                best_shadingName = shading_name
    return best_shadingName


def identify_color(card):
    gray = cv2.cvtColor(card.shape_focus, cv2.COLOR_BGR2GRAY)
    intensity = 170
    mask = np.zeros_like(card.shape_focus)
    mask[gray < intensity] = card.shape_focus[gray < intensity]  # only color texture, with black background
    color = card.shape_focus[gray < intensity]  # color vector
    mean_color = np.asarray([sum(color[:, 0]), sum(color[:, 1]), sum(color[:, 2])]) / color.shape[0]  # mean color
    largest_color_index = (np.where(mean_color == max(mean_color)))[0][0]
    if largest_color_index == 2:
        return "red"
    elif largest_color_index == 1:
        return "green"
    else:
        return "purple"