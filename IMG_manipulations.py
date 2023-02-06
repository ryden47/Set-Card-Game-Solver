import cv2
import math
import numpy as np
from PIL import Image, ImageStat
from GUI_utils import *


def extract_shape(image, shade=False, rgb=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 220)  # original: edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray_img)  # Create mask where white is what we want, black otherwise
    biggest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [biggest_contour], 0, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(gray_img)  # Extract out the object and place into output image
    out[mask == 255] = gray_img[mask == 255]
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out_gray = out[topy:bottomy + 1, topx:bottomx + 1]
    out_rgb = image[topy:bottomy + 1, topx:bottomx + 1]
    if rgb:
        return out_rgb
    if shade:
        return dilate[topy:bottomy + 1, topx:bottomx + 1]
    # return out_rgb
    return edged[topy:bottomy + 1, topx:bottomx + 1]


def rotate_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 220)  # original: edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea:
            maxArea = area
            best = contour

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # crop image inside bounding box
    scale = 1.2  # cropping margin, 1 == no margin      - 1 is original
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    rotated = False
    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    image = cv2.getRectSubPix(
        cropped, (int(croppedW * scale), int(croppedH * scale)), (size[0] / 2, size[1] / 2))
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    h, w = image.shape[:2]
    return image[int(h*0.15):int(-h*0.15), int(w*0.12):int(-w*0.12)]


def dalitate(img, color_id, iter=1):
    lbgr = cv2.cvtColor(img, color_id)
    ret, thresh_H = cv2.threshold(lbgr[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_S = cv2.threshold(lbgr[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    dilation1 = cv2.dilate(thresh_H + thresh_S, kernel, iterations=iter)
    dilation2 = cv2.dilate(thresh_H, kernel, iterations=iter)
    dilation3 = cv2.dilate(thresh_S, kernel, iterations=iter)
    return dilation1, dilation2, dilation3


def crop_contour(image, cnt):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_img)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, [cnt], 0, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(gray_img)  # Extract out the object and place into output image
    out[mask == 255] = gray_img[mask == 255]
    # Now crop
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out_rgb = image[topy:bottomy + 1, topx:bottomx + 1]
    return out_rgb, cnt


def brightness1(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im = im_pil.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def brightness2(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]


def brightness3(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def brightness4(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    stat = ImageStat.Stat(im)
    r,g,b = stat.rms
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def brightness5(img):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    stat = ImageStat.Stat(im)
    gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) for r, g, b in im.getdata())
    return sum(gs)/stat.count[0]


def find_contours(game):
    image = game.origin_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 250) #50,220   # original: edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=2)

    # MY TRIALS !!! ------------------------------------------------------------------------------------
    # image = game.origin_image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_flt = np.float32(gray)
    # dst = cv2.cornerHarris(image_flt, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # image[dst > 150] = [0, 0, 255]
    # showone(image, scale=0.2, title="trial")
    # exit()
    # --------------------------------------------------------------------------------------------------

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return [crop_contour(image, cnt) for cnt in sorted(contours, key=cv2.contourArea, reverse=True)][:game.num_of_cards]


def rgb2BWcnt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 220)  # original: edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return dilate, max(contours, key=cv2.contourArea)