import cv2
import os
import numpy as np
from easygui import buttonbox


def showboth(upper_img, lower_img, title="", save2path=False, show=True, scale=1, ret=False):
    if len(upper_img.shape) == 2 and len(lower_img.shape) == 3:
        upper_img = cv2.cvtColor(upper_img, cv2.COLOR_GRAY2RGB)
    if len(upper_img.shape) == 3 and len(lower_img.shape) == 2:
        lower_img = cv2.cvtColor(lower_img, cv2.COLOR_GRAY2RGB)
    if lower_img.shape != upper_img.shape:  # DEFAULT: match the upper to the lower
        if len(upper_img.shape) == 2:
            upper_img = cv2.resize(upper_img, (lower_img.shape[0], lower_img.shape[1]))
        else:  # 3
            upper_img = cv2.resize(upper_img, (lower_img.shape[0], lower_img.shape[1], 3))
    scaled_upper = cv2.resize(upper_img, (int(upper_img.shape[1] * scale), int(upper_img.shape[0] * scale)))
    scaled_lower = cv2.resize(lower_img, (int(lower_img.shape[1] * scale), int(lower_img.shape[0] * scale)))

    # concatenate image Horizontally
    Hori = np.concatenate((scaled_upper, scaled_lower), axis=1)
    # concatenate image Vertically
    Verti = np.concatenate((scaled_upper, scaled_lower), axis=0)
    # cv2.imshow('HORIZONTAL', Hori)
    if save2path:
        if save2path[-4] != '.':
            save2path += ".jpg"
        cv2.imwrite(save2path, Verti)
    if show:
        cv2.imshow(title, Verti)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if ret:
        return Verti


def showone(img, scale=1, title="", save2path=None, show=True):
    if save2path:
        if save2path[-4] != '.':
            save2path += ".jpg"
        cv2.imwrite(save2path, img)
    if show:
        scaled_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        cv2.imshow(title, scaled_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def ask_about_images(images, buttons=[], text="", title=""):
    for i in range(len(images)):
        cv2.imwrite(f"./ask_about_images/ask{i}.jpg", images[i])
    ans = buttonbox(text, choices=buttons, title=title, image=[f'./ask_about_images/ask{i}.jpg' for i in range(len(images))])
    for i in range(len(images)):
        os.remove(f"./ask_about_images/ask{i}.jpg")
    return ans
