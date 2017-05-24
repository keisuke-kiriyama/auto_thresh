import cv2
import colorsys
import numpy as np
import random


def split_value(max_val, min_val, separation):
    interval = int((max_val - min_val) / (separation - 1))
    value_list = []
    for i in range(separation):
        value_list.append(min_val)
        min_val += interval
    return value_list


def range_threshold(img, min_thresh, max_thresh, max_val):
    _, min_thresh_mask = cv2.threshold(img, min_thresh, max_val, cv2.THRESH_BINARY)
    _, max_thresh_mask = cv2.threshold(img, max_thresh, max_val, cv2.THRESH_BINARY_INV)
    thresh_mask = cv2.bitwise_and(min_thresh_mask, max_thresh_mask)
    return thresh_mask



def create_monochromatic_hsv2rgb(h, s, v, width, height):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    size = height, width, 3
    contours = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    color_img = np.zeros(size, dtype=np.uint8)
    cv2.fillPoly(color_img, pts=[contours], color=(b, g, r))
    return color_img



def multi_value_threshold(img_path, h_value_list, s_value_list, v_value_list, seperation):
    img_original = cv2.imread(img_path)
    img_width = 400
    img_height = 500
    img_original = cv2.resize(img_original, (img_width, img_height))
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    min_gray_value, max_gray_value, min_loc, max_loc = cv2.minMaxLoc(img_gray)
    thresh_value = min_gray_value
    thresh_value_interval = int((max_gray_value - min_gray_value) / seperation)
    multi_thresh_image = create_monochromatic_hsv2rgb(0, 0, 0, img_width, img_height)
    for i in range(seperation):
        color_stage_image = create_monochromatic_hsv2rgb(h_value_list[seperation - i - 1] / 360,
                                                 s_value_list[seperation - i - 1] / 100,
                                                 v_value_list[seperation - i - 1] / 100,
                                                 img_width, img_height)
        if i < (seperation - 1):
            thresh_value += thresh_value_interval
            thresh_mask = range_threshold(img_gray, thresh_value - thresh_value_interval, thresh_value, 255)
            multi_thresh_stage_image = cv2.bitwise_and(color_stage_image, color_stage_image, mask=thresh_mask)
            multi_thresh_image = cv2.bitwise_or(multi_thresh_image, multi_thresh_stage_image)
        elif i == (seperation - 1):
            ret, thresh_mask = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
            multi_thresh_stage_image = cv2.bitwise_and(color_stage_image, color_stage_image, mask=thresh_mask)
            multi_thresh_image = cv2.bitwise_or(multi_thresh_image, multi_thresh_stage_image)
    cv2.imshow("test", multi_thresh_image)
    return multi_thresh_image

def multi_value_threshold_mod(img_path, h_value_list, s_value_list, v_value_list, seperation):
    img_original = cv2.imread(img_path)
    img_width = 400
    img_height = 500
    img_original = cv2.resize(img_original, (img_width, img_height))
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    min_gray_value, max_gray_value, min_loc, max_loc = cv2.minMaxLoc(img_gray)
    thresh_value = min_gray_value
    thresh_value_interval = int((max_gray_value - min_gray_value) / seperation)
    multi_thresh_image = create_monochromatic_hsv2rgb(0, 0, 0, img_width, img_height)
    thresh_list = [40, 100, 160, 160]
    for i, thresh_value in enumerate(thresh_list):
        color_stage_image = create_monochromatic_hsv2rgb(h_value_list[seperation - i - 1] / 360,
                                                 s_value_list[seperation - i - 1] / 100,
                                                 v_value_list[seperation - i - 1] / 100,
                                                 img_width, img_height)
        if i < (seperation - 1) and i != 0:
            thresh_mask = range_threshold(img_gray, thresh_list[i - 1], thresh_list[i], 255)
            multi_thresh_stage_image = cv2.bitwise_and(color_stage_image, color_stage_image, mask=thresh_mask)
            multi_thresh_image = cv2.bitwise_or(multi_thresh_image, multi_thresh_stage_image)
        elif i == 0:
            thresh_mask = range_threshold(img_gray, min_gray_value, thresh_list[i], 255)
            multi_thresh_stage_image = cv2.bitwise_and(color_stage_image, color_stage_image, mask=thresh_mask)
            multi_thresh_image = cv2.bitwise_or(multi_thresh_image, multi_thresh_stage_image)
        elif i == (seperation - 1):
            ret, thresh_mask = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)
            multi_thresh_stage_image = cv2.bitwise_and(color_stage_image, color_stage_image, mask=thresh_mask)
            multi_thresh_image = cv2.bitwise_or(multi_thresh_image, multi_thresh_stage_image)
    cv2.imshow("test", multi_thresh_image)
    return multi_thresh_image


def create_form(latte_img):
    num_of_form =300
    maxRadius = 2
    color = (85, 105, 127)
    for i in range(num_of_form):
        x_random = int(random.random() * latte_img.shape[1])
        y_random = int(random.random() * latte_img.shape[0])
        radius_random = int(random.random() * maxRadius)
        cv2.circle(latte_img, (x_random, y_random), radius_random, color, -1)
    return latte_img



def convert_latte(img, back_img_path):
    img = create_form(img)
    cv2.imwrite("taku_latte.jpg", img)
    cv2.imshow("test", img)
    cv2.waitKey()
    back_img = cv2.imread(back_img_path)
    back_img = cv2.resize(back_img, (400, 500))
    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    img_weighted = cv2.addWeighted(img_gauss, 0.7, back_img, 0.3, 0.0)




if __name__ == '__main__':
    img_path = '../member image/taku.jpg'
    back_img_path ='back_coffee2.jpg'
    seperation = 4
    h_value_list = split_value(29, 29, seperation)
    s_value_list = split_value(100, 17, seperation)
    v_value_list = split_value(50, 100, seperation)
    # red_value_list = split_value(247, 249, seperation)
    # green_value_list = split_value(216, 100, seperation)
    # blue_value_list = split_value(197, 7, seperation)
    # multi_value_threshold(img_path, red_value_list, green_value_list, blue_value_list, seperation)
    multi_thresh_img = multi_value_threshold_mod(img_path, h_value_list, s_value_list, v_value_list, seperation)
    convert_latte(multi_thresh_img, back_img_path)