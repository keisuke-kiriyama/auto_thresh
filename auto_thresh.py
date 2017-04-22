import cv2
import numpy as np

def nothing(x):
    pass


def test(img):
    cv2.imshow("test", img)
    q = cv2.waitKey(1)&0xff
    return q


def controll_brightness(img, brightness):
    edit_array = np.float32(img) * brightness
    _, over_mask = cv2.threshold(edit_array, 255, 255, cv2.THRESH_BINARY)
    _, over_mask_inv = cv2.threshold(edit_array, 255, 255, cv2.THRESH_BINARY_INV)
    int_mask_inv = over_mask_inv.astype(np.int8)
    edit_array = cv2.bitwise_and(edit_array, edit_array, mask=int_mask_inv)
    controlled_array = cv2.bitwise_or(edit_array, over_mask)
    controlled_mat = np.around(controlled_array).astype(np.uint8)
    return controlled_mat

def set_auto_thresh(img_path):
    cv2.namedWindow('gray_image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('thre_image', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('brightness', 'gray_image', 0, 200, nothing)
    cv2.createTrackbar('thresh', 'thre_image', 0, 255, nothing)
    cv2.setTrackbarPos('brightness', 'gray_image', 100)
    cv2.setTrackbarPos('thresh', 'thre_image', 100)

    img_width = 400
    img_height = 500
    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (img_width, img_height))
    while True:
        brightness = cv2.getTrackbarPos('brightness', 'gray_image') / 100
        thresh = cv2.getTrackbarPos('thresh', 'thre_image')
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        img_gray = controll_brightness(img_gray, brightness)
        ret, img_thre = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        cv2.imshow('gray_image', img_gray)
        cv2.imshow('thre_image', img_thre)
        if cv2.waitKey(1) & 0xFF == ord('q') : break
    cv2.destroyAllWindows()




if __name__ == '__main__':
    img_path = 'kiriyama.jpg'
    set_auto_thresh(img_path)