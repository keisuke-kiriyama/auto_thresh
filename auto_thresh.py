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


def face_detection(img):
    # color = (255, 255, 255)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    facerect = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(facerect) > 0:
        rect = facerect[0]

        return rect
    else:
        return []



def get_mean_dev(img):
    mean, stddev = cv2.meanStdDev(img)
    print('mean brightness', mean[0])
    print('dev brightness', stddev[0])



def set_auto_thresh(img_path):
    cv2.namedWindow('gray_image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('thre_image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('parameter', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('brightness', 'parameter', 0, 300, nothing)
    cv2.createTrakbar('thresh', 'parameter', 0, 255, nothing)
    cv2.setTrackbarPos('brightness', 'parameter', 100)
    cv2.setTrackbarPos('thresh', 'parameter', 100)
    img_width = 400
    img_height = 500
    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (img_width, img_height))
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    face_rect = face_detection(img_original)
    if len(face_rect) > 0:
        face_region = img_gray[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0]:face_rect[0] + face_rect[2]]
        get_mean_dev(face_region)
    pre_brightness = 0
    pre_thresh = 0
    while True:
        brightness = cv2.getTrackbarPos('brightness', 'parameter') / 100
        if brightness != pre_brightness : print('brightness parameter: ', brightness)
        pre_brightness = brightness
        thresh = cv2.getTrackbarPos('thresh', 'parameter')
        if thresh != pre_thresh : print('thresh : ', thresh)
        pre_thresh = thresh
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        img_gray = controll_brightness(img_gray, brightness)
        ret, img_thre = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        cv2.rectangle(img_gray, tuple(face_rect[0:2]), tuple(face_rect[0:2] + face_rect[2:4]), 255, thickness=2)
        cv2.imshow('gray_image', img_gray)
        cv2.imshow('thre_image', img_thre)
        if cv2.waitKey(100) & 0xFF == ord('q') : break
    cv2.destroyAllWindows()




if __name__ == '__main__':
    img_path = '../member image/kiriyama2.jpg'
    set_auto_thresh(img_path)