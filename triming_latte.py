import cv2

member_list = ['uejun', 'basshi', 'kiri', 'taku', 'nyokki1', 'nyokki2', 'wadacchi']

def triming_latte(member):
    img_path = member + '_latte.jpg'
    back_img_path = 'back_coffee2.jpg'
    back_img_mask = 'back_coffee2_mask.jpg'
    bound_mask_path = 'back_coffee2_boundmask.jpg'

    img = cv2.imread(img_path)
    back_img = cv2.imread(back_img_path)
    back_mask = cv2.imread(back_img_mask, cv2.IMREAD_GRAYSCALE)
    back_mask_inv = cv2.bitwise_not(back_mask, cv2.IMREAD_GRAYSCALE)
    bound_mask = cv2.imread(bound_mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = back_img.shape[:2]
    img = cv2.resize(img, (width, height))
    back_mask = cv2.resize(back_mask, (width, height))
    back_mask_inv = cv2.resize(back_mask_inv, (width, height))
    bound_mask = cv2.resize(bound_mask, (width, height))

    triming_img = cv2.addWeighted(img, 0.7, back_img, 0.3, 0.0)

    def alpha_blend(img, back_img, alpha, mask):
        height, width = img.shape[:2]
        center = (height / 2, width / 2)
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 255:
                    if bound_mask[y, x] == 255:
                        img[y, x, 0] = img[y, x, 0] * alpha + back_img[y, x, 0] * (1 - alpha)
                        img[y, x, 1] = img[y, x, 1] * alpha + back_img[y, x, 1] * (1 - alpha)
                        img[y, x, 2] = img[y, x, 2] * alpha + back_img[y, x, 2] * (1 - alpha)
                    else:
                        img[y, x, 0] = img[y, x, 0] * 0.6 + back_img[y, x, 0] * 0.4
                        img[y, x, 1] = img[y, x, 1] * 0.6 + back_img[y, x, 1] * 0.4
                        img[y, x, 2] = img[y, x, 2] * 0.6 + back_img[y, x, 2] * 0.4

                else:
                    img[y, x, 0] = back_img[y, x, 0]
                    img[y, x, 1] = back_img[y, x, 1]
                    img[y, x, 2] = back_img[y, x, 2]
        return img

    triming_img = alpha_blend(img, back_img, 0.7, back_mask)
    triming_img = cv2.resize(triming_img, (400, 400))
    save_name = member + '_latte_triming.jpg'
    cv2.imwrite(save_name, triming_img)


for member in member_list:
    triming_latte(member)

