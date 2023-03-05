import cv2

def apply_color(img, res=512):
    input_img_color = cv2.resize(img, (res//64, res//64), interpolation=cv2.INTER_CUBIC)  
    input_img_color = cv2.resize(input_img_color, (res, res), interpolation=cv2.INTER_NEAREST)
    return input_img_color