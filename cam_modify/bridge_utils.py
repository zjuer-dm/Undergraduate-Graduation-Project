# 文件路径: ~/cam_modify/bridge_utils.py
import numpy as np
import cv2
from sensor_msgs.msg import Image

def imgmsg_to_cv2(img_msg, desired_encoding="passthrough"):
    dtype = np.uint8
    n_channels = 3
    if "8UC3" in img_msg.encoding or "rgb8" in img_msg.encoding or "bgr8" in img_msg.encoding:
        dtype = np.uint8
        n_channels = 3
    elif "8UC1" in img_msg.encoding or "mono8" in img_msg.encoding:
        dtype = np.uint8
        n_channels = 1
    
    img_buf = np.frombuffer(img_msg.data, dtype=dtype)
    if n_channels > 1:
        img_mat = img_buf.reshape((img_msg.height, img_msg.width, n_channels))
    else:
        img_mat = img_buf.reshape((img_msg.height, img_msg.width))

    if img_msg.encoding == "rgb8" and desired_encoding == "bgr8":
        return cv2.cvtColor(img_mat, cv2.COLOR_RGB2BGR)
    return img_mat

def cv2_to_imgmsg(cv_img, encoding="32FC1"):
    img_msg = Image()
    img_msg.height = cv_img.shape[0]
    img_msg.width = cv_img.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.step = cv_img.shape[1] * cv_img.itemsize * cv_img.shape[2] if len(cv_img.shape) > 2 else cv_img.shape[1] * cv_img.itemsize
    img_msg.data = cv_img.tobytes()
    return img_msg