#!/usr/bin/env python
import datetime
import socket
import numpy as np
import cv2
import os
import time
import struct
import pyrealsense2 as rs
import open3d as o3d


class RealsenseD435(object):

    def __init__(self, width=640, hight=480):
        """初始化函数
        Args:
            width (图像宽度): 默认640
            hight (图像高度): 默认480
            intrinsics (内参数矩阵):
            scale (深度缩放因子):
        """
        self.im_width = width
        self.im_height = hight
        self.intrinsics = None
        self.scale=None  # 0.001
        self.connect()

    def get_intrinsics(self, rgb_profile):
        """获取RGB相机的内参
        """
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera raw intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        # [[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([605.135,0,312.564,0,604.984,237.029,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array(
            [raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        return intrinsics

    def connect(self):
        """连接RS相机
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)
        # Start streaming
        profile = self.pipeline.start(config)
        # Determine intrinsics
        self.rgb_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(self.rgb_profile)
        # Determine depth scale
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()  # 0.001

    def get_data(self, sample_skip=0, isPrintShape=False, get_w=None, get_h=None, offset_x=0, offset_y=0):
        """
        获取对齐后的RGB图像和深度图像，并提供剪裁功能。
        - 如果设置了get_w和get_h，则会从图像中心开始，并根据提供的偏移量裁剪出指定大小的图像区域。
        - 如果剪裁区域超出了原始图像的边界，则会相应地调整剪裁区域以适应图像尺寸。
        Parameters:
            sample_skip (int): 要跳过的帧数，默认为0。这用于避免初始化时的不稳定帧。
            isPrintShape (bool): 是否打印输出图像的形状，默认为False。
            get_w (int or None): 剪裁宽度，默认为None，表示不进行剪裁。
            get_h (int or None): 剪裁高度，默认为None，表示不进行剪裁。
            offset_x (int): 相对于图像中心的水平偏移量，默认为0。
            offset_y (int): 相对于图像中心的垂直偏移量，默认为0。
        Returns:
            tuple: 包含三个元素的元组：
                - color_image (numpy.ndarray): RGB图像数组。
                - depth_image (numpy.ndarray): 深度图像数组，单位为米。
                - gray_depth_image (numpy.ndarray): 深度图像转换成的灰度图像数组，便于可视化。
        """
        
        "丢弃一定数量的帧，避免初始化时的不稳定帧"
        for i in range(sample_skip):
            frame = self.pipeline.wait_for_frames()
            depth = frame.get_depth_frame()
            color = frame.get_color_frame()
        "获取对齐后的帧数据"
        frames = self.pipeline.wait_for_frames()
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        "获取深度图像和 RGB 图像，并返回它们"
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)   # 单位为m
        # depth_image *= self.scale
        # depth_image = cv2.inpaint(depth_image)  # 补全缺失值
        color_image = np.asanyarray(color_frame.get_data())
        "图像剪裁"
        if get_w is not None and get_h is not None: # 计算剪裁区域
            center_x, center_y = color_image.shape[1] // 2, color_image.shape[0] // 2
            start_x = max(center_x + offset_x - get_w // 2, 0)
            start_y = max(center_y + offset_y - get_h // 2, 0)
            end_x = min(start_x + get_w, color_image.shape[1])
            end_y = min(start_y + get_h, color_image.shape[0])
            color_image = color_image[start_y:end_y, start_x:end_x] # 剪裁图像
            depth_image = depth_image[start_y:end_y, start_x:end_x] # 对于深度图，我们假设它有相同的尺寸
        "将深度图转换为灰度图以便显示"
        min_depth = np.min(depth_image)  # 深度图中的最小深度值（单位mm）
        max_depth = np.max(depth_image)  # 深度图中的最大深度值（单位mm）
        gray_depth = np.uint8(255 * ((depth_image - min_depth) / (max_depth - min_depth)))
        gray_depth_image = np.stack((gray_depth,)*3,axis=-1)
        if isPrintShape is True:
            print('color_image.shape: ', color_image.shape)
            print('depth_image.shape: ', depth_image.shape)
            print('gray_depth_image.shape: ', gray_depth_image.shape)
            # color_image.shape:  (480, 640, 3)
            # depth_image.shape:  (480, 640)
            # gray_depth_image.shape:  (480, 640, 3)
        return color_image, depth_image, gray_depth_image
    

    def sample_data_test(self, sample_skip=0):
        save_path = os.path.join('Data','RealSense_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saved_count=0
        cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("live", cv2.WND_PROP_TOPMOST, 1)  # 设置窗口始终位于最前面
        while True:
            # color_image, depth_image, gray_depth_image = self.get_data(sample_skip=sample_skip, isPrintShape=False)
            color_image, depth_image, gray_depth_image = self.get_data(sample_skip=sample_skip, isPrintShape=False,
                                                                       get_w=256, get_h=256, offset_x=0, offset_y=0)
            # print(color_image)
            images = np.hstack((color_image, gray_depth_image))
            cv2.imshow("live",images)
            key=cv2.waitKey(1)
            # s 保存图片
            if key & 0xFF == ord('s'):
                saved_count+=1
                cv2.imwrite(os.path.join((save_path), "{:04d}r.png".format(saved_count)), color_image)  # 保存RGB为png文件
                cv2.imwrite(os.path.join((save_path), "{:04d}d.tiff".format(saved_count)), depth_image)  # 保存深度图为tiff文件
                cv2.imwrite(os.path.join((save_path), "{:04d}d_gray.png".format(saved_count)), gray_depth_image)  # 保存深度灰度图为png文件
                print("{:04d}r.png 已保存！".format(saved_count))
            # q 退出
            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            else:
                pass

def test():
    depth_image = cv2.imread('Data/RealSense_test/0001d.tiff', cv2.IMREAD_UNCHANGED)
    print("深度tiff图像的形状: ", depth_image.shape)
    depth_gray_image = cv2.imread('Data/RealSense_test/0001d_gray.png', cv2.IMREAD_UNCHANGED)
    print("深度png灰度图像的形状: ", depth_gray_image.shape)
    rgb_image = cv2.imread('Data/RealSense_test/0001r.png', cv2.IMREAD_UNCHANGED)
    print("png图像的形状: ", rgb_image.shape)




if __name__ == "__main__":
    camera=RealsenseD435()
    # time.sleep(1)
    save_path = os.path.join('Data','RealSense_test')
    for i in range(5):
        color_image, depth_image, gray_depth_image = camera.get_data(sample_skip=0, isPrintShape=False,
                                                                    get_w=256, get_h=256, offset_x=0, offset_y=0)
        cv2.imwrite(os.path.join((save_path), "_test_{}.png".format(i)), color_image)  # 保存RGB为png文件
        time.sleep(0.5)
    print(color_image)
    
    print("开始捕获图片，按s保存图像，按q退出")
    camera.sample_data_test(sample_skip=0)
    test()
    rgb_profile = camera.rgb_profile
    intrinsics = camera.intrinsics
    scale = camera.scale
    print('rgb_profile: ', rgb_profile)
    print('intrinsics: ', intrinsics)
    print('scale: ', scale)