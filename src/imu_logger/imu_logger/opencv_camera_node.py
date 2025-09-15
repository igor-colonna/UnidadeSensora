#!/usr/bin/env python3
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

DEFAULT_TOPIC = '/camera/image_raw'
DEFAULT_DEVICE = '/dev/video0'
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_PIXEL_FORMAT = 'mjpg'  # mjpg | yuyv | any
QOS_DEPTH = 10


class OpenCVCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('opencv_camera_node')
        if cv2 is None:
            raise RuntimeError('OpenCV (cv2) não disponível')

        # Parâmetros
        self.declare_parameter('device', DEFAULT_DEVICE)
        self.declare_parameter('topic', DEFAULT_TOPIC)
        self.declare_parameter('width', DEFAULT_WIDTH)
        self.declare_parameter('height', DEFAULT_HEIGHT)
        self.declare_parameter('fps', DEFAULT_FPS)
        self.declare_parameter('pixel_format', DEFAULT_PIXEL_FORMAT)
        self.declare_parameter('frame_id', 'camera')

        self.device = self.get_parameter('device').get_parameter_value().string_value or DEFAULT_DEVICE
        self.topic = self.get_parameter('topic').get_parameter_value().string_value or DEFAULT_TOPIC
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.pixel_format = str(self.get_parameter('pixel_format').value or DEFAULT_PIXEL_FORMAT).lower()
        self.frame_id = str(self.get_parameter('frame_id').value or 'camera')

        # Abrir dispositivo
        # Permite tanto caminho string (/dev/video1) quanto índice numérico (0,1,...)
        cap_arg = self.device
        try:
            if self.device.isdigit():
                cap_arg = int(self.device)
        except Exception:
            pass
        self.cap = cv2.VideoCapture(cap_arg, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # fallback sem CAP_V4L2
            self.cap = cv2.VideoCapture(cap_arg)
        if not self.cap.isOpened():
            raise RuntimeError(f'Não foi possível abrir dispositivo: {self.device}')

        # Configura resolução / fps
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # Pixel format (se suportado): MJPG/YUYV
        if self.pixel_format == 'mjpg':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        elif self.pixel_format == 'yuyv':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

        qos = QoSProfile(depth=QOS_DEPTH)
        self.pub = self.create_publisher(Image, self.topic, qos)

        # Timer para captura
        period = 1.0 / max(1, self.fps)
        self.timer = self.create_timer(period, self._capture_once)
        self.get_logger().info(
            f'Publicando {self.width}x{self.height}@{self.fps} de {self.device} em {self.topic} (fmt={self.pixel_format})'
        )

    def _capture_once(self) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn('Falha ao capturar frame')
            return
        # Converte para bgr8 se necessário
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenCVCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 