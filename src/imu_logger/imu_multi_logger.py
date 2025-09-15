#!/usr/bin/env python3
import os
import csv
import time
import subprocess
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Time as RosTime
from sensor_msgs.msg import Imu, Image

# Configurações
MAX_BYTES = 1_073_741_824  # 1 GiB por arquivo (CSV IMU e cada MKV)
BASE_DIR = os.path.expanduser('~/imu_logs')
IMU_TOPIC = '/imu/data'
CAMERA_TOPICS_DEFAULT = ['/camera/image_raw']  # pode ser sobrescrito por parâmetro
QOS_DEPTH = 10

# Vídeo: 1280x720@30; H.264 com bitrate moderado para gravação longa
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
VIDEO_BITRATE = 4000  # kbps; ajuste conforme armazenamento/qualidade


def ros_time_to_ns(t: RosTime) -> int:
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


class CsvRotator:
    def __init__(self, directory: str, prefix: str, header: List[str]):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.prefix = prefix
        self.header = header
        self.file_index = 0
        self.csv_fh: Optional[object] = None
        self.writer: Optional[csv.writer] = None
        self._open_new_file()

    def _open_new_file(self) -> None:
        if self.csv_fh:
            try:
                self.csv_fh.close()
            except Exception:
                pass
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f'{self.prefix}_{ts}_{self.file_index:03d}.csv'
        path = os.path.join(self.directory, fname)
        self.csv_fh = open(path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.csv_fh)
        self.writer.writerow(self.header)
        self.file_index += 1

    def write_row(self, row: List[object]) -> None:
        assert self.writer is not None
        self.writer.writerow(row)
        if self.csv_fh and self.csv_fh.tell() >= MAX_BYTES:
            self._open_new_file()

    def close(self) -> None:
        if self.csv_fh:
            try:
                self.csv_fh.close()
            except Exception:
                pass


class GStreamerVideoWriter:
    def __init__(self, directory: str, camera_name: str,
                 width: int, height: int, fps: int, bitrate_kbps: int) -> None:
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.file_index = 0
        self.proc: Optional[subprocess.Popen] = None
        self.bytes_written = 0
        self._open_new_file()

    def _build_pipeline(self, filepath: str) -> List[str]:
        # Usa appsrc para empurrar frames raw (BGR) e encode h264 via x264enc (CPU) para ampla compatibilidade
        # Para hardware encoder (Jetson/VAAPI), ajuste aqui.
        caps = f'video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1'
        pipeline = [
            'gst-launch-1.0', '-q',
            'appsrc', f'name=appsrc0', 'is-live=true', 'format=time', 'do-timestamp=true', 'block=true',
            caps,
            '!', 'videoconvert',
            '!', 'x264enc', f'bitrate={self.bitrate_kbps}', 'speed-preset=ultrafast', 'tune=zerolatency', 'key-int-max=60',
            '!', 'h264parse',
            '!', 'matroskamux', 'writing-app=imu_logger',
            '!', 'filesink', f'location={filepath}', 'sync=false', 'async=false'
        ]
        return pipeline

    def _open_new_file(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f'{self.camera_name}_{ts}_{self.file_index:03d}.mkv'
        path = os.path.join(self.directory, fname)
        self.file_index += 1
        pipeline = self._build_pipeline(path)
        # Inicia processo GStreamer em background; alimentaremos via stdin não, pois gst-launch não aceita.
        # Alternativa: usar gst-python. Para simplicidade/portabilidade, reiniciaremos pipeline por arquivo e não faremos push por frame aqui.
        # Em vez disso, usaremos OpenCV VideoWriter que também usa gstreamer backend quando disponível.
        self.current_path = path
        self.bytes_written = 0
        # Marcação: não iniciado aqui por mudança de estratégia para OpenCV; mantemos classe para rotação por tamanho

    def add_bytes(self, num_bytes: int) -> None:
        self.bytes_written += num_bytes
        if self.bytes_written >= MAX_BYTES:
            self._open_new_file()


try:
    import cv2
except Exception:
    cv2 = None


class OpenCvVideoWriter:
    def __init__(self, directory: str, camera_name: str,
                 width: int, height: int, fps: int, bitrate_kbps: int) -> None:
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.file_index = 0
        self.writer: Optional['cv2.VideoWriter'] = None
        self.current_path: Optional[str] = None
        self.bytes_written = 0
        self._open_new_file()

    def _open_new_file(self) -> None:
        if cv2 is None:
            raise RuntimeError('OpenCV (cv2) não disponível')
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f'{self.camera_name}_{ts}_{self.file_index:03d}.mp4'
        path = os.path.join(self.directory, fname)
        self.current_path = path
        self.file_index += 1
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
        self.bytes_written = 0

    def write_frame(self, frame_bgr) -> None:
        if self.writer is None:
            return
        self.writer.write(frame_bgr)
        # Estimativa de bytes: bitrate_kbps -> bytes/s. Para 30 fps, bytes/frame ~ (bitrate*1000/8)/fps
        est_bytes = int((self.bitrate_kbps * 1000 / 8) / max(1, self.fps))
        self.bytes_written += est_bytes
        if self.bytes_written >= MAX_BYTES:
            self._open_new_file()

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass


class ImuMultiLogger(Node):
    def __init__(self):
        super().__init__('imu_multi_logger')
        os.makedirs(BASE_DIR, exist_ok=True)

        # Parâmetros
        self.declare_parameter('imu_topic', IMU_TOPIC)
        self.declare_parameter('camera_topics', CAMERA_TOPICS_DEFAULT)
        self.declare_parameter('video_width', VIDEO_WIDTH)
        self.declare_parameter('video_height', VIDEO_HEIGHT)
        self.declare_parameter('video_fps', VIDEO_FPS)
        self.declare_parameter('video_bitrate_kbps', VIDEO_BITRATE)

        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.camera_topics = [s for s in self.get_parameter('camera_topics').get_parameter_value().string_array_value]
        self.video_width = int(self.get_parameter('video_width').value)
        self.video_height = int(self.get_parameter('video_height').value)
        self.video_fps = int(self.get_parameter('video_fps').value)
        self.video_bitrate_kbps = int(self.get_parameter('video_bitrate_kbps').value)

        qos = QoSProfile(depth=QOS_DEPTH)

        # IMU CSV rotativo
        self.imu_csv = CsvRotator(
            directory=os.path.join(BASE_DIR, 'imu'),
            prefix='imu',
            header=[
                'sec','nanosec',
                'orient_x','orient_y','orient_z','orient_w',
                'cov_orient_00','cov_orient_01','cov_orient_02',
                'cov_orient_10','cov_orient_11','cov_orient_12',
                'cov_orient_20','cov_orient_21','cov_orient_22',
                'ang_vel_x','ang_vel_y','ang_vel_z',
                'cov_avel_00','cov_avel_01','cov_avel_02',
                'cov_avel_10','cov_avel_11','cov_avel_12',
                'cov_avel_20','cov_avel_21','cov_avel_22',
                'lin_acc_x','lin_acc_y','lin_acc_z',
                'cov_lacc_00','cov_lacc_01','cov_lacc_02',
                'cov_lacc_10','cov_lacc_11','cov_lacc_12',
                'cov_lacc_20','cov_lacc_21','cov_lacc_22'
            ]
        )
        self.create_subscription(Imu, self.imu_topic, self.on_imu, qos)

        # Câmeras: writer de vídeo + CSV timestamps por câmera
        self.camera_writers: Dict[str, OpenCvVideoWriter] = {}
        self.camera_csv: Dict[str, CsvRotator] = {}
        for topic in self.camera_topics:
            cam_name = topic.strip('/').replace('/', '_') or 'camera'
            cam_dir = os.path.join(BASE_DIR, cam_name)
            self.camera_writers[topic] = OpenCvVideoWriter(
                directory=cam_dir,
                camera_name=cam_name,
                width=self.video_width,
                height=self.video_height,
                fps=self.video_fps,
                bitrate_kbps=self.video_bitrate_kbps,
            )
            self.camera_csv[topic] = CsvRotator(
                directory=cam_dir,
                prefix=f'{cam_name}_timestamps',
                header=['sec', 'nanosec', 'frame_index']
            )
            self.create_subscription(Image, topic, self._make_image_cb(topic), qos)

        self.get_logger().info(f'IMU em {self.imu_topic}; Câmeras: {self.camera_topics}')

    def on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp
        row = [
            t.sec, t.nanosec,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            *msg.orientation_covariance,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            *msg.angular_velocity_covariance,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            *msg.linear_acceleration_covariance
        ]
        self.imu_csv.write_row(row)

    def _resize_to_target(self, img_msg: Image):
        if cv2 is None:
            return None
        # Converte sensor_msgs/Image em ndarray BGR sem cv_bridge para evitar overhead
        if img_msg.encoding in ('bgr8', 'rgb8'):
            import numpy as np
            step = img_msg.step
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            frame = arr.reshape((img_msg.height, step // 3, 3))[:, :img_msg.width, :]
            if img_msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # Para encodings diferentes, cai para cv_bridge se disponível
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                frame = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            except Exception:
                return None
        if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
            frame = cv2.resize(frame, (self.video_width, self.video_height), interpolation=cv2.INTER_AREA)
        return frame

    def _make_image_cb(self, topic: str):
        def cb(msg: Image):
            writer = self.camera_writers.get(topic)
            csv_logger = self.camera_csv.get(topic)
            if writer is None or csv_logger is None:
                return
            frame = self._resize_to_target(msg)
            if frame is None:
                return
            writer.write_frame(frame)
            # Index de frame aproximado: usamos contagem por arquivo estimando por FPS e/ou manter contador
            # Aqui mantemos contador por tópico
            idx_attr = f'_frame_idx_{topic}'
            cnt = getattr(self, idx_attr, 0)
            csv_logger.write_row([msg.header.stamp.sec, msg.header.stamp.nanosec, cnt])
            setattr(self, idx_attr, cnt + 1)
        return cb

    def destroy_node(self):
        try:
            self.imu_csv.close()
        except Exception:
            pass
        for w in self.camera_writers.values():
            try:
                w.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuMultiLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 