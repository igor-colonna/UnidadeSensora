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
from message_filters import Subscriber as MfSubscriber, ApproximateTimeSynchronizer

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
VIDEO_CODEC = 'h264'  # opções: h264, h265, mjpg, xvid
VIDEO_CONTAINER = ''  # vazio => derivar do codec
SYNC_SLOP_SEC = 0.02  # tolerância de sincronização (20 ms)


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


try:
    import cv2
except Exception:
    cv2 = None


class OpenCvVideoWriter:
    def __init__(self, directory: str, camera_name: str,
                 width: int, height: int, fps: int, bitrate_kbps: int,
                 codec: str = 'h264', container: str = '') -> None:
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.codec = codec.lower()
        self.container = container.lower()
        self.file_index = 0
        self.writer: Optional['cv2.VideoWriter'] = None
        self.current_path: Optional[str] = None
        self.bytes_written = 0
        self._open_new_file()

    def _pick_fourcc_and_ext(self) -> Tuple[List[str], str]:
        # Retorna lista de possíveis FOURCCs (em ordem de preferência) e extensão
        if self.container:
            ext = self.container
            if not ext.startswith('.'):
                ext = '.' + ext
        else:
            # deduz container por codec
            if self.codec == 'h265':
                ext = '.mkv'
            elif self.codec in ('mjpg', 'xvid'):
                ext = '.avi'
            else:
                ext = '.mp4'
        # possíveis fourccs por codec
        if self.codec == 'h264':
            fourccs = ['avc1', 'H264', 'X264']
        elif self.codec == 'h265':
            fourccs = ['hevc', 'H265', 'x265']
        elif self.codec == 'mjpg':
            fourccs = ['MJPG']
        elif self.codec == 'xvid':
            fourccs = ['XVID']
        else:
            # fallback para MJPG
            fourccs = ['MJPG']
        return fourccs, ext

    def _open_new_file(self) -> None:
        if cv2 is None:
            raise RuntimeError('OpenCV (cv2) não disponível')
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        ts = time.strftime('%Y%m%d_%H%M%S')
        fourccs, ext = self._pick_fourcc_and_ext()
        fname = f'{self.camera_name}_{ts}_{self.file_index:03d}{ext}'
        path = os.path.join(self.directory, fname)
        self.current_path = path
        self.file_index += 1
        # tenta em ordem de preferência
        opened = False
        for fcc in fourccs:
            code = cv2.VideoWriter_fourcc(*fcc)
            wr = cv2.VideoWriter(path, code, self.fps, (self.width, self.height))
            if wr is not None and wr.isOpened():
                self.writer = wr
                opened = True
                break
        if not opened:
            # fallback final para MJPG/AVI
            fallback_ext = '.avi'
            path = os.path.join(self.directory, f'{self.camera_name}_{ts}_{self.file_index:03d}{fallback_ext}')
            code = cv2.VideoWriter_fourcc(*'MJPG')
            wr = cv2.VideoWriter(path, code, self.fps, (self.width, self.height))
            if not (wr is not None and wr.isOpened()):
                raise RuntimeError('Falha ao abrir VideoWriter com qualquer codec (h264/h265/mjpg/xvid)')
            self.writer = wr
            self.current_path = path
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
        self.declare_parameter('video_codec', VIDEO_CODEC)
        self.declare_parameter('video_container', VIDEO_CONTAINER)
        self.declare_parameter('sync_slop', SYNC_SLOP_SEC)

        self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.camera_topics = [s for s in self.get_parameter('camera_topics').get_parameter_value().string_array_value]
        self.video_width = int(self.get_parameter('video_width').value)
        self.video_height = int(self.get_parameter('video_height').value)
        self.video_fps = int(self.get_parameter('video_fps').value)
        self.video_bitrate_kbps = int(self.get_parameter('video_bitrate_kbps').value)
        self.video_codec = str(self.get_parameter('video_codec').value or 'h264')
        self.video_container = str(self.get_parameter('video_container').value or '')
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        qos = QoSProfile(depth=QOS_DEPTH)

        # CSV unificado para sincronização IMU + Câmeras
        self.merged_csv = CsvRotator(
            directory=BASE_DIR,
            prefix='merged',
            header=[
                'type',            # 'imu_cam' (linha sincronizada)
                'topic',           # tópico da câmera
                'sec','nanosec',   # timestamp ROS (da câmera)
                'video_file',      # caminho do arquivo de vídeo
                'frame_index',     # índice do frame
                # campos IMU abaixo
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

        # Câmeras: writer de vídeo + CSV timestamps por câmera
        self.camera_writers: Dict[str, OpenCvVideoWriter] = {}
        self.camera_csv: Dict[str, CsvRotator] = {}
        self.syncs: List[ApproximateTimeSynchronizer] = []
        # Um subscriber de IMU compartilhado
        self.imu_sub_mf = MfSubscriber(self, Imu, self.imu_topic, qos_profile=qos)

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
                codec=self.video_codec,
                container=self.video_container,
            )
            self.camera_csv[topic] = CsvRotator(
                directory=cam_dir,
                prefix=f'{cam_name}_timestamps',
                header=['sec', 'nanosec', 'frame_index']
            )
            cam_sub = MfSubscriber(self, Image, topic, qos_profile=qos)
            ats = ApproximateTimeSynchronizer([self.imu_sub_mf, cam_sub], queue_size=QOS_DEPTH, slop=self.sync_slop)
            ats.registerCallback(self._make_sync_cb(topic))
            self.syncs.append(ats)

        self.get_logger().info(f'IMU em {self.imu_topic}; Câmeras: {self.camera_topics}; codec={self.video_codec}; slop={self.sync_slop}s')

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

    def _make_sync_cb(self, topic: str):
        def cb(imu_msg: Imu, img_msg: Image):
            writer = self.camera_writers.get(topic)
            csv_logger = self.camera_csv.get(topic)
            if writer is None or csv_logger is None:
                return
            # vídeo
            frame = self._resize_to_target(img_msg)
            if frame is None:
                return
            writer.write_frame(frame)
            # contador de frame por tópico
            idx_attr = f'_frame_idx_{topic}'
            cnt = getattr(self, idx_attr, 0)
            csv_logger.write_row([img_msg.header.stamp.sec, img_msg.header.stamp.nanosec, cnt])
            setattr(self, idx_attr, cnt + 1)
            # linha unificada (IMU+Cam) usando o timestamp do frame
            merged_row = [
                'imu_cam',
                topic,
                img_msg.header.stamp.sec, img_msg.header.stamp.nanosec,
                writer.current_path or '', cnt,
                imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w,
                *imu_msg.orientation_covariance,
                imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z,
                *imu_msg.angular_velocity_covariance,
                imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z,
                *imu_msg.linear_acceleration_covariance
            ]
            self.merged_csv.write_row(merged_row)
        return cb

    def destroy_node(self):
        try:
            self.merged_csv.close()
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