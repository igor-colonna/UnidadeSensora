#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import csv, os, time

MAX_BYTES = 1_073_741_824  # 1 GiB
LOG_DIR   = os.path.expanduser('~/imu_logs')
TOPIC     = '/imu/data'
DEPTH     = 10             # fila QoS

class ImuCsvRotator(Node):
    def __init__(self):
        super().__init__('imu_csv_rotator')
        qos_profile = rclpy.qos.QoSProfile(depth=DEPTH)
        self.sub = self.create_subscription(Imu, TOPIC, self.callback, qos_profile)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.file_index = 0
        self._open_new_file()
        self.get_logger().info(f'Logging em CSV rotativo: {LOG_DIR}')

    def _open_new_file(self):
        # Fecha arquivo anterior (se existir)
        try:
            self.csv_fh.close()
        except AttributeError:
            pass

        # Nomeia com timestamp + índice
        ts = time.strftime('%Y%m%d_%H%M%S')
        fname = f'imu_{ts}_{self.file_index:03d}.csv'
        path  = os.path.join(LOG_DIR, fname)
        # buffering=1 => line buffered
        self.csv_fh = open(path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.csv_fh)
        # Escreve cabeçalho
        self.writer.writerow([
            'sec','nanosec',
            # orientação
            'orient_x','orient_y','orient_z','orient_w',
            # covariança da orientação (3x3)
            'cov_orient_00','cov_orient_01','cov_orient_02',
            'cov_orient_10','cov_orient_11','cov_orient_12',
            'cov_orient_20','cov_orient_21','cov_orient_22',
            # velocidade angular
            'ang_vel_x','ang_vel_y','ang_vel_z',
            # covariância da velocidade angular
            'cov_avel_00','cov_avel_01','cov_avel_02',
            'cov_avel_10','cov_avel_11','cov_avel_12',
            'cov_avel_20','cov_avel_21','cov_avel_22',
            # aceleração linear
            'lin_acc_x','lin_acc_y','lin_acc_z',
            # covariância da aceleração linear
            'cov_lacc_00','cov_lacc_01','cov_lacc_02',
            'cov_lacc_10','cov_lacc_11','cov_lacc_12',
            'cov_lacc_20','cov_lacc_21','cov_lacc_22'
        ])

        self.get_logger().info(f'-> Novo arquivo: {fname}')
        self.file_index += 1

    def callback(self, msg: Imu):
        t = msg.header.stamp
        row = [
            t.sec, t.nanosec,
            # quaternion
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w,
            # orientation_covariance (9 valores)
            *msg.orientation_covariance,
            # velocidade angular
            msg.angular_velocity.x, msg.angular_velocity.y,
            msg.angular_velocity.z,
            # angular_velocity_covariance
            *msg.angular_velocity_covariance,
            # aceleração linear
            msg.linear_acceleration.x, msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            # linear_acceleration_covariance
            *msg.linear_acceleration_covariance
        ]
        self.writer.writerow(row)

        # Rotaciona se exceder MAX_BYTES
        # .tell() dá posição atual em bytes no arquivo
        if self.csv_fh.tell() >= MAX_BYTES:
            self._open_new_file()

    def destroy_node(self):
        super().destroy_node()
        self.csv_fh.close()

def main(args=None):
    rclpy.init(args=args)
    node = ImuCsvRotator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
