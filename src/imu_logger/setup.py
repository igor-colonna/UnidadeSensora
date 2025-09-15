from setuptools import setup

package_name = 'imu_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seu_nome',
    maintainer_email='seu@email.com',
    description='Logger de dados da IMU em CSV',
    license='MIT',
    entry_points={
        'console_scripts': [
            'imu_logger = imu_logger.imu_csv_rotator:main',
            'imu_multi_logger = imu_logger.imu_multi_logger:main',
            'opencv_camera_node = imu_logger.opencv_camera_node:main',
        ],
    },
)
