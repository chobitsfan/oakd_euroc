#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math
import csv
import pathlib
import numpy as np

pathlib.Path("oakd_lite/mav0/imu0").mkdir(parents=True, exist_ok=True)
pathlib.Path("oakd_lite/mav0/cam0/data").mkdir(parents=True, exist_ok=True)
pathlib.Path("oakd_lite/mav0/cam1/data").mkdir(parents=True, exist_ok=True)

fs = cv2.FileStorage("q250_imu_cali.yml", cv2.FILE_STORAGE_READ)
acc_misalign = fs.getNode("acc_misalign").mat()
acc_scale = fs.getNode("acc_scale").mat()
acc_cor = acc_misalign * acc_scale
acc_bias = fs.getNode("acc_bias").mat()
gyro_misalign = fs.getNode("gyro_misalign").mat()
gyro_scale = fs.getNode("gyro_scale").mat()
gyro_cor = acc_misalign * acc_scale
gyro_bias = fs.getNode("gyro_bias").mat()
fs.release()

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xout_imu = pipeline.create(dai.node.XLinkOut)
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)

xout_imu.setStreamName("imu")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 200)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 200)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(5)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)
monoLeft.setCamera("left")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setFps(20)
monoRight.setCamera("right")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setFps(20)

# Link plugins IMU -> XLINK
imu.out.link(xout_imu.input)
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device, open('oakd_lite/mav0/imu0/data.csv', 'w') as imu_file, open('oakd_lite/mav0/cam0/data.csv', 'w') as cam0_file, open('oakd_lite/mav0/cam1/data.csv', 'w') as cam1_file:
    print("start")
    imu_writer = csv.writer(imu_file)
    cam0_writer = csv.writer(cam0_file)
    cam1_writer = csv.writer(cam1_file)
    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    try:
        while True:
            queueName = device.getQueueEvent()
            if queueName == "imu":
                imuData = imuQueue.get()
                imuPackets = imuData.packets
                for imuPacket in imuPackets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope
                    acc_cali = acc_cor @ (np.array([acceleroValues.x, acceleroValues.y, acceleroValues.z]) - acc_bias)
                    gyro_cali = gyro_cor @ (np.array([gyroValues.x, gyroValues.y, gyroValues.z]) - gyro_bias)
                    # align with cam axis
                    imu_writer.writerow((int(acceleroValues.getTimestampDevice().total_seconds()*1e9), gyro_cali[0, 0], -gyro_cali[1, 0], -gyro_cali[2, 0], acc_cali[0, 0], -acc_cali[1, 0], -acc_cali[2, 0]))
            elif queueName == "left":
                inLeft = qLeft.get()
                ts = int(inLeft.getTimestampDevice().total_seconds()*1e9)
                cv2.imwrite(f"oakd_lite/mav0/cam0/data/{ts}.png", inLeft.getFrame())
                cam0_writer.writerow((ts, f"{ts}.png"))
            elif queueName == "right":
                inRight = qRight.get()
                ts = int(inRight.getTimestampDevice().total_seconds()*1e9)
                cv2.imwrite(f"oakd_lite/mav0/cam1/data/{ts}.png", inRight.getFrame())
                cam1_writer.writerow((ts, f"{ts}.png"))
    except KeyboardInterrupt:
        print("ctrl_c")
print("bye")

