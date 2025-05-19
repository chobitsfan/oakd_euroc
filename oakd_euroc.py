#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math
import csv
import pathlib

pathlib.Path("oakd_lite/mav0/imu0").mkdir(parents=True, exist_ok=True)
pathlib.Path("oakd_lite/mav0/cam0/data").mkdir(parents=True, exist_ok=True)
pathlib.Path("oakd_lite/mav0/cam1/data").mkdir(parents=True, exist_ok=True)

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
                    # to ros frame, easier to understand in rviz
                    imu_writer.writerow((int(acceleroValues.getTimestampDevice().total_seconds()*1e9), -gyroValues.z, -gyroValues.x, gyroValues.y, -acceleroValues.z, -acceleroValues.x, acceleroValues.y))
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

