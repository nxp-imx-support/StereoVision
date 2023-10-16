#!/usr/bin/env python3

# Copyright 2023 NXP
# SPDX-License-Identifier: BSD-3-Clause

from pypylon import genicam
from pypylon import pylon

from PIL import Image

import time
import os
import threading
import cv2
import argparse
import sys
import numpy as np

# Specify number of cameras used, for stereo we use two cameras
maxCamerasToUse = 2


i = 0
j = 0
start_computation_stereo = 0

converter = pylon.ImageFormatConverter()

# Converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


def depth_map(left_rectified, right_rectified):
    """Depth map calculation"""

    window_size = 5

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=5,
        numDisparities=3 * 16,  # has to be dividable by 16
        blockSize=window_size,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    imgL = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.3

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
    )
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def realtime_stereo():
    global left_frame, right_frame, name_left, name_right
    global left_map1, left_map2, right_map1, right_map2

    k = 0

    while True:
        # Rectify the images
        right_rectified = cv2.remap(
            right_frame, right_map1, right_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
        )
        left_rectified = cv2.remap(
            left_frame, left_map1, left_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
        )

        disparity_image = depth_map(
            left_rectified, right_rectified
        )  # Get the disparity map

        cv2.imshow("Disparity", disparity_image)
        key = cv2.waitKey(1)

        # Save the depth map in directories created before starting the application

        if args.mode == 3:
            cv2.imwrite("./" + args.left_dir + "/left_" + str(k) + ".png", left_frame)
            cv2.imwrite(
                "./" + args.right_dir + "/right_" + str(k) + ".png", right_frame
            )
            k = k + 1
            cv2.imwrite(
                "./" + args.disparity_map + "/depth_" + str(k) + ".png", disparity_image
            )
            print("Saved disparity for: ", name_left, name_right)


def start_stereo():
    print("Load parameters. ")

    # Load parameters obtained at calibration step
    global args
    global D1, D2, K1, K2

    K1 = np.load("./" + args.parameters + "/K1.npy")
    D1 = np.load("./" + args.parameters + "/D1.npy")
    K2 = np.load("./" + args.parameters + "/K2.npy")
    D2 = np.load("./" + args.parameters + "/D2.npy")

    R1 = np.load("./" + args.parameters + "/R1.npy")
    R2 = np.load("./" + args.parameters + "/R2.npy")
    P1 = np.load("./" + args.parameters + "/P1.npy")
    P2 = np.load("./" + args.parameters + "/P2.npy")

    width = 640
    height = 480

    global left_map1, left_map2, right_map1, right_map2

    # Compute a new optimal matrix using parameters for a better distorsion
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, (width, height), cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, (width, height), cv2.CV_32FC1
    )

    print("New camera matrix computed.")

    t1 = threading.Thread(target=realtime_stereo)
    t1.daemon = True
    t1.start()


def hardware_trigger():
    class ImageEventPrinter(pylon.ImageEventHandler):
        def OnImagesSkipped(self, camera, countOfSkippedImages):
            print(
                "OnImagesSkipped event for device ",
                camera.GetDeviceInfo().GetModelName(),
            )
            print(countOfSkippedImages, " images have been skipped.")

        def OnImageGrabbed(self, camera, grabResult):
            # print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())
            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # When the cameras in the array are created the camera context value
                # is set to the index of the camera in the array.
                # The camera context is a user settable value.
                # This value is attached to each grab result and can be used
                # to determine the camera that produced the grab result.
                cameraContextValue = grabResult.GetCameraContext()

                image = converter.Convert(grabResult)
                img = image.GetArray()

                # Print the index and the model name of the camera.
                global i, j, start_computation_stereo
                global left_frame, right_frame, name_left, name_right

                if cameraContextValue == 0:
                    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
                    cv2.imshow("left", img)
                    left_frame, name_left = img, "left_%s.png" % i
                    cv2.waitKey(1)

                    if (
                        start_computation_stereo == 0
                        and i > 0
                        and j > 0
                        and (args.mode == 2 or args.mode == 3)
                    ):
                        start_computation_stereo = 1
                        start_stereo()

                    if args.mode == 1:
                        cv2.imwrite("./" + args.left_dir + "/" + name_left, left_frame)
                        print("Saved: ", name_left)

                    i = i + 1

                elif cameraContextValue == 1:
                    cv2.namedWindow("right", cv2.WINDOW_NORMAL)
                    cv2.imshow("right", img)
                    right_frame, name_right = img, "right_%s.png" % j
                    cv2.waitKey(1)

                    if args.mode == 1:
                        cv2.imwrite(
                            "./" + args.right_dir + "/" + name_right, right_frame
                        )
                        print("Saved: ", name_right)
                    j = j + 1

            else:
                print(
                    "Error: ",
                    grabResult.GetErrorCode(),
                    grabResult.GetErrorDescription(),
                )

    # Get the transport layer factory.
    _TlFactory = pylon.TlFactory.GetInstance()

    # get all available cameras
    lstDevices = _TlFactory.EnumerateDevices()

    if len(lstDevices) == 0:
        print("\n   No Basler camera devices are connected!")
        sys.exit(1)

    def print_cameras(lstDevices):
        txt = ""
        for i, dev in enumerate(lstDevices):
            txt += (
                "\n  " + str(i) + ". " + dev.GetFullName() + " " + dev.GetFriendlyName()
            )
        return txt

    print("Working camera ports are :\n" + print_cameras(lstDevices) + "\n\n")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(lstDevices), maxCamerasToUse))

    for i, cam in enumerate(cameras):
        cam.Attach(_TlFactory.CreateDevice(lstDevices[i]))

        # The image event printer serves as sample image processing.
        # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
        # results must be created and registered.
        cam.RegisterImageEventHandler(
            ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete
        )

        cam.Open()
        cam.Width = 640
        cam.Height = 480
        cam.TriggerSelector.SetValue("FrameStart")
        cam.TriggerMode.SetValue("On")
        cam.TriggerSource.SetValue("Line1")
        cam.TriggerActivation.SetValue("RisingEdge")

        print("Using device ", cam.GetDeviceInfo().GetModelName())

    cameras.StartGrabbing(
        pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera
    )

    # Wait for user input to trigger the camera or exit the program.
    # The grabbing is stopped, the device is closed and destroyed automatically when the camera object goes out of scope.
    while True:
        time.sleep(1)


if __name__ == "__main__":
    print()

    parser = argparse.ArgumentParser(description="Depth map commputation. ")

    parser.add_argument(
        "--left_dir", type=str, required=True, help="left images directory path"
    )
    parser.add_argument(
        "--right_dir", type=str, required=True, help="right images directory path"
    )
    parser.add_argument(
        "--disparity_map",
        type=str,
        required=True,
        help="directory used to save depth_map",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        required=True,
        help="directory containing parameters obtained at calibration step",
    )
    parser.add_argument(
        "--mode",
        type=int,
        required=True,
        help="""The mode the way the application will run:
                                                                            1 - only capture frames with saving for use in calibration
                                                                            2 - calculating the depth map without saving frames and the map
                                                                            3 - calculating the depth map with saving frames and the map""",
    )
    args = parser.parse_args()

    print("Mode : ", args.mode)

    print(
        "Start hardware trigger -> Toggle GPIO0_15 using gpio_toggle_example application. "
    )

    os.system("./gpio_toggle_example /dev/gpiochip0 15 & ")

    try:
        hardware_trigger()
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.", e.GetDescription())
