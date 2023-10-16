#!/usr/bin/env python3

# Copyright 2023 NXP
# SPDX-License-Identifier: BSD-3-Clause

import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse
import textwrap


def calibrate(left_dirpath, right_dirpath, save_dir, square_size, width, height):
    chessboard_size = (width, height)
    # Defining lists to save detected points
    obj_points_left = []
    img_points_left = []
    obj_points_right = []
    img_points_right = []

    # Declaring grid and output points
    objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Reading images captured by left camera
    calibration_paths_left = glob.glob("./" + left_dirpath + "/*")
    calibration_paths_left.sort()  # sort the list to be in the right order

    # Set a minim ret to find the best combination of photos, the expected value should be less than 0.2
    min_ret = 100

    [n1, n2] = [0, 15]

    while n2 < 128:
        # Empty the lists every 15 photos
        del img_points_left[:]
        del obj_points_left[:]
        del img_points_right[:]
        del obj_points_right[:]

        print("Computing parameters for 15 photos starting with ", n1)
        for image_path in tqdm(calibration_paths_left[n1:n2]):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

            if ret == True:
                # Find corners if chessboard is detected
                cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                obj_points_left.append(objp)
                img_points_left.append(corners)
            else:
                print("chessboard not found", image_path)

        calibration_paths_right = glob.glob("./" + right_dirpath + "/*")
        calibration_paths_right.sort()

        for image_path in tqdm(calibration_paths_right[n1:n2]):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            if ret == True:
                cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                obj_points_right.append(objp)
                img_points_right.append(corners)
            else:
                print("chessboard not found", image_path)

        n1 = n1 + 15
        n2 = n2 + 15

        # Compute the left camera calibration parameters
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            obj_points_left, img_points_left, gray_image.shape[::-1], None, None
        )
        print("Left calibration rms: ", ret_left)

        # Compute the left camera calibration parameters
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            obj_points_right, img_points_right, gray_image.shape[::-1], None, None
        )
        print("Right calibration rms: ", ret_right)

        K1 = K_left
        D1 = dist_left
        K2 = K_right
        D2 = dist_right
        flag = 0
        flag |= cv2.CALIB_USE_INTRINSIC_GUESS

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points_left,
            img_points_left,
            img_points_right,
            K1,
            D1,
            K2,
            D2,
            gray_image.shape[::-1],
            criteria,
            flag,
        )
        print("Stereo calibration rms: ", ret)
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            gray_image.shape[::-1],
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        if ret < min_ret:
            min_ret = ret
            print("Saved parameters for images ", n1 - 15, n2 - 15)
            np.save("./" + save_dir + "/K1", K1)
            np.save("./" + save_dir + "/D1", D1)
            np.save("./" + save_dir + "/K2", K2)
            np.save("./" + save_dir + "/D2", D2)

            np.save("./" + save_dir + "/R1", R1)
            np.save("./" + save_dir + "/R2", R2)
            np.save("./" + save_dir + "/P1", P1)
            np.save("./" + save_dir + "/P2", P2)

    print("Parameters have been saved for minimum ret: ", min_ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="./calibartion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Stereo calibration!
            --------------------------------
                Use the command specifying the names of the directories and the dimensions of the chess board. Default values can be used if appropriate.
                Take as many photos as you like and let the program decide which of them are the best 15 photos.
                At the end, the parameters that had the lowest ret value will be saved. (it should be less than 0.2)
            """
        ),
        epilog="The default parameters will be set as in the following command: ./calibration.py --left_dir left --right_dir right --square_size 2.5 --save_dir parameters --height 7 --width 10.",
    )

    parser.add_argument(
        "--left_dir",
        type=str,
        required=False,
        help='left images directory path, the default is "left"',
        default="left",
    )
    parser.add_argument(
        "--right_dir",
        type=str,
        required=False,
        help='right images directory path, the default is "right"',
        default="right",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        help="chessboard width size, default is 10",
        default=10,
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        help="chessboard height size, default is 7",
        default=7,
    )
    parser.add_argument(
        "--square_size",
        type=float,
        required=False,
        help="chessboard square size, the default is 2.5",
        default=2.5,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help='the directory where to save the parameters (please create it in the work directory before start), the default is "parameters" ',
        default="parameters",
    )

    args = parser.parse_args()

    calibrate(
        args.left_dir,
        args.right_dir,
        args.save_dir,
        args.square_size,
        args.width,
        args.height,
    )
