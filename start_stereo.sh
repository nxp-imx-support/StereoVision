#!/bin/bash

# Copyright 2023 NXP
# SPDX-License-Identifier: BSD-3-Clause

mkdir -p disparity # create if not exist 
mkdir -p left
mkdir -p right

DIR="parameters"
if [ -d "$DIR" ]; then
 echo "Directory with the parameters found. "
else
  echo "Please calibrate the cameras with calibration script and put the obtained parameters here. "
  exit 1
fi

python3 stereo.py --left_dir left --right_dir right --disparity_map disparity --parameters parameters --mode $1
