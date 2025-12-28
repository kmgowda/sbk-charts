#!/usr/local/bin/python3
# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##
import os

# The logo insertion function works fine only if the package pillow is installed.
def add_sbk_logo(wb):
    ws = wb.add_worksheet("SBK")
    img_path = os.path.abspath("./images/sbk-logo.png")
    if os.path.exists(img_path):
        print(f"SBK logo image found: {img_path}")
        try:
            ws.insert_image('K7', img_path, {'x_scale': 0.5, 'y_scale': 0.5})
        except Exception as ex:
            print(f"Failed to insert image: {ex}")
    else:
        print(f"SBK logo Image not found: {img_path}")
