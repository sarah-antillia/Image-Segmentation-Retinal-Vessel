# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatsetGenerator.py

# 2023/06/07 Toshiyuki Arai @antillia.com


import os
import numpy as np
import shutil
import glob
import cv2
from PIL import Image

import traceback

#from PIL import Image, ImageDraw, ImageFilter

class ImageMaskDatasetGenerator:

  def __init__(self, data_dir="./", output_dir="./master", width=256, height=256, 
               rotation=False, debug=False):
   
    self.data_dir   = data_dir
    self.output_dir = output_dir
    self.W          = width
    self.H          = height
    self.debug      = debug
    self.rotation   = rotation
    self.IMAGES_DIR = "images"
    self.MASKS_DIR  = "masks"
    self.output_img = ".jpg"
    self.TRAIN      = "train"
    self.VALID      = "valid"
    self.TEST       = "test"
    
  def create_master(self, debug=False):

      
      image_files = glob.glob(self.data_dir + "/*.jpg")
      # Get category name from images_dir string array.
      
      num_images = len(image_files)
      num_train  = int(num_images * 0.7)
      num_valid  = int(num_images * 0.2)
      num_test   = int(num_images * 0.1)
      train_image_files = image_files[0: num_train]
      valid_image_files = image_files[num_train: num_train + num_valid]
      test_image_files  = image_files[num_train + num_valid:]
      print("num_train {}".format(num_train))
      print("num_valid {}".format(num_valid))
      print("num_test {}".format(num_test))
      
      self.create_dataset(train_image_files, output_dir, self.TRAIN, debug=debug)
      self.create_dataset(valid_image_files, output_dir, self.VALID, debug=debug)
      self.create_dataset(test_image_files,  output_dir, self.TEST,  debug=debug)

  def create_dataset(self, image_files, output_dir, dataset, debug=False):
    output_dir = os.path.join(output_dir, dataset)
    output_images_dir = os.path.join(output_dir, "images")

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)

    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    output_masks_dir  = os.path.join(output_dir, "masks")

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
        
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)
        
    for image_file in image_files:
      image = cv2.imread(image_file)


      image = cv2.imread(image_file)
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      self.save_image(output_images_dir, name, image_file, mask=False)

      mask_file = name + "_2ndHO.png"
      mask_filepath = os.path.join(self.data_dir, mask_file)
      mask_image = cv2.imread(mask_filepath)
      self.save_image(output_masks_dir, name, mask_filepath, mask=True)


  def save_image(self, output_dir, name, image_file, mask=False):
    image = Image.open(image_file)
    w, h = image.size
    SQ_SIZE = w
    if h > SQ_SIZE:
      SQ_SIZE = h
    
    background = Image.new("RGB", (SQ_SIZE, SQ_SIZE), (0, 0, 0))
    rx = (SQ_SIZE - w) // 2
    ry = (SQ_SIZE - h) // 2
    background.paste(image, (rx, ry) )
    if mask:
      background = background.convert("L")

    resized = background.resize((self.W, self.H))
    resized = self.pil_to_cv(resized)

    self.rotate(resized, name, output_dir)
    self.flip(resized,   name, output_dir)
   

  def pil_to_cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: # JPG
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # PNG
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


  def rotate(self, image, name, output_dir):
    #ANGLES = [0, 90, 180, 270]
    ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
    print("=== Output_dir {}".format(output_dir))
    for angle in ANGLES:
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H))

      rotated_filename         = "rotated-" + str(angle) + "-" +  "_" + name + self.output_img
      rotated_filepath = os.path.join(output_dir, rotated_filename )
      print("---- Saved {}".format(rotated_filepath))

      cv2.imwrite(rotated_filepath, rotated)


  def flip(self, image, name, output_dir):
    DIRECTIONS = [0, -1, 1]
    print("=== Output_dir {}".format(output_dir))
    for direction in DIRECTIONS:
      flipped = cv2.flip(image, direction)

      flipped_filename         = "flipped-" + str(direction) + "-" + "_" + name + self.output_img
      flipped_filepath = os.path.join(output_dir, flipped_filename )
      print("---- Saved {}".format(flipped_filepath))

      cv2.imwrite(flipped_filepath, flipped)

"""
Input
./CHASEDB1
  +-- Image_01L.jpg
  +-- Image_01L_1stHO.png
  +-- Image_01L_2bdHO.png
  +-- Image_01R.jpg
  +-- Image_01R_1stHO.png
  +-- Image_01R_2bdHO.png
  ...
  +-- Image_14L.jpg
  +-- Image_14L_1stHO.png
  +-- Image_14L_2bdHO.png
  +-- Image_14R.jpg
  +-- Image_14R_1stHO.png
  +-- Image_14R_2bdHO.png

├
"""
"""
Output
./Retian-Vessel
  +-- test
    +-- images
    +-- masks

  +-- train
      +-- images
      +-- masks
  +-- valid
      +-- images
      +-- masks


"""

if __name__ == "__main__":
  try:
    root_dir   = "./CHASEDB1"
    output_dir = "./Retinal-Vessel"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    generator = ImageMaskDatasetGenerator(data_dir=root_dir, output_dir=output_dir, 
                                           width=512, height=512, 
                                          rotation=True, debug=True)
    generator.create_master( debug=True)

  except:
    traceback.print_exc()

