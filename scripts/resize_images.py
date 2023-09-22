import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from torch.nn import functional as F


MAX_WIDTH = 1280            # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
MAX_HEIGHT = 1024           # NEED THIS CONSTANT TO AVOID "CUDA out of memory"


class ResizeImage():
    def __init__(self, annotation_pth, images_pth) -> None:
        self.annotations_data = None
        self.annotation_pth = annotation_pth
        self.images_data = None
        self.images_pth = images_pth
        self.images_to_resize = {}
        self.new_annotations = []
        self.new_images_pth = None

        self.load_annotation()

    def load_annotation(self):
        with open(self.annotation_pth, "r") as file:
            data = json.load(file)

        self.images_data = data["images"]
        self.annotations_data = data["annotations"]

        for image in self.images_data:
            if image["height"] > MAX_HEIGHT and image["width"] > MAX_WIDTH:
                self.images_to_resize[image["file_name"]] = (image["id"], image["height"], image["width"])

    def resize(self):
        self.new_images_pth = self.images_pth + "_resized"
        for image in self.images_to_resize:
            self.image_ori = np.asarray(Image.open(self.images_pth + "/" + image).convert("RGB"))
            self.height_ori = self.images_to_resize[image][1]
            self.width_ori = self.images_to_resize[image][2]

            self.resize_image()  # resize current self.image_ori. Also get self.new_width and self.new_height
            self.save_image(self.image_ori, image)

            self.image_annotations = [x for x in self.annotations_data if x["image_id"] == self.images_to_resize[image][0]]
            for annotation in self.image_annotations:
                self.segmentation_ori = annotation["segmentation"]
                self.area_ori = annotation["area"]
                self.bbox_ori = annotation["bbox"]

                self.resize_bboxes() # resize bboxes, get self.bbox_new
                self.resize_masks()

                ##### NEED TO SAVE NEW SEGM, AREA, BBOX FOR CURRENT ANNOTATION

            ##### NEED TO SAVE NEW ANNOTATIONS FOR CURRENT IMAGE
            i=0

        ##### NEED TO SAVE NEW ANNOTATIONS FOR CURRENT DATASET
        
    def resize_image(self):
        w_scale = round(MAX_WIDTH * 100 / self.width_ori)
        h_scale = round(MAX_HEIGHT * 100 / self.height_ori)
        scale_percent = max(w_scale, h_scale)
        self.new_width = int(self.width_ori * scale_percent / 100)
        self.new_height = int(self.height_ori * scale_percent / 100)
        dim = (self.new_width, self.new_height)
        self.image_ori = cv2.resize(self.image_ori, dim, interpolation = cv2.INTER_AREA)

    def save_image(self, image, image_name):
        image = Image.fromarray(image)
        if not os.path.exists(self.new_images_pth):
            os.makedirs(self.new_images_pth)
        image.save(os.path.join(self.new_images_pth+"/", image_name))

    def resize_bboxes(self):
        height_box = self.height_ori
        width_box = self.width_ori

        x1 = int(self.bbox_ori[0])
        y1 = int(self.bbox_ori[1])
        x2 = int(self.bbox_ori[0] + self.bbox_ori[2])
        y2 = int(self.bbox_ori[1] + self.bbox_ori[3])

        scale = torch.tensor([self.new_width/width_box, self.new_height/height_box, \
                            self.new_width/width_box, self.new_height/height_box])[None,:]

        box = torch.Tensor([x1, y1, x2, y2])
        bbox_new = box * scale
        bbox_new = [int(coord) for coord in bbox_new.numpy()[0]]
        self.bbox_new = [bbox_new[0], bbox_new[1], bbox_new[2]-bbox_new[0], bbox_new[3]-bbox_new[1]]

    def resize_masks(self):
        rle_mask = self.segmentation_ori["counts"]
        segm_height, segm_width = self.segmentation_ori["size"]
        bin_mask = self.rle2mask(rle_mask, (segm_width, segm_height))

        masks = torch.Tensor(bin_mask)
        # masks = masks[:, : masks.shape[1], : masks.shape[2]].expand(1, -1, -1, -1)
        masks = masks.expand(1, -1, -1, -1)
        print("Before interpolation", masks.shape)
        masks = F.interpolate(
            masks, size=(self.new_height, self.new_width), mode="nearest"
        )[0]
        print("After interpolation", masks.shape)

        i=0

    def rle2mask(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        i, counter = 0, 0
        for j in mask_rle:
            counter+=1
            if not counter % 2: pixel = 255
            else: pixel = 0
            img[i:i+j] = pixel
            i = i+j
        return img.reshape(shape).T


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', help='Path to the user input folder of images', default='images/')
    parser.add_argument('--annotation_path', help='Path to the user annotation of images', default='images/')
    cmdline_args = parser.parse_args()
    images_pth = cmdline_args.images_path
    annotation_pth = cmdline_args.annotation_path

    resized = ResizeImage(annotation_pth, images_pth)
    resized.resize()


if __name__ == "__main__":
    main()
    sys.exit(0)