import os
import sys
import cv2
import tqdm
import json
import torch
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm
from itertools import groupby
import matplotlib.pyplot as plt
from torch.nn import functional as F


MAX_WIDTH = 512            # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
MAX_HEIGHT = 512           # NEED THIS CONSTANT TO AVOID "CUDA out of memory"


class ResizeImage():
    def __init__(self, annotation_pth, images_pth) -> None:
        self.annotations_data = None
        self.annotation_pth = annotation_pth
        self.data = None
        self.images_data = None
        self.images_pth = images_pth
        self.images_to_resize = {}

        self.new_annotations = []
        self.new_images_pth = None
        self.new_width = MAX_WIDTH
        self.new_height = MAX_HEIGHT

        self.load_annotation()

    def load_annotation(self):
        with open(self.annotation_pth, "r") as file:
            self.data = json.load(file)

        self.images_data = self.data["images"]
        self.annotations_data = self.data["annotations"]

    def resize(self):
        self.new_images_pth = self.images_pth + "_resized"
        print("resizing images")
        for image in tqdm.tqdm(self.images_data):
            self.image_name = image["file_name"]
            self.image_ori = np.asarray(Image.open(self.images_pth + "/" + self.image_name).convert("RGB"))
            self.height_ori = self.image_ori.shape[0]
            self.width_ori = self.image_ori.shape[1]

            image["width"] = self.new_width
            image["height"] = self.new_height

            self.resize_image()  # resize current self.image_ori. Also get self.new_width and self.new_height
            self.save_image(self.image_ori, self.image_name)

        print("resizing annotations")
        for annotation in tqdm.tqdm(self.annotations_data):
            self.segmentation_ori = annotation["segmentation"]
            self.area_ori = annotation["area"]
            self.bbox_ori = annotation["bbox"]

            self.height_ori = annotation["segmentation"]['size'][0]
            self.width_ori = annotation["segmentation"]['size'][1]

            self.resize_bboxes() # resize bboxes, get self.bbox_new
            self.resize_masks()

            annotation["bbox"] = self.bbox_new
            annotation["segmentation"] = self.segmentation_new
            annotation["area"] = self.area_new

        self.save_annotations()
        
    def resize_image(self):
        dim = (self.new_width, self.new_height)
        self.image_ori = cv2.resize(self.image_ori, dim, interpolation = cv2.INTER_AREA)

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

        dim = (self.new_width, self.new_height)
        resized_mask = cv2.resize(bin_mask, dim, interpolation = cv2.INTER_NEAREST)

        self.segmentation_new, self.area_new = self.mask2rle(resized_mask)

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

    def mask2rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        area = 0
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 255:
                counts.append(0)
            cnt = len(list(elements))
            counts.append(cnt)
            if value == 255:
                area += cnt
        return rle, area

    def save_image(self, image, image_name):
        image = Image.fromarray(image)
        if not os.path.exists(self.new_images_pth):
            os.makedirs(self.new_images_pth)
        image.save(os.path.join(self.new_images_pth+"/", image_name))

    def save_annotations(self):
        if not os.path.exists(self.new_images_pth):
            os.makedirs(self.new_images_pth)

        with open(self.new_images_pth + "/annotation.json", "w") as outfile:
            json.dump(self.data, outfile)

        print("saved annotation to ", self.new_images_pth + "/annotation.json")


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
