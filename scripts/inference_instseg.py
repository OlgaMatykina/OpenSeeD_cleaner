import os
import sys
import logging
import tqdm
import glob
import gc
import cv2
import time
import json
import base64
import zlib
from json import encoder
from itertools import groupby
import matplotlib.pyplot as plt

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
import typing as t
np.random.seed(2)

import torch
from torchvision import transforms
from torch.nn import functional as F

from pycocotools import _mask as coco_mask
from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from openseed.BaseModel import BaseModel
from openseed import build_model
from detectron2.utils.colormap import random_color
from utils.visualizer import Visualizer

gc.collect()
torch.cuda.empty_cache()
logger = logging.getLogger(__name__)
encoder.FLOAT_REPR = lambda o: format(o, '.5f')


REQUEST_LIST = [            # LIST OF REQUESTS
    "firehose", 
    "hose", 
    "wire", 
    "rope",
    "poop", 
    "puddle", 
    "pit", 
    "bump", 
    "curb",
    "rail track",
    "sand",
    "manhole",
    "catch basin", 
    "sidewalk",
    "pedestrian area",
    "bike lane",
    "grass",
    "vegetation",
    "terrain",
    "road",
]
COLORS = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [84, 84, 0],
            [0, 170, 255],
            [218, 218, 218],
            [84, 84, 255],
            [76, 76, 76],
            [170, 255, 255],
            [170, 84, 255],
            [170, 0, 127],
            [170, 255, 0],
            [255, 170, 127],
            [0, 84, 255],
            [118, 171, 47],
            [84, 0, 255],
            [255, 84, 255],
        ]
TIME_NUMBER_START = 2       # START NUMBER OF CALCULATING AVERAGE INFERENCE TIME
MAX_WIDTH = 512            # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
MAX_HEIGHT = 512           # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
THRESHOLD = 0.5             # THRESHOLD FOR VISUALIZATION AND ANNOTATION


class InstanceSegm():
    def __init__(self, opt, pretrained_pth, output_root, images_pth, annotation_pth) -> None:
        self.opt = opt
        self.pretrained_pth = pretrained_pth
        self.output_root = output_root
        self.images_pth = images_pth
        self.annotation_pth = annotation_pth
        self.images_ids = {}

        self.treshold = THRESHOLD
        self.total_time = 0
        self.skipped_first_time = 0

        self.init_model()
        self.read_annotations()

    def init_model(self):
        self.model = BaseModel(self.opt, build_model(self.opt)).from_pretrained(self.pretrained_pth).eval().cuda()

        t = []
        t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

        self.thing_classes = REQUEST_LIST
        # self.thing_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(self.thing_classes))]
        self.thing_colors = COLORS
        thing_dataset_id_to_contiguous_id = {x:x for x in range(len(self.thing_classes))}

        MetadataCatalog.get("demo").set(
            thing_colors=self.thing_colors,
            thing_classes=self.thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        )
        # model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + ["background"], is_eval=False)
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.thing_classes, is_eval=True)
        self.metadata = MetadataCatalog.get("demo")
        self.model.model.metadata = self.metadata
        self.model.model.sem_seg_head.num_classes = len(self.thing_classes)

        self.output_subroot = self.images_pth.split("/")[-1]
        self.output_root += self.output_subroot
        print("OUTPUT DIR: ", self.output_root)

    def read_annotations(self):
        with open(self.annotation_pth, "r") as file:
            data = json.load(file)
            images_data = data["images"]
            for img in images_data:
                self.images_ids[img["file_name"]] = img["id"]

    def start_instance(self):
        self.predicted_annotation = []
        self.images_files = sorted(glob.glob(f"{self.images_pth}/*"))
        for idx, image_pth in enumerate(tqdm.tqdm(self.images_files)):
            image_name = image_pth.split("/")[-1]

            with torch.no_grad():
                self.image_ori = Image.open(image_pth).convert("RGB")
                self.width = self.image_ori.size[0]
                self.height = self.image_ori.size[1]

                image = self.transform(self.image_ori)
                self.image_ori = np.asarray(self.image_ori)

                self.cur_width = self.width
                self.cur_height = self.height
                # If image too large it needs to beeing resized to avoid "CUDA out of memory"
                self.resize_img()

                self.images = torch.from_numpy(self.image_new.copy()).permute(2,0,1).cuda()
                batch_inputs = [{"image": self.images, "height": self.cur_height, "width": self.cur_width}]
                
                cur_time = time.time()
                # Get output from model
                outputs = self.model.forward(batch_inputs)
                self.total_time += time.time() - cur_time
                if idx >= TIME_NUMBER_START:
                    self.skipped_first_time += time.time() - cur_time
                
                inst_seg = outputs[-1]["instances"]

                inst_seg.pred_masks = inst_seg.pred_masks.cpu()
                inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()

                visual = Visualizer(self.image_new, metadata=self.metadata)
                demo = visual.draw_instance_predictions(inst_seg, threshold=self.treshold) # rgb Image
                self.save_instance(demo, image_name)
                
                scores = inst_seg.scores.cpu()
                keep = (scores > self.treshold)
                scores = scores[keep]
                masks = inst_seg.pred_masks[keep]
                bboxes = inst_seg.pred_boxes.tensor[keep]
                classes = inst_seg.pred_classes.cpu()[keep]

                if masks.shape[0] > 0:
                    masks = masks[:, : masks.shape[1], : masks.shape[2]].expand(1, -1, -1, -1)
                    # print("Before interpolation", masks.shape)
                    masks = F.interpolate(
                        masks, size=(self.height, self.width), mode="nearest"
                    )[0]
                    # print("After interpolation", masks.shape)

                    height_box = self.height
                    width_box = self.width
                    x1 = bboxes[:,0]
                    y1 = bboxes[:,1]
                    x2 = bboxes[:,2]
                    y2 = bboxes[:,3]
                    bboxes = torch.stack([x1,y1,x2,y2]).permute(1,0)
                    scale = torch.tensor([width_box/self.cur_width, height_box/self.cur_height, width_box/self.cur_width, height_box/self.cur_height])[None,:].to(bboxes.device)
                    bboxes = bboxes*scale

                scores = scores.numpy()
                masks = masks.cpu().numpy()
                bboxes = bboxes.cpu().numpy()
                classes = classes.cpu().numpy()

                for obj_idx, score in enumerate(scores):
                    image_id = self.images_ids[image_name]
                    category_id = classes[obj_idx] + 1
                    bbox = [
                            int(bboxes[obj_idx].tolist()[0]),
                            int(bboxes[obj_idx].tolist()[1]),
                            int(bboxes[obj_idx].tolist()[2] - bboxes[obj_idx].tolist()[0]),
                            int(bboxes[obj_idx].tolist()[3] - bboxes[obj_idx].tolist()[1]),
                    ]
                    # rle mask in list format
                    segmentation, area = self.binary_mask_to_rle(masks[obj_idx])

                    # self.draw_instance(category_id, masks[obj_idx], bboxes[obj_idx])

                    self.DT_annotation = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "segmentation": segmentation,
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                        "score": float(score),
                    }
                    self.predicted_annotation.append(self.DT_annotation)

                # self.save_instance(image_name)
                i=0
                # break
        
        self.total_time /= len(self.images_files)
        self.skipped_first_time /= len(self.images_files) - TIME_NUMBER_START
        self.save_annotations()    

    def draw_instance(self, category_id=None, mask=None, box=None):
        self.image_ori = cv2.rectangle(self.image_ori, (int(box.tolist()[0]), int(box.tolist()[1])), \
            (int(box.tolist()[2]), int(box.tolist()[3])), color=self.thing_colors[category_id-1], thickness=3)

        bin_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        color_arr = np.full_like(self.image_ori, self.thing_colors[category_id-1])
        mask_img = np.where(bin_mask==[1, 1, 1], color_arr, self.image_ori)

        self.image_ori = cv2.addWeighted(self.image_ori, 0.5, mask_img, 0.5, 0)

        # plt.imshow(self.image_ori)
        # plt.show()
        i=0

    def resize_img(self):
        self.cur_width = MAX_WIDTH
        self.cur_height = MAX_HEIGHT
        dim = (self.cur_width, self.cur_height)
        self.image_new = cv2.resize(self.image_ori.copy(), dim, interpolation = cv2.INTER_AREA)
    
    def encode_binary_mask(self, mask: np.ndarray) -> t.Text:
        """Converts a binary mask into OID challenge encoding ascii text."""

        # check input mask --
        if mask.dtype != np.bool_:
            raise ValueError(
                "encode_binary_mask expects a binary mask, received dtype == %s" %
                mask.dtype)

        mask = np.squeeze(mask)
        if len(mask.shape) != 2:
            raise ValueError(
                "encode_binary_mask expects a 2d mask, received shape == %s" %
                mask.shape)

        # convert input mask to expected COCO API input --
        mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask_to_encode = mask_to_encode.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)

        # RLE encode mask --
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

        # compress and base64 encoding --
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str.decode()

    def binary_mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        area = 0
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            num = len(list(elements))
            counts.append(num)
            if value == 1:
                area += num
        return rle, area

    def save_instance(self, demo, image_name):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        demo.save(os.path.join(self.output_root, "inst_" + image_name))

        # cv2.imwrite(os.path.join(self.output_root, "inst_" + image_name), self.image_ori)

    def save_annotations(self):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)

        with open(self.output_root + "/annotation.json", "w") as outfile:
            json.dump(self.predicted_annotation, outfile)
        self.show_info()  

    def show_info(self):
        print("TOTAL TIME: ", self.total_time)
        print("TRUTH TIME: ", self.skipped_first_time)

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
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt["user_dir"] = absolute_user_dir

    # opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt["WEIGHT"])
    output_root = cmdline_args.output_root
    images_pth = cmdline_args.images_path
    annotation_pth = cmdline_args.annotation_path

    instance = InstanceSegm(opt, pretrained_pth, output_root, images_pth, annotation_pth)
    instance.start_instance()


if __name__ == "__main__":
    main()
    sys.exit(0)
