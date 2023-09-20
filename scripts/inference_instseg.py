import os
import sys
import logging
import tqdm
import glob
import gc
import cv2
import time
import json
from itertools import groupby

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(2)

import torch
from torchvision import transforms

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


REQUEST_LIST = [            # LIST OF REQUESTS
    "wire",
    "curb",
    "bike lane",
    "pedestrian area",
    "rail track",
    "road",
    "sidewalk",
    "sand",
    "terrain",
    "vegetation",
    "manhole",
    "catch basin",
]
TIME_NUMBER_START = 2       # START NUMBER OF CALCULATING AVERAGE INFERENCE TIME
MAX_WIDTH = 1280            # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
MAX_HEIGHT = 1024           # NEED THIS CONSTANT TO AVOID "CUDA out of memory"
THRESHOLD = 0.1             # THRESHOLD FOR VISUALIZATION AND ANNOTATION


class InstanceSegm():
    def __init__(self, opt, pretrained_pth, output_root, images_pth, annotation_pth) -> None:
        self.opt = opt
        self.pretrained_pth = pretrained_pth
        self.output_root = output_root
        self.images_pth = images_pth
        self.annotation_pth = annotation_pth
        self.images_ids = {}

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
        self.thing_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(self.thing_classes))]
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
                self.cur_width = self.image_ori.size[0]
                self.cur_height = self.image_ori.size[1]

                image = self.transform(self.image_ori)
                self.image_ori = np.asarray(self.image_ori)

                # If image too large it needs to beeing resized to avoid "CUDA out of memory"
                if self.cur_width > MAX_WIDTH or self.cur_height > MAX_HEIGHT:
                    self.resize_img()

                self.images = torch.from_numpy(self.image_ori.copy()).permute(2,0,1).cuda()
                batch_inputs = [{"image": self.images, "height": self.cur_height, "width": self.cur_width}]
                
                cur_time = time.time()
                # Get output from model
                outputs = self.model.forward(batch_inputs)
                self.total_time += time.time() - cur_time
                if idx >= TIME_NUMBER_START:
                    self.skipped_first_time += time.time() - cur_time
                
                visual = Visualizer(self.image_ori, metadata=self.metadata)

                inst_seg = outputs[-1]["instances"]
                inst_seg.pred_masks = inst_seg.pred_masks.cpu()
                inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
                demo = visual.draw_instance_predictions(inst_seg, threshold=THRESHOLD) # rgb Image

                scores = inst_seg.scores.cpu().numpy()
                keep = (scores > THRESHOLD)
                scores = scores[keep]
                masks = inst_seg.pred_masks.numpy()[keep]
                bboxes = inst_seg.pred_boxes.tensor.numpy()[keep]
                classes = inst_seg.pred_classes.cpu().numpy()[keep]

                for obj_idx, score in enumerate(scores):
                    image_id = self.images_ids[image_name]
                    category_id = classes[obj_idx]
                    bbox = [
                            int(bboxes[obj_idx].tolist()[0]),
                            int(bboxes[obj_idx].tolist()[1]),
                            int(bboxes[obj_idx].tolist()[2] - bboxes[obj_idx].tolist()[0]),
                            int(bboxes[obj_idx].tolist()[3] - bboxes[obj_idx].tolist()[1]),
                    ]
                    # May be it`s needs to rewrite rle masks from list of counts to str format
                    segmentation, area = self.binary_mask_to_rle(masks[obj_idx])

                    self.DT_annotation = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "segmentation": segmentation,
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                        "score": int(score),
                    }
                    self.predicted_annotation.append(self.DT_annotation)

                # break
                self.save_instance(demo, image_name)
        
        self.total_time /= len(self.images_files)
        self.skipped_first_time /= len(self.images_files) - TIME_NUMBER_START

        with open(self.output_root + "/annotation.json", "w") as outfile:
            json.dump(self.predicted_annotation, outfile)
        self.show_info()      

    def resize_img(self):
        w_scale = round(MAX_WIDTH * 100 / self.cur_width)
        h_scale = round(MAX_HEIGHT * 100 / self.cur_height)
        scale_percent = max(w_scale, h_scale)
        self.cur_width = int(self.cur_width * scale_percent / 100)
        self.cur_height = int(self.cur_height * scale_percent / 100)
        dim = (self.cur_width, self.cur_height)
        self.image_ori = cv2.resize(self.image_ori, dim, interpolation = cv2.INTER_AREA)
    
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

    def show_info(self):
        print("TOTAL TIME: ", self.total_time)
        print("TRUTH TIME: ", self.skipped_first_time)


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
