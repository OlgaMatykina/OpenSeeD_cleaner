import os
import sys
import logging
import tqdm
import glob
import gc
import cv2

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


REQUEST_LIST = [
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

MAX_WIDTH = 1280
MAX_HEIGHT = 1024


class InstanceSegm():
    def __init__(self, opt, pretrained_pth, output_root, images_pth, annotation_pth) -> None:
        self.opt = opt
        self.pretrained_pth = pretrained_pth
        self.output_root = output_root
        self.images_pth = images_pth
        self.annotation_pth = annotation_pth

        self.init_model()

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
        self.metadata = MetadataCatalog.get('demo')
        self.model.model.metadata = self.metadata
        self.model.model.sem_seg_head.num_classes = len(self.thing_classes)

        self.output_subroot = self.images_pth.split("/")[-1]
        self.output_root += self.output_subroot
        print("OUTPUT DIR: ", self.output_root)

    def start_instance(self):
        self.images_files = sorted(glob.glob(f"{self.images_pth}/*"))
        for image_pth in tqdm.tqdm(self.images_files):
            image_name = image_pth.split("/")[-1]

            with torch.no_grad():
                self.image_ori = Image.open(image_pth).convert('RGB')
                self.cur_width = self.image_ori.size[0]
                self.cur_height = self.image_ori.size[1]

                image = self.transform(self.image_ori)
                self.image_ori = np.asarray(self.image_ori)

                # If image too large it needs to beeing resized to avoid "CUDA out of memory"
                if self.cur_width > MAX_WIDTH or self.cur_height > MAX_HEIGHT:
                    self.resize_img()

                self.images = torch.from_numpy(self.image_ori.copy()).permute(2,0,1).cuda()

                batch_inputs = [{'image': self.images, 'height': self.cur_height, 'width': self.cur_width}]
                outputs = self.model.forward(batch_inputs)
                visual = Visualizer(self.image_ori, metadata=self.metadata)

                inst_seg = outputs[-1]['instances']
                inst_seg.pred_masks = inst_seg.pred_masks.cpu()
                inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
                demo = visual.draw_instance_predictions(inst_seg) # rgb Image

                self.save_instance(demo, image_name)

    def resize_img(self):
        w_scale = round(MAX_WIDTH * 100 / self.cur_width)
        h_scale = round(MAX_HEIGHT * 100 / self.cur_height)
        scale_percent = max(w_scale, h_scale)
        self.cur_width = int(self.cur_width * scale_percent / 100)
        self.cur_height = int(self.cur_height * scale_percent / 100)
        dim = (self.cur_width, self.cur_height)
        self.image_ori = cv2.resize(self.image_ori, dim, interpolation = cv2.INTER_AREA)

    def save_instance(self, demo, image_name):
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        demo.save(os.path.join(self.output_root, "inst_" + image_name))


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = cmdline_args.output_root
    images_pth = cmdline_args.images_path
    annotation_pth = cmdline_args.annotation_path

    instance = InstanceSegm(opt, pretrained_pth, output_root, images_pth, annotation_pth)
    instance.start_instance()


if __name__ == "__main__":
    main()
    sys.exit(0)
