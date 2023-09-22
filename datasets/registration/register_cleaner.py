# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
from pycocotools import mask as maskUtils

CLEANER_CATEGORIES = [
        {'id': 1, 'name': 'firehose', 'supercategory': 'firehose'},
        {'id': 2, 'name': 'hose', 'supercategory': 'hose'},
        {'id': 3, 'name': 'wire', 'supercategory': 'wire'},
        {'id': 4, 'name': 'rope', 'supercategory': 'wire'},
        {'id': 5, 'name': 'waste', 'supercategory': 'waste'},
        {'id': 6, 'name': 'puddle', 'supercategory': 'puddle'},
        {'id': 7, 'name': 'pit', 'supercategory': 'breakroad'},
        {'id': 8, 'name': 'bump', 'supercategory': 'breakroad'},
        {'id': 9, 'name': 'sidewalk', 'supercategory': 'sidewalk'},
        {'id': 10, 'name': 'pedestrian area', 'supercategory': 'sidewalk'},
        {'id': 11, 'name': 'grass', 'supercategory': 'terrain'},
        {'id': 12, 'name': 'vegetation', 'supercategory': 'terrain'},
        {'id': 13, 'name': 'terrain', 'supercategory': 'terrain'},
        {'id': 14, 'name': 'road', 'supercategory': 'road'},
        {'id': 15, 'name': 'curb', 'supercategory': 'breakroad'},
        {'id': 16, 'name': 'bike lane', 'supercategory': 'sidewalk'},
        {'id': 17, 'name': 'rail track', 'supercategory': 'breakroad'},
        {'id': 18, 'name': 'sand', 'supercategory': 'breakroad'},
        {'id': 19, 'name': 'manhole', 'supercategory': 'breakroad'},
        {'id': 20, 'name': 'catch basin', 'supercategory': 'breakroad'},
    ]

_PREDEFINED_SPLITS_CLEANER = {
    "cleaner": {
        "cleaner_v1_train": ("valid_cleaner/train", "valid_cleaner/train.json"),
        "cleaner_v1_val": ("valid_cleaner/val", "valid_cleaner/val.json"),
    },
}

def get_cleaner_instances_meta_v1():
    assert len(CLEANER_CATEGORIES) == 20
    cat_ids = [k["id"]-1 for k in CLEANER_CATEGORIES]
    assert min(cat_ids) == 0 and max(cat_ids) == len(
        cat_ids
    )-1, "Category ids are not in [0, #categories-1], as expected"
    # Ensure that the category list is sorted by id
    #thing_ids = [k["id"] for k in ROSBAG_CATEGORIES if k["name"] not in ["wall", "floor", "ceiling"]]
    thing_ids = [k["id"] for k in CLEANER_CATEGORIES]
    # lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    # thing_classes = [k["name"] for k in O365_CATEGORIES]
    def preprocess_name(name):
        name = name.lower().strip()
        name = name.replace('_', ' ')
        return name
    thing_classes = [preprocess_name(k["name"]) for k in CLEANER_CATEGORIES]
    meta = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
            }
    return meta


def register_cleaner_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_cleaner_json(image_root, json_file, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_cleaner_json(image_root, annot_json, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    with PathManager.open(annot_json) as f:
        json_info = json.load(f)

    imageid2ann = {}
    for anno in json_info["annotations"]:
        image_id = anno['image_id']
        if image_id not in imageid2ann:
            imageid2ann[image_id] = []
        anno["bbox_mode"] = BoxMode.XYWH_ABS
        anno["category_id"] = anno["category_id"] - 1
        
        if type(anno['segmentation']) == list and len(anno['segmentation']) < 1:
            continue
        imageid2ann[image_id] += [anno]

    ret = []
    cnt_empty = 0
    for image in json_info["images"]:
        image_file = os.path.join(image_root, image["file_name"])
        image_id = image['id']
        if image_id not in imageid2ann:
            cnt_empty += 1
            continue

        anns = imageid2ann[image_id]
        h, w = image['height'], image['width']
        for ann in anns:
            segm = ann['segmentation']
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)
            else:
                # rle
                rle = ann['segmentation']
            ann['segmentation'] = rle
        
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "height": image['height'],
                "width": image['width'],
                "annotations": imageid2ann[image_id],
            }
        )

    print("Empty annotations: {}".format(cnt_empty))
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_all_cleaner(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CLEANER.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_cleaner_instances(
                key,
                get_cleaner_instances_meta_v1(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_root = os.getenv("DATASET3", "datasets")
register_all_cleaner(_root)