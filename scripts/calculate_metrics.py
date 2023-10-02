from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import logging 


def evaluate_metrics(cocoEval, params = None, display_summary = False):
    # Display the iouThresholds for which the evaluation took place
    if params:
        cocoEval.params = params  
    print("IoU Thresholds: ",cocoEval.params.iouThrs)
    iou_lookup = {float(format(val, '.2f')): index for index, val in enumerate(cocoEval.params.iouThrs)} 
    
    cocoEval.evaluate() # Calculates the metrics for each class
    cocoEval.accumulate(p = params) # Stores the values in the cocoEval's 'eval' object
    if display_summary:
        cocoEval.summarize()  # Display the metrics.

    # Extract the metrics from accumulated results.
    precision = cocoEval.eval["precision"]
    recall = cocoEval.eval["recall"]
    scores = cocoEval.eval["scores"]

    return precision, recall, scores, iou_lookup


# Print final results
def display_metrics(precision, recall, scores, iou_lookup, class_name=None, log_path='evaluation.txt'):   
    # Initialize logger
    logger = logging.getLogger('eval_log')
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

        # logger.warning("| IoU | mAP | F1-Score | Precision | Recall |")
        # logger.warning("|-----|-----|----------|-----------|--------|")
    
    # iou = list(iou_lookup)[0]
    for iou in iou_lookup.keys():
        precesion_iou = precision[iou_lookup[iou], :, :, 0, -1].mean(1)
        scores_iou = scores[iou_lookup[iou], :, :, 0, -1].mean(1)
        recall_iou = recall[iou_lookup[iou], :, 0, -1]
        prec = precesion_iou.mean()
        rec = recall_iou.mean()
                
        if class_name:
            print("{:10s} {:2.2f} {:6.3f} {:2.3f} {:2.2f} {:2.2f}".format(
            class_name, iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec 
            ))
            # logger.warning("|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
            #     class_name, iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec 
            # ))
        else:
            print("{:2.2f} {:6.3f} {:2.3f} {:2.2f} {:2.2f}".format(
            iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec 
            ))
                        
            # logger.warning("|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
            #     iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec 
            # ))


def load_files(annFile, resFile):
    cocoGT = COCO(annFile)
    cocoDT = cocoGT.loadRes(resFile)
    return cocoGT, cocoDT


def calculate_metrics(annFile, resFile, show_eval_summary):
    cocoGT, cocoDT = load_files(annFile, resFile)
    
    annType = ['bbox', 'segm']
    
    for ann in annType:
        cocoEval = COCOeval(cocoGT, cocoDT, ann)
        params = cocoEval.params

        # Calculate the metrics
        precision, recall, scores, iou_lookup = evaluate_metrics(cocoEval, params, show_eval_summary)

        print("| IoU | mAP | F1-Score | Precision | Recall |")
        print("|-----|-----|----------|-----------|--------|")
        # take precision for all classes, all areas and 100 detections
        display_metrics(precision, recall, scores, iou_lookup)

        print("| Class Name | IoU | mAP | F1-Score | Precision | Recall |")
        print("|------------|-----|-----|----------|-----------|--------|")

        # Calculate metrics for each category
        for cat in cocoGT.loadCats(cocoGT.getCatIds()):
            # Calculate the metrics
            params.catIds = [cat["id"]]
            # precision, recall, scores, iou_lookup = evaluate_metrics(cocoEval, params, False)
            # take precision for all classes, all areas and 100 detections
            display_metrics(precision, recall, scores, iou_lookup, class_name=cat["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_gt", help="add path to gt")
    parser.add_argument("path_to_result", help="add path to result")
    args = parser.parse_args()
    calculate_metrics(args.path_to_gt, args.path_to_result, True)

    
