"""
research_v3/phase10_routing/metrics.py

Evaluation metrics for Answer-Region Detection, Page Mapping, and Structural Routing:
1. Bounding Box Intersection-over-Union (IoU)
2. Precision / Recall at IoU thresholds (IoU >= 0.50, IoU >= 0.75)
3. Page Assignment Accuracy
4. Subquestion Slot Accuracy
5. Cross-Question Contamination Rate (bleeding into adjacent questions)
6. Boundary Bleed / Truncation Rate
7. Student Omission Classification Metrics (Precision, Recall, F1)
8. Multi-Page Continuation Linking Completeness
9. Failure Taxonomy Classification
"""

from typing import Dict, Any, List, Optional, Tuple


def compute_bbox_iou(
    boxA: Optional[Tuple[float, float, float, float]],
    boxB: Optional[Tuple[float, float, float, float]]
) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes [x0, y0, x1, y1].
    Returns 0.0 if either box is None or union is zero.
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_overlap_ratio(
    target_box: Optional[Tuple[float, float, float, float]],
    intruder_box: Optional[Tuple[float, float, float, float]]
) -> float:
    """
    Computes the fraction of target_box area that is overlapped by intruder_box.
    Returns 0.0 if either box is None or target area is zero.
    """
    if target_box is None or intruder_box is None:
        return 0.0

    xA = max(target_box[0], intruder_box[0])
    yA = max(target_box[1], intruder_box[1])
    xB = min(target_box[2], intruder_box[2])
    yB = min(target_box[3], intruder_box[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    target_area = max(0.0, target_box[2] - target_box[0]) * max(0.0, target_box[3] - target_box[1])
    if target_area <= 0.0:
        return 0.0
    return float(inter_area / target_area)


def evaluate_routing_prediction(
    gt_record: Dict[str, Any],
    pred_record: Dict[str, Any],
    adjacent_gt_boxes_on_page: Optional[List[Tuple[float, float, float, float]]] = None
) -> Dict[str, Any]:
    """
    Evaluates a single subquestion routing prediction against ground truth.
    
    gt_record fields:
      student_id, subquestion_id, page_id, x0, y0, x1, y1, is_omitted
    pred_record fields:
      pred_page_id, pred_subquestion_id, pred_bbox, is_omitted
    """
    is_gt_omitted = gt_record.get("is_omitted", False)
    is_pred_omitted = pred_record.get("is_omitted", False)
    gt_subq = gt_record.get("subquestion_id", "")
    pred_subq = pred_record.get("pred_subquestion_id", "")

    # Ground truth box
    if not is_gt_omitted and gt_record.get("x0") is not None:
        gt_box = (
            float(gt_record["x0"]),
            float(gt_record["y0"]),
            float(gt_record["x1"]),
            float(gt_record["y1"])
        )
    else:
        gt_box = None

    pred_box = pred_record.get("pred_bbox")
    if pred_box is not None:
        pred_box = (
            float(pred_box[0]),
            float(pred_box[1]),
            float(pred_box[2]),
            float(pred_box[3])
        )

    # 1. Omission Handling
    if is_gt_omitted:
        if is_pred_omitted:
            return {
                "correct_page": True,
                "exact_slot_match": True,
                "iou": 1.0,
                "iou_ge_05": 1,
                "iou_ge_75": 1,
                "omission_correct": 1,
                "continuation_complete": 1,
                "cross_question_contaminated": 0,
                "boundary_bleed": 0,
                "answer_covered": 1,
                "error_category": "none"
            }
        else:
            return {
                "correct_page": False,
                "exact_slot_match": False,
                "iou": 0.0,
                "iou_ge_05": 0,
                "iou_ge_75": 0,
                "omission_correct": 0,
                "continuation_complete": 1,
                "cross_question_contaminated": 0,
                "boundary_bleed": 0,
                "answer_covered": 0,
                "error_category": "false_positive_written"
            }

    # If written in GT but predicted omitted:
    if is_pred_omitted:
        return {
            "correct_page": False,
            "exact_slot_match": False,
            "iou": 0.0,
            "iou_ge_05": 0,
            "iou_ge_75": 0,
            "omission_correct": 0,
            "continuation_complete": 0,
            "cross_question_contaminated": 0,
            "boundary_bleed": 0,
            "answer_covered": 0,
            "error_category": "missed_region"
        }

    # Written in both GT and Prediction:
    correct_page = (pred_record.get("pred_page_id") == gt_record.get("page_id"))
    exact_slot_match = (pred_subq == gt_subq)

    # IoU is evaluated only on the correct page; cross-page errors receive 0.0 IoU
    if correct_page and gt_box is not None and pred_box is not None:
        iou = compute_bbox_iou(gt_box, pred_box)
    else:
        iou = 0.0

    iou_ge_05 = 1 if iou >= 0.50 else 0
    iou_ge_75 = 1 if iou >= 0.75 else 0

    # Cross-question contamination:
    # 1. Slot misattribution (e.g. answer 1b merged into slot 1a)
    # 2. Predicted box bleeds into an adjacent question's territory beyond ground-truth baseline
    cross_contaminated = 0
    if not exact_slot_match and correct_page:
        cross_contaminated = 1
    elif correct_page and pred_box is not None and adjacent_gt_boxes_on_page:
        for adj_box in adjacent_gt_boxes_on_page:
            if gt_box is not None and compute_bbox_iou(adj_box, gt_box) < 0.20:
                pred_overlap = compute_overlap_ratio(adj_box, pred_box)
                gt_baseline_overlap = compute_overlap_ratio(adj_box, gt_box)
                if pred_overlap >= 0.15 and (pred_overlap - gt_baseline_overlap) > 0.10:
                    cross_contaminated = 1
                    break

    # Boundary bleed: correct page and slot, but truncated or over-expanded (IoU < 0.50)
    boundary_bleed = 1 if (correct_page and exact_slot_match and iou < 0.50) else 0

    # Answer covered: IoU >= 0.50 or substantial overlap
    answer_covered = 1 if (correct_page and iou >= 0.50) else 0

    # Error categorization
    if not correct_page:
        error_cat = "wrong_page"
    elif not exact_slot_match:
        error_cat = "wrong_subquestion_header"
    elif cross_contaminated:
        error_cat = "cross_question_merge"
    elif boundary_bleed:
        error_cat = "answer_split_or_truncated"
    elif iou >= 0.50:
        error_cat = "none"
    else:
        error_cat = "suboptimal_localization"

    return {
        "correct_page": correct_page,
        "exact_slot_match": exact_slot_match,
        "iou": iou,
        "iou_ge_05": iou_ge_05,
        "iou_ge_75": iou_ge_75,
        "omission_correct": 1,
        "continuation_complete": 1 if pred_record.get("continuation_complete", True) else 0,
        "cross_question_contaminated": cross_contaminated,
        "boundary_bleed": boundary_bleed,
        "answer_covered": answer_covered,
        "error_category": error_cat
    }
