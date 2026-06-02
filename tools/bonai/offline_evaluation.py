import argparse
import csv
import json
import math
import os
import os.path as osp
import pickle
import re
from collections import defaultdict, OrderedDict

import cv2
import numpy as np

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    shapely_wkt = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline BONAI evaluation for MMDetection 3.x results.')
    parser.add_argument(
        '--result',
        help='BONAIRLEMetric intermediate file, e.g. *.bonai_rle.pkl.')
    parser.add_argument(
        '--gt-roof-csv',
        default='data/BONAI/csv/'
        'shanghai_xian_v3_merge_val_roof_crop1024_gt_minarea500.csv',
        help='Ground-truth roof CSV in BONAI format.')
    parser.add_argument(
        '--gt-footprint-csv',
        default='data/BONAI/csv/'
        'shanghai_xian_v3_merge_val_footprint_crop1024_gt_minarea500.csv',
        help='Ground-truth footprint CSV in BONAI format.')
    parser.add_argument(
        '--pred-roof-csv',
        help='Prediction roof CSV. If omitted, it is dumped from --result.')
    parser.add_argument(
        '--pred-footprint-csv',
        help='Prediction footprint CSV. If omitted, it is dumped from --result.'
    )
    parser.add_argument(
        '--out-dir',
        help='Directory for prediction CSVs, summary CSV and JSON results.')
    parser.add_argument(
        '--summary-file',
        help='Summary CSV path. Defaults to <out-dir>/bonai_offline_summary.csv.'
    )
    parser.add_argument('--score-thr', type=float, default=0.4)
    parser.add_argument('--min-area', type=float, default=500)
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold used by BONAI segmentation evaluation.')
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=(1024, 1024),
        metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument(
        '--backend',
        choices=('auto', 'shapely', 'raster'),
        default='auto',
        help='Polygon IoU backend. auto uses shapely when available.')
    parser.add_argument(
        '--no-skip-empty',
        action='store_false',
        dest='skip_empty',
        help='Count images with empty GT or predictions as FN/FP.')
    parser.set_defaults(skip_empty=True)
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Only dump prediction CSVs from --result, without evaluation.')
    return parser.parse_args()


def mkdir_or_exist(path):
    if path:
        os.makedirs(path, exist_ok=True)


def get_out_dir(args):
    if args.out_dir:
        return args.out_dir
    if args.result:
        return osp.join(osp.dirname(args.result), 'offline_bonai_eval')
    return 'work_dirs/bonai_offline_eval'


def image_name(record):
    img_path = record.get('img_path')
    if img_path:
        return osp.splitext(osp.basename(img_path))[0]
    img_id = record.get('img_id')
    if img_id is None:
        return None
    return str(img_id)


def load_rle_results(result_file):
    with open(result_file, 'rb') as file:
        data = pickle.load(file)
    if not isinstance(data, dict) or 'results' not in data:
        raise TypeError(
            f'{result_file} is not a BONAIRLEMetric result file. '
            'Run tools/test.py with BONAIRLEMetric enabled, or provide '
            '--pred-roof-csv and --pred-footprint-csv directly.')
    return data['results'], data.get('meta', {})


def ensure_rle_counts_bytes(rle):
    rle = dict(rle)
    if isinstance(rle.get('counts'), str):
        rle['counts'] = rle['counts'].encode()
    return rle


def decode_rle(rle):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            'pycocotools is required to convert *.bonai_rle.pkl to CSV.'
        ) from exc
    return mask_utils.decode(ensure_rle_counts_bytes(rle)).astype(bool)


def mask_area_from_rle(rle):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            'pycocotools is required to convert *.bonai_rle.pkl to CSV.'
        ) from exc
    return float(mask_utils.area(ensure_rle_counts_bytes(rle)))


def contour_to_ring(contour):
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points


def ring_to_wkt(ring):
    coords = ', '.join(f'{int(x)} {int(y)}' for x, y in ring)
    return f'(({coords}))'


def mask_to_wkt(mask, min_area):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue
        ring = contour_to_ring(contour)
        if ring is not None:
            rings.append(ring)
    if not rings:
        return None
    rings.sort(key=lambda ring: abs(cv2.contourArea(ring.astype(np.float32))),
               reverse=True)
    if len(rings) == 1:
        return f'POLYGON {ring_to_wkt(rings[0])}'
    polygons = ', '.join(ring_to_wkt(ring) for ring in rings)
    return f'MULTIPOLYGON ({polygons})'


def dump_prediction_csvs(results, roof_csv, footprint_csv, score_thr,
                         min_area):
    mkdir_or_exist(osp.dirname(roof_csv))
    mkdir_or_exist(osp.dirname(footprint_csv))

    counters = {'roof': defaultdict(int), 'footprint': defaultdict(int)}
    with open(roof_csv, 'w', newline='') as roof_file, open(
            footprint_csv, 'w', newline='') as footprint_file:
        roof_writer = csv.writer(roof_file)
        footprint_writer = csv.writer(footprint_file)
        header = ['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence']
        roof_writer.writerow(header)
        footprint_writer.writerow(header)

        for gt, pred in results:
            cur_image_name = image_name(pred) or image_name(gt)
            scores = np.asarray(pred.get('scores', []), dtype=np.float32)
            roof_rles = pred.get('roof', [])
            footprint_rles = pred.get('footprint', [])

            for idx, score in enumerate(scores):
                if score < score_thr or idx >= len(roof_rles):
                    continue
                if mask_area_from_rle(roof_rles[idx]) < min_area:
                    continue

                roof_mask = decode_rle(roof_rles[idx])
                roof_wkt = mask_to_wkt(roof_mask, min_area)
                if roof_wkt:
                    building_id = counters['roof'][cur_image_name]
                    roof_writer.writerow(
                        [cur_image_name, building_id, roof_wkt,
                         float(score)])
                    counters['roof'][cur_image_name] += 1

                if idx >= len(footprint_rles):
                    continue
                footprint_mask = decode_rle(footprint_rles[idx])
                footprint_wkt = mask_to_wkt(footprint_mask, min_area)
                if footprint_wkt:
                    building_id = counters['footprint'][cur_image_name]
                    footprint_writer.writerow([
                        cur_image_name, building_id, footprint_wkt,
                        float(score)
                    ])
                    counters['footprint'][cur_image_name] += 1


def split_rings(wkt):
    return re.findall(r'\(([^()]+)\)', wkt)


def parse_ring(ring_text):
    points = []
    for coord in ring_text.split(','):
        xy = coord.strip().split()
        if len(xy) < 2:
            continue
        points.append([float(xy[0]), float(xy[1])])
    if len(points) < 3:
        return None
    points = np.asarray(points, dtype=np.float32)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points


def rasterize_wkt(wkt, image_size):
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for ring_text in split_rings(wkt):
        ring = parse_ring(ring_text)
        if ring is None:
            continue
        cv2.fillPoly(mask, [np.rint(ring).astype(np.int32)], 1)
    return mask.astype(bool)


def load_csv_objects(csv_file, image_size, backend):
    use_shapely = backend == 'shapely' or (backend == 'auto'
                                           and shapely_wkt is not None)
    if backend == 'shapely' and shapely_wkt is None:
        raise ImportError('shapely is required when --backend=shapely.')

    groups = defaultdict(list)
    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            polygon_wkt = row['PolygonWKT_Pix']
            obj = dict(
                image_id=row['ImageId'],
                building_id=row.get('BuildingId'),
                confidence=float(row.get('Confidence', 1.0) or 1.0),
                wkt=polygon_wkt)
            if use_shapely:
                geom = shapely_wkt.loads(polygon_wkt)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                obj['geom'] = geom
                obj['area'] = float(geom.area)
            else:
                mask = rasterize_wkt(polygon_wkt, image_size)
                obj['mask'] = mask
                obj['area'] = float(mask.sum())
            groups[obj['image_id']].append(obj)
    return groups, 'shapely' if use_shapely else 'raster'


def pairwise_iou(pred_objects, gt_objects, backend):
    ious = np.zeros((len(pred_objects), len(gt_objects)), dtype=np.float32)
    for pred_idx, pred in enumerate(pred_objects):
        for gt_idx, gt in enumerate(gt_objects):
            if backend == 'shapely':
                intersection = float(pred['geom'].intersection(
                    gt['geom']).area)
                union = pred['area'] + gt['area'] - intersection
            else:
                intersection = float(
                    np.logical_and(pred['mask'], gt['mask']).sum())
                union = pred['area'] + gt['area'] - intersection
            ious[pred_idx, gt_idx] = intersection / (union + 1.0)
    return ious


def count_image_confusion(gt_objects, pred_objects, iou_thr, skip_empty,
                          backend):
    num_gts = len(gt_objects)
    num_preds = len(pred_objects)
    if skip_empty and (num_gts == 0 or num_preds == 0):
        return 0, 0, 0, np.empty((0, 2), dtype=np.int64)
    if num_gts == 0:
        return 0, num_preds, 0, np.empty((0, 2), dtype=np.int64)
    if num_preds == 0:
        return 0, 0, num_gts, np.empty((0, 2), dtype=np.int64)

    ious = pairwise_iou(pred_objects, gt_objects, backend)
    matched = np.argwhere(ious >= iou_thr)
    if matched.size == 0:
        return 0, num_preds, num_gts, matched

    pred_tp_indexes = matched[:, 0].tolist()
    gt_tp_indexes = matched[:, 1].tolist()
    tp = len(gt_tp_indexes)
    fp = len(set(range(num_preds)) - set(pred_tp_indexes))
    fn = len(set(range(num_gts)) - set(gt_tp_indexes))
    return tp, fp, fn, matched


def evaluate_csv(gt_csv, pred_csv, image_size, iou_thr, skip_empty, backend):
    gt_groups, gt_backend = load_csv_objects(gt_csv, image_size, backend)
    pred_groups, pred_backend = load_csv_objects(pred_csv, image_size, backend)
    assert gt_backend == pred_backend

    total_tp = total_fp = total_fn = 0
    for cur_image_name in gt_groups.keys():
        tp, fp, fn, _ = count_image_confusion(
            gt_groups.get(cur_image_name, []),
            pred_groups.get(cur_image_name, []), iou_thr, skip_empty,
            gt_backend)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (total_tp / (total_tp + total_fp)
                 if total_tp + total_fp > 0 else 0.0)
    recall = (total_tp / (total_tp + total_fn)
              if total_tp + total_fn > 0 else 0.0)
    f1_score = (2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0.0)
    return OrderedDict(
        F1_score=f1_score,
        Precision=precision,
        Recall=recall,
        TP=total_tp,
        FN=total_fn,
        FP=total_fp)


def rle_pairwise_iou(pred_rles, gt_rles):
    pred_masks = [decode_rle(rle) for rle in pred_rles]
    gt_masks = [decode_rle(rle) for rle in gt_rles]
    pred_areas = [float(mask.sum()) for mask in pred_masks]
    gt_areas = [float(mask.sum()) for mask in gt_masks]
    ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            intersection = float(np.logical_and(pred_mask, gt_mask).sum())
            union = pred_areas[pred_idx] + gt_areas[gt_idx] - intersection
            ious[pred_idx, gt_idx] = intersection / (union + 1.0)
    return ious


def evaluate_offsets_from_rle(results, score_thr, min_area, iou_thr,
                              skip_empty):
    gt_offsets = []
    pred_offsets = []
    for gt, pred in results:
        valid_pred_indexes = []
        for idx, score in enumerate(pred.get('scores', [])):
            if score < score_thr:
                continue
            if idx >= len(pred.get('roof_areas', [])):
                continue
            if pred['roof_areas'][idx] < min_area:
                continue
            valid_pred_indexes.append(idx)

        gt_masks = gt.get('footprint', [])
        pred_masks = [pred['footprint'][idx] for idx in valid_pred_indexes]
        if skip_empty and (len(gt_masks) == 0 or len(pred_masks) == 0):
            continue
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            continue

        matched = np.argwhere(rle_pairwise_iou(pred_masks, gt_masks) >= iou_thr)
        if matched.size == 0:
            continue

        cur_pred_offsets = [
            pred['offsets'][idx] if idx < len(pred.get('offsets', [])) else
            np.zeros(2, dtype=np.float32) for idx in valid_pred_indexes
        ]
        for pred_idx, gt_idx in matched:
            if gt_idx >= len(gt.get('footprint_offsets', [])):
                continue
            gt_offsets.append(gt['footprint_offsets'][gt_idx])
            pred_offsets.append(cur_pred_offsets[pred_idx])

    if not gt_offsets:
        return OrderedDict(aEPE=0.0, aAE=0.0)

    gt_offsets = np.asarray(gt_offsets, dtype=np.float32)
    pred_offsets = np.asarray(pred_offsets, dtype=np.float32)
    error_vectors = gt_offsets - pred_offsets
    epe = np.sqrt(error_vectors[:, 0]**2 + error_vectors[:, 1]**2)
    gt_angle = np.arctan2(gt_offsets[:, 1], gt_offsets[:, 0])
    pred_angle = np.arctan2(pred_offsets[:, 1], pred_offsets[:, 0])
    angle_error = np.abs(gt_angle - pred_angle)
    return OrderedDict(aEPE=float(epe.mean()), aAE=float(angle_error.mean()))


def write_summary_csv(summary_file, results, meta_info):
    mkdir_or_exist(osp.dirname(summary_file))
    with open(summary_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Meta Info'])
        for key, value in meta_info.items():
            writer.writerow([key, value])
        writer.writerow([''])
        for mask_type in ['roof', 'footprint']:
            result = results[mask_type]
            writer.writerow([mask_type])
            writer.writerow([dict(result)])
            writer.writerow(['F1 Score', result['F1_score']])
            writer.writerow(['Precision', result['Precision']])
            writer.writerow(['Recall', result['Recall']])
            writer.writerow(['True Positive', result['TP']])
            writer.writerow(['False Positive', result['FP']])
            writer.writerow(['False Negative', result['FN']])
            writer.writerow([''])
        writer.writerow([''])


def main():
    args = parse_args()
    out_dir = get_out_dir(args)
    mkdir_or_exist(out_dir)

    pred_roof_csv = args.pred_roof_csv or osp.join(out_dir, 'pred_roof.csv')
    pred_footprint_csv = args.pred_footprint_csv or osp.join(
        out_dir, 'pred_footprint.csv')

    rle_results = None
    if args.result:
        rle_results, _ = load_rle_results(args.result)
        if args.pred_roof_csv is None or args.pred_footprint_csv is None:
            dump_prediction_csvs(rle_results, pred_roof_csv,
                                 pred_footprint_csv, args.score_thr,
                                 args.min_area)

    if args.format_only:
        print(f'Prediction roof CSV: {pred_roof_csv}')
        print(f'Prediction footprint CSV: {pred_footprint_csv}')
        return

    if not osp.exists(pred_roof_csv) or not osp.exists(pred_footprint_csv):
        raise FileNotFoundError(
            'Prediction CSVs are required. Provide --result so they can be '
            'generated, or pass --pred-roof-csv and --pred-footprint-csv.')

    image_size = tuple(args.image_size)
    results = OrderedDict()
    results['roof'] = evaluate_csv(args.gt_roof_csv, pred_roof_csv,
                                   image_size, args.iou_thr,
                                   args.skip_empty, args.backend)
    results['footprint'] = evaluate_csv(args.gt_footprint_csv,
                                        pred_footprint_csv, image_size,
                                        args.iou_thr, args.skip_empty,
                                        args.backend)

    if rle_results is not None:
        results['offset'] = evaluate_offsets_from_rle(
            rle_results, args.score_thr, args.min_area, args.iou_thr,
            args.skip_empty)

    summary_file = args.summary_file or osp.join(out_dir,
                                                 'bonai_offline_summary.csv')
    meta_info = OrderedDict(
        result_file=args.result or '',
        gt_roof_csv_file=args.gt_roof_csv,
        gt_footprint_csv_file=args.gt_footprint_csv,
        pred_roof_csv_file=pred_roof_csv,
        pred_footprint_csv_file=pred_footprint_csv,
        score_thr=args.score_thr,
        min_area=args.min_area,
        iou_thr=args.iou_thr,
        skip_empty=args.skip_empty)
    write_summary_csv(summary_file, results, meta_info)

    json_file = osp.join(out_dir, 'bonai_offline_results.json')
    with open(json_file, 'w') as file:
        json.dump(results, file, indent=2)

    print('Summary (BONAI offline style):')
    for mask_type in ['roof', 'footprint']:
        result = results[mask_type]
        print(
            f'{mask_type}: F1={result["F1_score"]:.6f}, '
            f'Precision={result["Precision"]:.6f}, '
            f'Recall={result["Recall"]:.6f}, TP={result["TP"]}, '
            f'FN={result["FN"]}, FP={result["FP"]}')
    if 'offset' in results:
        print(
            f'offset: aEPE={results["offset"]["aEPE"]:.6f}, '
            f'aAE={results["offset"]["aAE"]:.6f}')
    print(f'Prediction roof CSV: {pred_roof_csv}')
    print(f'Prediction footprint CSV: {pred_footprint_csv}')
    print(f'Summary CSV: {summary_file}')
    print(f'Results JSON: {json_file}')


if __name__ == '__main__':
    main()
