# Script to convert the results for different datasets and add them to a "fiftyone" dataset
# Resources: https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html

import argparse
from pathlib import Path
import pandas as pd
import fiftyone as fo

import os
import sys
sys.path.append(os.getcwd())

from src.utils import bbox_to_fiftyone_format


def add_res_to_fiftyone_samples(dataset, res_path, method_name):
    """
    in: dataset, res_path
    out: samples
    """
    # This is a little messy, but it is to work with the fiftyone iterator and to not open several times the same file.
    prev_csv = None
    df_res = None
    classes = []
    emb_glob = dict()
    with fo.ProgressBar() as pb:
        for i, sample in enumerate(pb(dataset.iter_samples())):
            img_path = Path(sample.filepath)
            # img_name: YYYY_MM_DD_hh_mm-frame.jpg
            img_info = img_path.stem.split('-')
            img_source = img_info[0]
            year_str = img_source.split('_')[0]
            frame = int(img_info[-1])
            res_csv_path = res_path / f"{year_str}/{img_source}.csv"
            if not res_csv_path.exists():
                continue

            # Precomputed metadata in "convert_dataset_to_fiftyone.py"
            img_height = sample['metadata']['height']
            img_width = sample['metadata']['width']

            # Only open in case the image source changed
            if prev_csv != res_csv_path:
                df_res = pd.read_csv(res_csv_path)
            # Filter by frame
            df = df_res[df_res['frame'] == frame]
            if df.empty:
                continue

            detections = []
            for _, row in df.iterrows():
                label_id = str(row['ID']).split('_')[-1]
                # Use dist_id as a "confidence" score in order to establish an order
                dist_id = row['dist_ID']
                conf = 1 - dist_id
                # bbox given in cx, cy, w, h
                # Sample (from fiftyone) accepts bbox relative to the image size
                # Fiftyone bbox format: x1, y1, w, h
                bbox_cxcywh = list(map(int, [row['cx'], row['cy'], row['w'], row['h']]))
                bbox_rel = bbox_to_fiftyone_format(bbox_cxcywh, img_size=[img_width, img_height])
                bbox_area = bbox_cxcywh[2] * bbox_cxcywh[3]
                detections.append(
                    fo.Detection(label=str(row['ID']), bounding_box=bbox_rel, confidence=conf,
                                 label_id=label_id, area=bbox_area)  #label_id is a custom attribute
                )
                if row['ID'] not in classes:
                    classes.append(str(row['ID']))

            sample[f"{method_name}"] = fo.Detections(detections=detections)

            sample.save()

    dataset.default_classes = classes
    dataset.save()


def main():
    # Load dataset previously generated with "convert_dataset_to_fiftyone.py"
    # Sanity check to assure the channel dataset is already generated and the channel is effectively part of the GT.
    datasets_list = fo.list_datasets()
    channels_list = (channel_path.stem for channel_path in ROOT_DATASET_PATH.iterdir() if channel_path.is_dir())
    channels_list = list(set(datasets_list) & set(channels_list))
    # Debug
    # channels_list = ['CNNW']

    for channel_name in channels_list:
        print(f"{channel_name}")
        dataset = fo.load_dataset(channel_name)
        method_name = f"{DETECTOR}-{FEATS}-{CLASSIFIER}"
        res_path = ROOT_RES_PATH / Path(f"{channel_name}/train/{method_name}")
        add_res_to_fiftyone_samples(dataset, res_path, method_name)

    print('Success!')


if __name__ == "__main__":
    # Config
    parser = argparse.ArgumentParser(description='Populate fiftyone dataset')
    parser.add_argument('--detector', type=str, default='yolo', help='Face detector. MTCNN, DFSD, YOLO-face.')
    parser.add_argument('--feats', type=str, default='resnetv1', help='Feature extractor. ResnetV1.')
    parser.add_argument('--mod_feat', type=str, default='fcg_average_vote', help='Modify feature combination. single, min_dist, all_dist, mean_dist, min_plus_mean_dist, all_plus_mean_dist')
    args = parser.parse_args()

    ROOT_DATASET_PATH = Path(f"data/dataset/train")
    ROOT_RES_PATH = Path("data/results")
    DETECTOR = args.detector
    FEATS = args.feats
    CLASSIFIER = args.mod_feat

    main()
