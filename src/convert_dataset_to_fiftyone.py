# Script to convert our datasets to fiftyone format, so we can latter check it with their interface
# Guide: https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html

from pathlib import Path
import fiftyone as fo

import os
import sys
sys.path.append(os.getcwd())

from src.utils import read_gt_csv_data, bbox_to_fiftyone_format


def populate_metadata(sample, height, width):
    # Extract image metadata using your preferred method
    metadata = {
        "height": height,
        "width": width
    }

    # Set the "ImageMetadata" field for the sample
    sample["metadata"] = fo.ImageMetadata(**metadata)


def data_to_fiftyone_samples(channel_path):
    """
    in: year_path (e.g. .../train/CNNW/2011
    out: fiftyone-style samples
    """
    years_list = (year_path for year_path in channel_path.iterdir() if year_path.is_dir())

    samples = []
    with fo.ProgressBar() as pb:
        for year_path in pb(years_list):
            year_str = year_path.stem
            img_list = (img_path for img_path in year_path.iterdir() if img_path.suffix == '.metaPA22')

            for img_path in img_list:
                sample = fo.Sample(filepath=img_path)
                with open(img_path, 'r') as r_file:
                    row = r_file.readline().split(',')
                    img_height = int(row[0])
                    img_width = int(row[1])

                populate_metadata(sample, img_height, img_width)

                csv_path = img_path.with_suffix('.csv')
                if csv_path.exists():


                    df = read_gt_csv_data(csv_path, year=year_str)
                    detections = []
                    for _, row in df.iterrows():
                        label_id = row['ID'].split('_')[-1]
                        # bbox given in cx, cy, w, h
                        # Sample (from fiftyone) accepts bbox relative to the image size
                        # Fiftyone bbox format: x1, y1, w, h
                        bbox_cxcywh = [row['cx'], row['cy'], row['w'], row['h']]
                        bbox_area = bbox_cxcywh[2] * bbox_cxcywh[3]
                        bbox_rel = bbox_to_fiftyone_format(bbox_cxcywh, img_size=[img_width, img_height])
                        detections.append(
                            fo.Detection(label=row['ID'], bounding_box=bbox_rel,
                                         label_id=label_id, area=bbox_area)  #label_id is a custom attribute
                        )

                    sample["ground_truth"] = fo.Detections(detections=detections)
                sample["year"] = year_str
                samples.append(sample)

    return samples


def main():
    channels_list = (channel_path for channel_path in ROOT_DATASET_PATH.iterdir() if channel_path.is_dir())
    for channel_path in channels_list:
        print(f"{channel_path}")
        channel_name = channel_path.stem
        # Return image + detections samples
        samples = data_to_fiftyone_samples(channel_path)
        # Generate dataset based on channel name and samples. Can we somehow filter by year / date in fiftyone?
        dataset = fo.Dataset(name=f"{channel_name}", persistent=True, overwrite=True)
        dataset.add_samples(samples)

    print('Success')


if __name__ == "__main__":
    ROOT_DATASET_PATH = Path(f"data/dataset/train")
    main()
