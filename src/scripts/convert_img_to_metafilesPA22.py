# As we are not including images in the PA22 replication package, make a script to generate a list of images with their corresponding

from pathlib import Path
from PIL import Image

import os
import sys
sys.path.append(os.getcwd())


def frames_to_meta(channel_path):
    """
    in: year_path (e.g. .../train/CNNW/2011
    out: fiftyone-style samples
    """
    years_list = (year_path for year_path in channel_path.iterdir() if year_path.is_dir())

    frames_list = []
    for year_path in years_list:
        year_str = year_path.stem
        print(year_str)
        if channel_path.stem == 'news7-lv' and int(year_str) < 2013 and int(year_str) >= 2022:
            continue

        img_list = (img_path for img_path in year_path.iterdir() if img_path.suffix == '.jpg' or img_path.suffix == '.png' or img_path.suffix == '.jpeg')

        for img_path in img_list:
            img = Image.open(img_path)
            save_path = SAVE_DATASET_PATH / channel_path.stem / year_str / img_path.stem
            save_path = save_path.with_suffix('.metaPA22')
            with open(save_path, 'w') as w_file:
                row = f"{img.height},{img.width}"
                w_file.write(row)

    return frames_list


def main():
    channels_list = (channel_path for channel_path in ROOT_DATASET_PATH.iterdir() if channel_path.is_dir())
    for channel_path in channels_list:
        print(f"{channel_path}")
        # Return image + detections samples
        frames_to_meta(channel_path)

    print('Success')


if __name__ == "__main__":
    ROOT_DATASET_PATH = Path(f"/home/agirbau/work/politics/data/dataset/train")
    SAVE_DATASET_PATH = Path(f"/home/agirbau/work/PA22_replication/data/dataset/train")
    main()
