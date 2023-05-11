from collections import Counter
import csv

import copy
import cv2
import datetime
import json
import numpy as np
import os
from pathlib import Path

import pandas
import pandas as pd
from PIL import Image

# Japanese data
REPRESENTATIVES_DATES_MONTH = ['2003_11', '2005_09', '2009_08', '2012_12', '2014_12', '2017_10', '2021_10']
REPRESENTATIVES_DATES_DAY = ['2003_11_09', '2005_09_11', '2009_08_30', '2012_12_16', '2014_12_14', '2017_10_22', '2021_10_31']
PARLIAMENT_DISSOLUTION_DAY = ['2003_10_11', '2005_08_08', '2009_07_21', '2012_11_16', '2014_11_21', '2017_09_28', '2021_10_14']
PARL_CAMPAIGN_START_DAY = ['2003_10_28', '2005_08_30', '2009_08_18', '2012_12_04', '2014_12_02', '2017_10_10', '2021_10_19']

COUNCILLORS_DATES_MONTH = ['2001_07', '2004_07', '2007_07', '2010_07', '2013_07', '2016_07', '2019_07']
COUNCILLORS_DATES_DAY = ['2001_07_29', '2004_07_11', '2007_07_29', '2010_07_11', '2013_07_21', '2016_07_10', '2019_07_21']
COUNC_CAMPAIGN_START_DAY = ['2001_07_12', '2004_06_24', '2007_07_12', '2010_06_24', '2013_07_04', '2016_06_22', '2019_07_04']

LDP_DATES_MONTH = ['2001_04', '2006_09', '2007_09', '2008_09', '2009_10', '2012_10', '2020_09', '2021_10']
LDP_DATES_DAY = ['2001_04_24', '2006_09_20', '2007_09_23', '2008_09_22', '2009_09_28', '2012_09_26', '2020_09_14', '2021_09_29']

POLITICIANS = ['Akihiro_Ota', 'Ichiro_Ozawa', 'Junichiro_Koizumi', 'Katsuya_Okada', 'Kazuo_Shii', 'Mizuho_Fukushima',
               'Naoto_Kan', 'Natsuo_Yamaguchi', 'Sadakazu_Tanigaki', 'Seiji_Maehara', 'Shintaro_Ishihara', 'Shinzo_Abe',
               'Takako_Doi', 'Takenori_Kanzaki', 'Taro_Aso', 'Toru_Hashimoto', 'Yasuo_Fukuda', 'Yoshihiko_Noda',
               'Yoshimi_Watanabe', 'Yukio_Hatoyama',
               'Yukiko_Kada', 'Yuko_Mori', 'Taro_Yamamoto', 'Kenji_Eda', 'Yorihisa_Matsuno', 'Ichiro_Matsui',
               'Banri_Kaieda', 'Renho_Renho', 'Kohei_Otsuka', 'Yuichiro_Tamaki', 'Yukio_Edano', 'Takashi_Tachibana',
               'Shigefumi_Matsuzawa', 'Yoshihide_Suga', 'Kyoko_Nakayama', 'Masashi_Nakano', 'Takeo_Hiranuma',
               'Seiji_Mataichi', 'Tadatomo_Yoshida', 'Nariaki_Nakayama', 'Fumio_Kishida']

PRIME_MINISTERS = ['Junichiro_Koizumi', 'Junichiro_Koizumi', 'Junichiro_Koizumi', 'Shinzo_Abe', 'Yasuo_Fukuda',
                   'Taro_Aso', 'Yukio_Hatoyama', 'Naoto_Kan', 'Yoshihiko_Noda', 'Shinzo_Abe',
                   'Shinzo_Abe', 'Shinzo_Abe', 'Yoshihide_Suga', 'Fumio_Kishida']
PM_DATES_MONTH = ['2001_04', '2003_11', '2005_09', '2006_09', '2007_09',
                  '2008_09', '2009_08', '2010_06', '2011_09', '2012_12',
                  '2014_12', '2017_10', '2020_09', '2021_10']
PM_DATES_DAY = ['2001_04_26', '2003_11_09', '2005_09_11', '2006_09_26', '2007_09_26',
                '2008_09_24', '2009_08_30', '2010_06_08', '2011_09_02', '2012_12_16',
                '2014_12_14', '2017_10_22', '2020_09_16', '2021_10_04']

OP_LEADERS = ['Yukio_Hatoyama', 'Naoto_Kan', 'Katsuya_Okada', 'Seiji_Maehara', 'Ichiro_Ozawa',
              'Yukio_Hatoyama', 'Sadakazu_Tanigaki', 'Shinzo_Abe', 'Banri_Kaieda', 'Katsuya_Okada',
              'Renho_Renho', 'Seiji_Maehara', 'Yukio_Edano']
OP_DATES_MONTH = ['2001_05', '2002_12', '2004_08', '2005_09', '2006_04',
                  '2009_05', '2009_09', '2012_09', '2012_12', '2014_12',
                  '2016_10', '2017_09', '2017_10']
OP_DATES_DAY = ['2001_05_01', '2002_12_01', '2004_08_01', '2005_09_01', '2006_04_01',
                '2009_05_01', '2009_09_01', '2012_09_01', '2012_12_01', '2014_12_01',
                '2016_10_01', '2017_09_01', '2017_10_01']  # Temporal solution, check dates with Kob. sensei


COLOR_PALETTE_DICT = {'Junichiro_Koizumi': "#db162f", 'Shinzo_Abe': "#f58b0a", 'Yasuo_Fukuda': "#772d8b",  # PM
                      'Taro_Aso': "#23d10f", 'Yukio_Hatoyama': "#1e97fa", 'Naoto_Kan': "#ff99c9",
                      'Yoshihiko_Noda': "#3317bf", 'Yoshihide_Suga': "#92909e", 'Fumio_Kishida': "#3DAD82",

                      'Ichiro_Ozawa': "#99e9f0", 'Natsuo_Yamaguchi': "#F221A3", 'Kazuo_Shii': "#7321F2",  # OPP
                      'Banri_Kaieda': "#28A14E", 'Toru_Hashimoto': "#C18C92", 'Yukio_Edano': "#9A3520",
                      'Katsuya_Okada': "#6341F2", 'Renho_Renho': "#A1EF8B", "Seiji_Maehara": "#EF8275",
                      'Sadakazu_Tanigaki': "#cf9893",

                      "LDP": "#db162f", "DPJ": "#2AD6DE", 'CP': "#7321F2", 'PJK': "#C62ADE", 'SDP': "#4BC720",  # Party
                      'JIP': "#A2BF21",

                      "Akihiro_Ota": "#2f4025",  # Other pols

                      0: '#de5454', 1: '#3bb0d1', '.': "#FFFFFF"}  # Extra

PARTIES = ['Komeito', 'DPJ', 'LDP', 'JIP', 'DPP', 'CP', 'SDP', 'LP', 'Reiwa', 'UP', 'IO', 'CDP', 'NHK', 'PoH', 'PJK']
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
COLOR_PALETTE = ["#de5454", "#3bb0d1","#7283e0","#447604","#b87d4b","#a1ef8b","#cf9893","#ef8275","#CBB8A9","#e0afa0",
                 "#f99af4","#fa7b05","#9DB5B2","#103701","#0b22a3","#f10e83","#4158d9",
                 "#447604", "#92d5e6", "#b87d4b", "#a1ef8b", "#cf9893", "#ef8275", "#CBB8A9", "#e0afa0",
                 "#f99af4", "#fa7b05", "#9DB5B2", "#103701", "#0b22a3", "#f10e83", "#447604", "#92d5e6", "#b87d4b", "#a1ef8b", "#cf9893", "#ef8275", "#CBB8A9", "#e0afa0",
                 "#f99af4", "#fa7b05", "#9DB5B2", "#103701", "#0b22a3", "#f10e83", "#447604", "#92d5e6", "#b87d4b", "#a1ef8b", "#cf9893", "#ef8275", "#CBB8A9", "#e0afa0",
                 "#f99af4", "#fa7b05", "#9DB5B2", "#103701", "#0b22a3", "#f10e83", "#447604", "#92d5e6", "#b87d4b", "#a1ef8b", "#cf9893", "#ef8275", "#CBB8A9", "#e0afa0",
                 "#f99af4", "#fa7b05", "#9DB5B2", "#103701", "#0b22a3", "#f10e83", "#447604", "#92d5e6", "#b87d4b", "#a1ef8b", "#cf9893", "#ef8275", "#CBB8A9", "#e0afa0",
                 "#f99af4", "#fa7b05", "#9DB5B2", "#103701", "#0b22a3", "#f10e83"
                 ]

# US data
US_ACTORS = ['Amy_Klobuchar', 'Barack_Obama', 'Ben_Carson', 'Bernie_Sanders', "Beto_O'Rourke", 'Bill_Clinton',
             'Bill_De_Blasio', 'Bobby_Jindal', 'Carly_Fiorina', 'Chris_Christie', 'Dick_Durbin', 'Donald_Trump',
             'Elizabeth_Warren', 'Gary_Johnson', 'George_W_Bush', 'George_Zimmerman', 'Harry_Reid', 'Herman_Cain',
             'Hillary_Clinton', 'Jeb_Bush', 'Jim_Gilmore', 'Jim_Webb', 'Joe_Biden', 'John_Boehner', 'John_McCain',
             'Jon_Huntsman_Jr', 'Kamala_Harris', 'Kellyanne_Conway', 'Kevin_McCarthy', 'Lincoln_Chafee',
             'Lindsey_Graham', 'Marco_Rubio', "Martin_O'Malley", 'Michele_Bachmann', 'Michelle_Obama',
             'Mike_Huckabee', 'Mitch_McConnell', 'Mitt_Romney', 'Nancy_Pelosi', 'Newt_Gingrich', 'Orrin_Hatch',
             'Paul_Ryan', 'Pete_Buttigieg', 'Rand_Paul', 'Rick_Perry', 'Rick_Santorum', 'Ron_Paul', 'Sarah_Palin',
             'Steve_Scalise', 'Ted_Cruz', 'Tim_Kaine', 'Trayvon_Martin', 'Tulsi_Gabbard',
             'John_Kasich'  # Added (no GT)
             ]


US_COLOR_PALETTE_DICT = {'Donald_Trump': "#db162f", 'Barack_Obama': "#f58b0a", 'Hillary_Clinton': "#772d8b",  # PM
                        }


def read_channel_data(dataset_path, **kwargs):
    from_date = kwargs.get("from_date", convert_str_to_date("1900_01_01"))
    to_date = kwargs.get("from_date", convert_str_to_date("2100_01_01"))

    years_path = (year_path for year_path in dataset_path.iterdir() if year_path.is_file())

    df_list = []
    for dataset_year_path in years_path:
        year_str = dataset_year_path.stem

        files_annot = [f for f in dataset_year_path.iterdir() if f.suffix == '.csv']
        files_annot = filter_path_list_by_date(files_annot, from_date, to_date)

        for f in files_annot:
            df = read_gt_csv_data(f, year=year_str)
            df_list.append(df)

    df_res = pd.concat(df_list, ignore_index=True, axis=0)
    return df_res


def read_gt_csv_data(csv_path, **kwargs):
    # Function to read csv file for GT data
    year_str = kwargs.get('year', '-1')
    source = csv_path.stem.split('-')[0]
    frame = csv_path.stem.split('-')[1]

    df = pd.read_csv(csv_path)
    rows = df.values.tolist()

    row_list = []
    for row in rows:
        row_list.append([source, year_str, frame, row[0], row[1], row[2], row[3], row[4]])

    return pd.DataFrame(data=row_list, columns=['source', 'year', 'frame', 'ID', 'cx', 'cy', 'w', 'h'])


def crop_face_images(img, bboxes):
    # Gets a pillow-type image and the bounding boxes as numpy array and returns pillow images cropped to the bboxes
    faces_img = []
    for bbox in bboxes:
        faces_img.append(img.crop(bbox))

    return faces_img


def crop_resize(img, box, output_size, resize_ok=True, bbox_air=0):
    # https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/utils/detect_face.py#L309
    # box: [x1, y1, x2, y2]
    if isinstance(img, np.ndarray):
        p_top = int(max(0, box[1] - bbox_air))
        p_bot = int(min(box[3] + bbox_air, img.shape[0]))
        p_left = int(max(0, box[0] - bbox_air))
        p_right = int(min(box[2] + bbox_air, img.shape[1]))
        img = img[p_top:p_bot, p_left:p_right]
        if resize_ok:
            out = cv2.resize(
                img,
                (output_size, output_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        else:
            out = img.copy()
    else:
        if resize_ok:
            out = img.crop(box).copy().resize((output_size, output_size), Image.BILINEAR)
        else:
            out = img.crop(box).copy()

    return out


def embeddings_to_tensorboard(tensor_writer, embs, labs):
    # writer.add_embedding(embed)
    # labls = [str(t) for t in labs.numpy()]
    tensor_writer.add_embedding(embs, metadata=labs)


def embeddings_as_images_to_tensorboard(embs, labs, centroids=None):
    # Following: https://medium.com/@pylieva/how-to-visualize-image-feature-vectors-1e309d45f28f
    # "It is necessary to save the final image feature vectors in a tab-separated format."

    if not os.path.isdir('vis'):
        os.mkdir('vis')

    # 1. Save embeddings to csv
    embs_vec = [emb.detach().cpu().numpy() for emb in embs]
    with open('vis/feature_vecs.tsv', 'w') as w_feats:
        csv_writer = csv.writer(w_feats, delimiter='\t')
        if centroids is not None:
            # centroids is already a list
            csv_writer.writerows(centroids)

        csv_writer.writerows(embs_vec)

    # 2. Save metadata to csv
    with open('vis/metadata.tsv', 'w') as w_meta:
        csv_writer = csv.writer(w_meta, delimiter='\n')
        # w_meta.write('label\n')
        if centroids is not None:
            # centroids is already numpy array
            for i, _ in enumerate(centroids):
                w_meta.write('c_{}\n'.format(i))

        csv_writer.writerows(labs)


def create_image_sprite(images, num_centroids=0):
    images_pil = []
    for image in images:
        images_pil.append(image.resize((64, 64)))

    image_width, image_height = images_pil[0].size

    one_square_size = int(np.ceil(np.sqrt(len(images) + num_centroids)))
    master_width = image_width * one_square_size
    master_height = image_height * one_square_size

    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))  # fully transparent

    for count, img in enumerate(images_pil, start=num_centroids):
        div, mod = divmod(count, one_square_size)
        h_loc = image_width * div
        w_loc = image_width * mod
        spriteimage.paste(img, (w_loc, h_loc))

    spriteimage.convert("RGB").save('vis/sprite.jpg', transparency=0)


def bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    bbox_c_w = [cx, cy, w, h]
    return bbox_c_w


def bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox):
    x1 = bbox[0] - (bbox[2]/2)
    y1 = bbox[1] - (bbox[3]/2)
    x2 = bbox[0] + (bbox[2]/2)
    y2 = bbox[1] + (bbox[3]/2)

    bbox_x_y = [x1, y1, x2, y2]
    return bbox_x_y


def bbox_absolute_to_relative(bbox, img_size):
    # Get bbox (in whatever coordinates) and normalize it with respect to the image size
    # img_size = [w, h]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    img_w = img_size[0]
    img_h = img_size[1]
    bbox_rel = [x1/img_w, y1/img_h, x2/img_w, y2/img_h]

    return bbox_rel


def bbox_to_fiftyone_format(bbox, img_size):
    # Get bbox in (cx, cy, w, h) and return it in fiftyone format (x1, y1, w, h) relative to the image size
    # img_size = [w, h]
    cx = bbox[0]
    cy = bbox[1]
    w = bbox[2]
    h = bbox[3]
    x1 = cx - w/2
    y1 = cy - h/2
    bbox_x1y1wh = [x1, y1, w, h]

    bbox_res = bbox_absolute_to_relative(bbox_x1y1wh, img_size)
    return bbox_res


def compute_jaccard(a, b):
    # |-----|
    # |--|xx|---|
    # |--|xx|---|
    #    |------|
    # Compute jaccard. Input bboxes in the form of x1_y1_x2_y2.
    xa, ya, x2a, y2a = a[0], a[1], a[2], a[3]
    xb, yb, x2b, y2b = b[0], b[1], b[2], b[3]

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def compute_ious_bboxes(bboxes1, bboxes2):
    # Compute the IoU of a set of bounding bboxes1 vs. the set of bboxes2
    ious = np.zeros((len(bboxes1), len(bboxes2)))
    for idx1, box_1 in enumerate(bboxes1):
        for idx2, box_2 in enumerate(bboxes2):
            ious[idx1, idx2] = compute_jaccard(box_1, box_2)

    return ious


def cm_to_inch(value):
    return value/2.54


def modify_palette(idx_list):
    color_palette = [COLOR_PALETTE[idx] for idx in idx_list]
    return color_palette


def get_politician_palette(pol_list: list) -> list:
    color_palette = []
    for i, pol in enumerate(pol_list):
        if pol in COLOR_PALETTE_DICT.keys():
            color_palette.append(COLOR_PALETTE_DICT[pol])
        elif pol in US_COLOR_PALETTE_DICT.keys():
            color_palette.append(US_COLOR_PALETTE_DICT[pol])
        else:
            color_palette.append(COLOR_PALETTE[i])

    return color_palette


def set_vid_to_frame_with_offset(cap, frame_num, neg_offset=500):
    # OpenCV has a terrible bug where set gets (probably) to the closest I frame instead of frame_num
    pos = max(0, frame_num-neg_offset)
    # If the current frame is closest to frame_num do not set the frame position
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if pos > current_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def skip_frames_until(cap, frame_num):
    # Some videos do set correctly the frame with video[frame_num] or cap.set(2, frame_num), see https://github.com/opencv/opencv/issues/20227
    set_vid_to_frame_with_offset(cap, frame_num)
    while True:
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_num == current_frame:
            return 0
        elif frame_num < current_frame:
            # Remember that VideoCapture.set() doesn't set at the 1st frame --> close and reopen
            # It has to be reopened outside, if not weird things happen
            return -1
        else:
            success = cap.grab()
            if not success:
                return -2


def plot_bbox_label(bbox_xy, label, img, color, text_color=(0, 0, 0)):
    # Bounding box info
    x1 = int(bbox_xy[0])
    y1 = int(bbox_xy[1])
    x2 = int(bbox_xy[2])
    y2 = int(bbox_xy[3])
    # Take only surname for better visualisation
    label_plot = label.split('_')[-1]

    # Plot bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Plot label having a rectangle as background
    # From: https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label_plot, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Prints the text.
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label_plot, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


# def fixed_batch_image_standardization(image_batch_tensor):
#     for image_tensor in image_batch_tensor:
#         processed_tensor = fixed_image_standardization(image_tensor)


def fixed_image_standardization(image_tensor):
    # From MTCNN
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def count_models_images(gt_politicians_path: Path):
    img_suffixes = ['.jpeg', '.jpg', '.png']
    politicians_path = sorted([img for img in gt_politicians_path.iterdir()])
    pols_col = ['ID', 'instances']
    pols_row = []
    for model_path in politicians_path:
        frames_pol = [img.stem for img in model_path.iterdir() if img.suffix in img_suffixes]
        # Erase repeated instances when counting
        frames_pol = list(set(frames_pol))
        pols_row.append([model_path.stem, len(frames_pol)])

    df = pandas.DataFrame(data=pols_row, columns=pols_col).sort_values(by=['instances'], ascending=False)
    print(df)


def count_path_images(root_path: Path):
    # To count images present in things like "dataset/to_annotate/news7-lv/year/frames"
    img_suffixes = ['.jpeg', '.jpg', '.png']
    total_imgs = 0
    news_paths = [nn for nn in root_path.iterdir() if nn.is_dir()]
    for news_path in news_paths:
        years_paths = [yy for yy in news_path.iterdir() if yy.is_dir()]
        for year_path in years_paths:
            imgs_paths = [ii for ii in year_path.iterdir() if ii.suffix in img_suffixes]
            total_imgs += len(imgs_paths)

    print(f'Total images: {total_imgs}')

    return total_imgs


def choose_n_pols(df, num=0):
    # Function to test the influence of number of images of politicians in classification
    if num > 0:
        df_samples = df.groupby(["ID"]).sample(n=num)
    else:
        df_samples = df
    return df_samples


def get_gt_embeddings(face_detector, feat_extractor, gt_politicians_path):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import my_transforms

    from face_dataset import FaceDataset

    # Dataloader
    transform = transforms.Compose([
        my_transforms.ToTensor()
    ])

    face_dataset = FaceDataset(data_path=gt_politicians_path, transform=transform)
    face_dataloader = DataLoader(face_dataset, batch_size=2, shuffle=False, collate_fn=face_dataset.collate_fn)

    # Model already as a parameter
    # Face embeddings of the dataset per politician
    face_path_imgs = []
    face_imgs = []
    face_bboxes = []
    face_embeddings = torch.Tensor().to(face_detector.device)
    face_labels = []

    for i, batch in enumerate(face_dataloader):
        # TODO: Correctly pass several images as batch
        imgs_path, imgs, labels = batch[0], batch[1], batch[2]
        for img_path, img, label in zip(imgs_path, imgs, labels):
            img_pil = transforms.ToPILImage()(img).convert('RGB')
            bboxes, probs = face_detector.detect(img_pil)
            max_prob_idx = np.argmax(probs)  # In case there are 2 detections (Taro Aso in MTCNN)
            bboxes, probs = np.expand_dims(bboxes[max_prob_idx], axis=0), np.expand_dims(probs[max_prob_idx], axis=0)
            # Get face embeddings
            face_images_pil = crop_face_images(img_pil, bboxes)
            # Embeddings info
            face_path_imgs.append(img_path)
            [face_imgs.append(face_img_pil) for face_img_pil in face_images_pil]
            [face_bboxes.append((bbox, prob)) for bbox, prob in zip(bboxes, probs)]
            face_embs_run = feat_extractor.crop_and_get_embeddings(img_pil, bboxes)
            if isinstance(face_embs_run, np.ndarray):
                # For facenet (tensorflow returns numpy arrays)
                face_embs_run = torch.from_numpy(face_embs_run).to(face_embeddings.device)

            face_embeddings = torch.cat((face_embeddings, face_embs_run))
            face_labels.append(label)

    return face_path_imgs, face_imgs, face_bboxes, face_embeddings, face_labels


def add_padding(img, top, bot, left, right):
    # Add padding to an image
    # In case the input is a pillow image
    img_array = np.array(img)

    img_pad = cv2.copyMakeBorder(img_array, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Reconvert to pillow image
    if type(img) == Image.Image:
        img_pad = Image.fromarray(img_pad)

    return img_pad


def unpad_bbox(bbox, top, bot, left, right):
    # bbox: x1, y1, x2, y2
    bbox_unpad = copy.copy(bbox)
    bbox_unpad[0] -= left
    bbox_unpad[1] -= top
    bbox_unpad[2] -= right
    bbox_unpad[3] -= bot

    return bbox_unpad


def dict_to_df_politics(pol_dict):
    import pandas as pd

    df_dict = dict()
    df_dict['source'] = []
    df_dict['ID'] = []
    df_dict['cx'] = []
    df_dict['cy'] = []
    df_dict['w'] = []
    df_dict['h'] = []
    df_dict['prob'] = []
    df_dict['emb'] = []
    for k_pol in pol_dict.keys():
        for img_path, bbox_xy, prob, emb in zip(pol_dict[k_pol]['source'], pol_dict[k_pol]['bbox'], pol_dict[k_pol]['prob'], pol_dict[k_pol]['emb']):
            bbox_c_w = bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox_xy)
            cx = bbox_c_w[0]
            cy = bbox_c_w[1]
            w = bbox_c_w[2]
            h = bbox_c_w[3]
            df_dict['source'].append(img_path)
            df_dict['ID'].append(k_pol)
            df_dict['cx'].append(cx)
            df_dict['cy'].append(cy)
            df_dict['w'].append(w)
            df_dict['h'].append(h)
            df_dict['prob'].append(prob)
            df_dict['emb'].append(emb.detach().cpu().numpy())

    df = pd.DataFrame(data=df_dict)
    return df


def list_to_df_embeddings(frame_embs_list: list) -> pd.DataFrame:
    # Function to transform the frame-embeddings tuple to a dataframe
    import pandas as pd

    df_dict = dict()
    # source,ID,frame,cx,cy,w,h,prob_det,emb
    source_list = [f_e[0] for f_e in frame_embs_list]
    id_list = [f_e[1] for f_e in frame_embs_list]
    frame_list = [int(f_e[2]) for f_e in frame_embs_list]
    cx_list = [f_e[3] for f_e in frame_embs_list]
    cy_list = [f_e[4] for f_e in frame_embs_list]
    w_list = [f_e[5] for f_e in frame_embs_list]
    h_list = [f_e[6] for f_e in frame_embs_list]
    prob_list = [f_e[7] for f_e in frame_embs_list]
    emb_list = [f_e[8] for f_e in frame_embs_list]

    df_dict['source'] = source_list
    df_dict['ID'] = id_list
    df_dict['frame'] = frame_list
    df_dict['cx'] = cx_list
    df_dict['cy'] = cy_list
    df_dict['w'] = w_list
    df_dict['h'] = h_list
    df_dict['prob_det'] = prob_list
    df_dict['emb'] = emb_list
    df = pd.DataFrame(df_dict)

    return df


def read_rel_file(rel_file):
    with open(rel_file, 'r') as json_file:
        rel_dict = json.load(json_file)

    return rel_dict


def convert_str_to_date(date_str: str) -> datetime.datetime:
    date_split = list(map(int, date_str.split('_')))
    yy = 2000
    mm = 1
    dd = 1
    if len(date_split) >= 1:
        yy = date_split[0]
    if len(date_split) >= 2:
        mm = date_split[1]
    if len(date_split) >= 3:
        dd = date_split[2]

    return datetime.datetime(yy, mm, dd)


def filter_list_by_date(str_dates_list: list, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    dates_list = [convert_str_to_date(x.split('-')[0]) for x in str_dates_list]  # split for csv train files (after '-' there's the frame number)
    idx_dates = [from_date.date() <= dd.date() <= to_date.date() for dd in dates_list]
    return idx_dates


def filter_path_list_by_date(path_dates_list: list, from_date: datetime.datetime, to_date: datetime.datetime) -> list:
    # Filter which videos to process
    videos_year_day_path_name = [vv.stem for vv in path_dates_list]
    idx_videos = filter_list_by_date(videos_year_day_path_name, from_date, to_date)
    videos_year_day_path_list = [dd for dd, bb in zip(path_dates_list, idx_videos) if bb]
    return videos_year_day_path_list


def get_same_els_list(list1: list, list2: list) -> list:
    union_set = set(list1) & set(list2)
    idxs = [list1.index(pol) for pol in union_set]
    return idxs


def get_video_from_path(video_path: Path, ext='.mpg'):
    # This needs to exists due to mismatch between folder path and video name in hodost-lv
    videos_in_path = [video_path for video_path in video_path.iterdir() if video_path.suffix == ext]
    if len(videos_in_path) > 1:
        print(f'Care, found more than one video: {videos_in_path}, taking {videos_in_path[0]} (first by default).')

    return videos_in_path[0]


def get_path_from_video(video_path: Path, ext='.mpg'):
    # I messed up naming the source information inside the dataframe (used video instead of folder info). For now:
    # This needs to exists due to mismatch between folder path and video name in hodost-lv
    date_video = video_path.stem[:10]
    folder_in_path = [folder_path for folder_path in video_path.parent.iterdir() if folder_path.stem.find(date_video) > -1]
    if len(folder_in_path) > 1:
        print(f'Care, found more than one video: {folder_in_path}, taking {folder_in_path[0]} (first by default).')

    return folder_in_path[0]


def get_most_repeated_value(in_list: list) -> list:
    # If 2 or more values are the most repeated return everything
    # From https://moonbooks.org/Articles/How-to-find-the-element-of-a-list-with-the-maximum-of-repetitions-occurrences-in-python-/
    out_list = Counter(in_list).most_common()
    return out_list


def drop_df_rows(df, col_id, rows_drop):
    # e.g. drop df['ID'] == 'Overall'
    if not isinstance(rows_drop, list):
        rows_drop_list = [rows_drop]
    else:
        rows_drop_list = rows_drop

    df_res = df.copy()
    for row_drop in rows_drop_list:
        mask_drop = df[col_id] == row_drop
        df_res = df_res[~mask_drop]

    return df_res


def read_and_append_df_csv(df, csv_path):
    if os.path.isfile(csv_path):
        df_csv = pd.read_csv(csv_path)
        df_res = pd.concat([df_csv, df])
    else:
        df_res = df

    return df_res
