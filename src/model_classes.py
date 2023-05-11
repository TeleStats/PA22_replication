# This file contains the class FaceDetector and the models we want to use
import copy
import sys

import cv2
from KDEpy import TreeKDE, NaiveKDE, FFTKDE
import pandas as pd
from facenet_pytorch import MTCNN
import face_detection
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.neighbors import KernelDensity
from scipy import stats
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.nn.functional import normalize as th_norm

#For FaceNet
import os
import re
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto(device_count={'GPU': 0})
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
from tensorflow.python.platform import gfile

from config import get_parser
from external import model_yolo5
from utils import get_gt_embeddings
from utils import bbox_x1_y1_x2_y2_to_cx_cy_w_h, crop_resize, fixed_image_standardization, add_padding, unpad_bbox


class FaceDetectorBase:
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        self.device = device
        self.feat_extractor = None
        self.det_path = det_path  # In case we use pre-computed detections
        self.results_path = res_path
        # Video info
        self.video_name = video_name
        self.video_fps = video_fps
        self.video_writer = None
        # Politician models
        self.model_detection = None
        self.model_gender = None
        self.video_dets_df = None
        # Other variables
        self.gender_transforms = None
        self.embs_list = []
        self.faces_frame = None
        # Configuration file arguments
        self.args = None
        self._parse_args_()
        # Initialize face detector and feature extraction models
        self._init_detection_model_()
        # Initialise dataframe for pre-computed detections
        if det_path:
            self._init_df_dets_()
        # For backwards compatibility be able to set feature extractor from here
        if kwargs.get('gt_politicians_path', False):
            gt_politicians_path = kwargs.get('gt_politicians_path')
            feat_extractor = kwargs.get('feat_extractor', 'resnetv1')
            input_size = kwargs.get('input_size', 160)
            if feat_extractor == 'facenet':
                feat_extractor_inst = FeatureExtractorTF(gt_politicians_path, feat_extractor, input_size, device=self.device)
            else:
                feat_extractor_inst = FeatureExtractor(gt_politicians_path, feat_extractor, input_size, device=self.device)

            self.set_feat_extractor(feat_extractor_inst)

    def set_info(self, res_path='data/results', video_name='demo', video_fps=29.97, det_path=None, device=None):
        if device is not None:
            self.device = device

        self.results_path = res_path
        self.video_name = video_name
        self.video_fps = video_fps
        self.det_path = det_path

        if det_path:
            self._init_df_dets_()

    def set_feat_extractor(self, feat_extractor):
        self.feat_extractor = feat_extractor

    def _parse_args_(self):
        parser = get_parser()
        self.args = parser.parse_args()

    def _init_detection_model_(self):
        # Define the models detector and feature extraction
        print('Build _init_model_ function for your method')
        pass

    def _init_video_writer_(self, img):
        img_h, img_w = img.shape[0], img.shape[1]
        write_path = (Path(self.results_path) / self.video_name).with_suffix('.mp4')
        self.video_writer = cv2.VideoWriter(str(write_path), cv2.VideoWriter_fourcc(*'MP4V'), self.video_fps, (img_w, img_h))

    def _init_df_dets_(self):
        self.video_dets_df = pd.read_csv(self.det_path)

    def detect(self, img):
        # Input: image in the form of pillow Image
        # Output: bounding boxes and probabilties of the detections
        print('Build detect function for your method')
        return -1, -1

    def crop_and_get_embeddings(self, img, bboxes):
        # Input: image and bounding boxes
        # Output: embeddings per image
        face_embeddings = self.feat_extractor.crop_and_get_embeddings(img, bboxes)
        return face_embeddings

    def get_embeddings(self, face_imgs):
        # Input: batch of cropped face images
        # Output: embeddings per image
        face_embeddings = self.feat_extractor.get_embeddings(face_imgs)
        return face_embeddings

    @staticmethod
    def draw_detections(img, bboxes):
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        for i, bbox in enumerate(bboxes):
            draw.rectangle(bbox.tolist(), width=5)

        img_draw.show()

    @staticmethod
    def draw_detections_cv2(img, bboxes, labels):
        color = (255, 255, 255)
        for bbox, label in zip(bboxes, labels):
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)

        cv2.imshow('Faces', img)
        cv2.waitKey(1)

    def write_in_video(self, img, bboxes, labels, dists, frame_id=-1):
        if self.video_writer is None:
            self._init_video_writer_(img)

        color = (255, 255, 255)
        for bbox, label, dist in zip(bboxes, labels, dists):
            if sum(bbox) == 0:  # bbox = [0, 0, 0, 0]
                continue

            if self.args.gender:
                dist_str = str(round(dist * 100, 1))
                if label == 'female':
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                elif label == 'male':
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                else:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(img, dist_str, (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            else:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(img, label, (int(bbox[0])+5, int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            if frame_id > -1:
                cv2.putText(img, str(frame_id), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        self.video_writer.write(img)

    def concat_embs_to_extract(self, res_file, frame_ID, bboxes, probs, embs):
        for bbox, prob_det, emb in zip(bboxes, probs, embs.cpu().numpy()):
            bbox_c_w = bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox)
            res_row_list = [
                res_file.stem,
                '-1',
                str(int(frame_ID)),
                str(int(bbox_c_w[0])),
                str(int(bbox_c_w[1])),
                str(int(bbox_c_w[2])),
                str(int(bbox_c_w[3])),
                str(prob_det)[:4],  # 2 decimals
                emb
            ]

            self.embs_list.append(res_row_list)

    def detect_and_write_to_file(self, frame, frame_ID, w_res_file, res_file, **kwargs):
        # kwargs variables (flags)
        flag_only_detections = kwargs.get('flag_only_detections', False)
        flag_extract_embs = kwargs.get('flag_extract_embs', False)
        flag_save_vid = kwargs.get('flag_save_vid', False)

        # Input: frame and file writer (already opened)
        if type(frame) != np.ndarray:
            # Video has errors or something
            return

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.det_path is None:
            bboxes, probs = self.detect(img)
        else:
            df = self.video_dets_df[self.video_dets_df['frame'] == frame_ID]
            x1s = (df['cx'] - df['w']/2).to_numpy()
            y1s = (df['cy'] - df['h']/2).to_numpy()
            x2s = (df['cx'] + df['w']/2).to_numpy()
            y2s = (df['cy'] + df['h']/2).to_numpy()
            probs = df['prob_det'].to_numpy()
            bboxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)]
            probs = [prob for prob in probs]

        # For now, no labels
        if bboxes is not None:
            if flag_only_detections:
                # Fill elements of the csv file
                labels = ['0' for _ in bboxes]
                emb_dists = [-1 for _ in bboxes]
            else:
                if self.args.gender:
                    # To reuse fields
                    # labels: gender (male / female)
                    # emb_dists = prob_gender (from softmax)
                    embs_gender, probs_gender = self.feat_extractor.crop_and_get_gender(img, bboxes)
                    if flag_extract_embs and embs_gender is not None:
                        self.concat_embs_to_extract(res_file, frame_ID, bboxes, probs, embs_gender)
                        return

                    labels, emb_dists = self.feat_extractor.assign_gender(probs_gender, prob_thresh=0.7)
                else:
                    # Get face embeddings
                    embs_face = self.crop_and_get_embeddings(img, bboxes)
                    if isinstance(embs_face, np.ndarray):
                        # For tensorflow (returns numpy)
                        embs_face = torch.from_numpy(embs_face).to(self.device)

                    if flag_extract_embs and embs_face is not None:
                        self.concat_embs_to_extract(res_file, frame_ID, bboxes, probs, embs_face)
                        return

                    labels, emb_dists = self.feat_extractor.assign_label(embs_face)
                    # labels = [str(i) for i in range(len(bboxes))]
        else:
            bboxes = [[0, 0, 0, 0]]
            probs = [0]
            labels = ['0']
            emb_dists = [-1]

        # res_row = ','.join(['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID']) + '\n'
        flag_written = False
        for bbox, label, prob_det, emb_dist in zip(bboxes, labels, probs, emb_dists):
            if label == '0' and not flag_only_detections:  # Ignore other detections
                continue

            bbox_c_w = bbox_x1_y1_x2_y2_to_cx_cy_w_h(bbox)
            res_row_list = [
                res_file.stem,
                label,
                str(frame_ID),
                str(int(bbox_c_w[0])),
                str(int(bbox_c_w[1])),
                str(int(bbox_c_w[2])),
                str(int(bbox_c_w[3])),
                str(prob_det)[:4],  # 2 decimals
                str(emb_dist)[:4]
            ]
            res_row = ','.join(res_row_list) + '\n'
            w_res_file.write(res_row)
            flag_written = True

        # To have the information on which frames are visited, if there is no politician detected, write a line indicating the frame
        if not flag_written:
            res_row_list = [
                res_file.stem,
                '0',
                str(frame_ID),
                str(0),
                str(0),
                str(0),
                str(0),
                str(0),
                '-1'
            ]
            res_row = ','.join(res_row_list) + '\n'
            w_res_file.write(res_row)

        if flag_save_vid:
            self.write_in_video(frame, bboxes, labels, dists=emb_dists, frame_id=frame_ID)


class FeatureExtractor:
    def __init__(self, gt_politicians_path, feat_extractor='resnetv1', feat_input_size=112, device='cpu', **kwargs):
        # Input arguments
        self.feat_extractor_name = feat_extractor
        self.feat_input_size = feat_input_size
        self.device = device
        # Other arguments
        # These arguments are for online computation
        self.mod_feat = kwargs.get('mod_feat', 'knn_1')
        self.dist = kwargs.get('dist', 0.3)
        self.logprob = kwargs.get('logprob', 1700)
        self.gender = kwargs.get('gender', False)
        # Variables
        self.model = None
        self.politician_dict = None
        # Initialize variables
        # Dictionary for politician models
        self._init_politician_dict_()
        # Feature extractor to use
        self._init_feats_model_()
        # For backwards compatibility for extracting face embeddings of politicians we need a face detector
        if kwargs.get('face_detector', False):
            self.face_detector = kwargs['face_detector']
        # GT embeddings
        self._init_gt_embeddings_(gt_politicians_path)
        # In case of classifying gender
        if self.gender:
            self._init_gender_model_()

    def _init_politician_dict_(self):
        self.politician_dict = dict()

    def _init_feats_model_(self):
        # Feature extractor model
        if self.feat_extractor_name == 'resnetv1':
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        elif self.feat_extractor_name == 'resnetv1-casio':
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='casia-webface', device=self.device).eval()
        elif self.feat_extractor_name == 'magface':
            from collections import namedtuple
            from external.model_magface import builder_inf
            args_magface = {
                'arch': 'iresnet100',
                'embedding_size': 512,
                'cpu_mode': True if self.device == 'cpu' else False,
                'device': self.device,
                'resume': 'src/external/magface_epoch_00025.pth'  # Make sure to have this inside "external"
            }
            # Dictionary to object
            # https://www.kite.com/python/answers/how-to-convert-a-dictionary-into-an-object-in-python
            args_magface_object = namedtuple("MagFaceArgs", args_magface.keys())(*args_magface.values())
            self.model = builder_inf(args_magface_object)
            self.model.to(self.device).eval()

    def _init_gender_model_(self):
        # As a test we put it here (for gender detection), but provides both face detection and gender class
        import insightface
        from insightface.app import FaceAnalysis
        self.model_gender = FaceAnalysis()
        self.model_gender.prepare(ctx_id=0, det_size=(640, 640))

        # from torchvision import transforms
        # from external.model_gender import ModelGen
        # self.model_gender = ModelGen(self.device)
        # self.gender_transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def initialize(self):
        pass

    def _init_gt_embeddings_(self, gt_politicians_path):
        self.initialize()  # For tensorflow model initialization
        gt_imgs_path, gt_imgs, gt_dets, gt_embs, gt_labs = get_gt_embeddings(self.face_detector, self, gt_politicians_path)
        for img_path, img, det, embedding, label in zip(gt_imgs_path, gt_imgs, gt_dets, gt_embs, gt_labs):
            if label not in self.politician_dict.keys():
                self.politician_dict[label] = dict()
                self.politician_dict[label]['source'] = []
                self.politician_dict[label]['bbox'] = []
                self.politician_dict[label]['prob'] = []
                self.politician_dict[label]['emb'] = []

            self.politician_dict[label]['source'].append(img_path)
            self.politician_dict[label]['bbox'].append(det[0])
            self.politician_dict[label]['prob'].append(det[1])
            self.politician_dict[label]['emb'].append(embedding)

        if self.mod_feat.find('kde') > -1:
            self.generate_pdf_from_embeddings()

        # For unique values
        for label in set(gt_labs):
            self.politician_dict[label]['emb'] = torch.stack(self.politician_dict[label]['emb'])
            if self.mod_feat == 'mean_dist':
                self.politician_dict[label]['emb'] = torch.mean(self.politician_dict[label]['emb'], dim=0).unsqueeze(0)
            elif self.mod_feat == 'min_plus_mean_dist' or self.mod_feat == 'all_plus_mean_dist' or self.mod_feat == 'min_plus_mean_nowhite_dist':
                mean_emb = torch.mean(self.politician_dict[label]['emb'], dim=0).unsqueeze(0)
                self.politician_dict[label]['emb'] = torch.cat((self.politician_dict[label]['emb'], mean_emb))

    def generate_pdf_from_embeddings(self):
        # Test array
        # x_b = np.expand_dims(np.array([0.04598,-0.01434,-0.04170,-0.00388,0.00516,0.01146,-0.06772,0.04547,-0.06075,0.01192,0.00988,
        #                 0.04575,-0.01053,-0.00378,0.00498,0.01131,0.03293,0.05699,-0.01257,-0.00259,0.06708,0.02313,
        #                 -0.00991,0.00529,0.01697,-0.02763,0.06188,-0.00333,0.00000,-0.00458,0.09747,-0.00076,0.07348,
        #                 0.09768,-0.02773,-0.08969,-0.07247,-0.04555,0.00173,0.01663,-0.02292,-0.06048,-0.02358,0.02946,-0.01200,0.05976,-0.00956,-0.02871,-0.02539,0.04234,-0.01469,-0.00863,0.05232,-0.04593,0.01952,-0.00403,0.00497,-0.05733,0.01230,-0.03458,-0.06185,0.10617,-0.02310,0.06127,0.03762,-0.03293,-0.04541,-0.03845,0.00801,-0.03103,-0.00155,-0.00543,-0.06162,0.06786,0.04129,0.09439,-0.01672,-0.04386,-0.00278,-0.05379,0.06226,-0.00459,-0.00372,-0.00523,0.03726,-0.02707,0.03109,0.03578,-0.00185,-0.06789,-0.06601,-0.03314,0.00568,0.02009,0.01869,0.01265,-0.08361,-0.00462,0.00626,0.02882,0.08571,0.10864,0.04981,0.01569,0.02070,-0.00756,-0.03908,-0.07481,0.00950,-0.03847,0.01343,0.01637,-0.06341,-0.04994,-0.00223,0.06294,-0.03526,0.04754,0.03036,0.08821,0.02597,0.00557,-0.04676,-0.03105,-0.05383,0.03978,0.08275,-0.06888,0.06416,-0.08379,0.04215,0.00886,-0.05403,-0.02121,0.02166,-0.01263,-0.01164,-0.05916,-0.02680,-0.03611,-0.00003,0.09407,-0.06547,-0.07444,0.08963,-0.04052,-0.01829,0.02168,0.02561,-0.01060,-0.03062,0.00449,0.03290,0.03966,0.04068,-0.00252,-0.05280,-0.02655,0.01589,-0.09498,-0.05593,0.05503,-0.01200,0.06394,0.04396,0.01922,-0.02283,0.10796,-0.10594,-0.01663,-0.00598,0.01875,-0.01865,0.02979,-0.00918,0.05365,-0.08501,0.03288,-0.05395,0.06363,0.06577,-0.07546,0.05435,0.05600,0.10571,-0.02334,0.01624,0.02392,0.01161,-0.01621,0.00381,0.02159,0.08514,-0.04932,0.05406,-0.01661,-0.00948,0.01216,0.03495,0.02331,0.05876,-0.02533,0.07950,-0.02729,0.02235,-0.01501,-0.01192,0.00962,0.01842,0.03038,0.03284,0.06907,-0.04047,0.06477,-0.01849,0.00262,0.03060,-0.01594,0.04578,-0.07763,-0.01835,0.07469,0.03230,0.09480,0.07218,-0.06345,-0.02840,-0.03579,-0.00062,-0.00617,-0.02645,-0.02451,0.05569,-0.03584,0.01039,-0.03858,0.01202,0.02827,0.05660,0.00694,0.01522,0.01427,0.01176,-0.01271,0.00082,0.07133,-0.00836,-0.01879,0.00180,0.03033,-0.05864,-0.01011,0.07160,0.02224,-0.00043,-0.05668,0.05120,0.05133,0.02553,0.01010,-0.05987,-0.03373,0.01097,-0.04181,-0.04414,-0.08761,0.07071,-0.02826,0.02170,-0.04997,0.05059,0.02912,0.03703,-0.07560,-0.00012,0.05904,-0.00258,-0.02461,-0.02442,0.01767,0.02128,-0.07534,-0.01657,-0.02068,0.05802,0.07449,0.04227,-0.12291,-0.01214,0.00819,-0.02213,0.00607,0.01894,-0.01486,-0.01264,-0.04375,-0.00508,-0.04654,0.08143,0.03305,0.08892,-0.04505,-0.02573,0.01799,0.00919,0.00692,-0.01906,0.00459,-0.06904,-0.02015,-0.07995,-0.01703,0.01289,0.03081,-0.01176,-0.00240,-0.01547,-0.02911,0.11569,0.00652,0.02915,-0.06703,-0.05777,-0.10350,0.00079,-0.00438,0.07482,0.00396,0.03095,0.00230,-0.00758,0.02870,-0.06807,0.02643,-0.00686,0.07614,-0.02116,-0.08116,-0.09558,0.02516,-0.02314,-0.01826,0.05816,0.02890,0.01579,0.05546,0.04390,0.04603,0.01833,0.06188,-0.00350,-0.04052,-0.01746,0.04933,0.04331,0.00498,-0.01383,-0.05838,-0.05952,-0.01254,-0.01834,-0.10695,0.00624,0.01198,0.03144,0.05116,-0.00984,0.04345,-0.01364,0.01569,0.00893,0.00286,-0.00357,-0.05330,-0.00608,-0.03015,-0.06247,0.00342,-0.00678,0.02013,-0.05373,-0.03363,-0.03734,0.00548,-0.03568,-0.02973,0.06205,-0.00714,0.01175,0.00700,-0.09670,0.04985,-0.00009,-0.02273,0.03099,0.06474,-0.04879,0.06205,0.06063,-0.06143,0.05610,0.01053,0.09955,-0.02065,0.03355,-0.00544,-0.03039,0.05905,-0.02759,-0.05420,0.03108,-0.04621,0.05917,0.02458,-0.05640,-0.05089,0.10956,0.00669,0.01906,-0.00045,-0.02426,0.00829,-0.03444,0.03294,-0.02383,-0.05610,-0.00647,0.01126,0.04646,-0.05020,-0.01173,-0.01559,-0.06630,-0.01473,0.05760,-0.00880,-0.00663,-0.04079,0.01604,-0.03159,-0.02474,-0.04894,-0.02301,0.00516,0.02617,-0.00984,0.01241,0.02292,-0.04226,0.00842,0.00504,0.04166,0.06421,0.05635,-0.04279,-0.00049,-0.04512,0.01719,-0.00378,0.12290,0.00219,0.04536,0.04282,-0.03159,0.07118,0.01850,-0.02565,-0.03348,0.03195,0.03424,0.03710,0.00753,0.02008,0.01258,-0.10418,-0.04554,-0.00074,-0.05082,0.05404,-0.02493,0.04183,-0.06883,-0.02805,0.01559,0.00728,-0.05783,-0.01307,0.00423,0.04833,-0.01779,-0.00443,-0.07417,0.09432,0.04477,-0.00993,-0.08437,0.05803,-0.04157,0.03979,-0.01495,0.00788,0.01594,-0.01077,-0.04075,-0.04649,0.02297,0.00382,0.06503,-0.06412,-0.03839,0.04383,-0.01174
        #                 ]), axis=0)
        for k_pol in self.politician_dict.keys():
            X = np.array([emb.detach().cpu().numpy() for emb in self.politician_dict[k_pol]['emb']])
            self.politician_dict[k_pol]['pdf'] = KernelDensity(bandwidth=0.05, kernel='gaussian').fit(X)
            # self.politician_dict[k_pol]['pdf'] = stats.gaussian_kde(X.T)
            # self.politician_dict[k_pol]['pdf'] = NaiveKDE(bw=0.05, kernel='gaussian').fit(X)
            # if k_pol == 'Yukio_Hatoyama':
            #     aa = self.politician_dict[k_pol]['pdf'].score_samples(x_b)
            #     # aa = self.politician_dict[k_pol]['pdf'].evaluate(x_b)

    def crop_images(self, img, bboxes, output_size=160, tolerance=0):
        # Crop face images and put them into a torch tensor (not normalized)
        bboxes_int = [[int(bbox[0] - (tolerance / 2)), int(bbox[1] - (tolerance / 2)),
                       int(bbox[2] + (tolerance / 2)), int(bbox[3] + (tolerance / 2))] for bbox in bboxes]
        # TODO: Check if it improves when changing the input size to 64x64 (right now is 160x160)
        face_imgs = [crop_resize(img, bbox, output_size) for bbox in bboxes_int]
        return face_imgs

    def crop_and_batch(self, img, bboxes, output_size=160, tolerance=0, opt_norm=False):
        # Crop face images and put them into a torch tensor (not normalized)
        face_imgs = self.crop_images(img, bboxes, output_size, tolerance)
        if opt_norm:
            face_imgs_th = torch.stack([F.to_tensor(face_img) for face_img in face_imgs])
        else:
            face_imgs_th = torch.stack([F.to_tensor(np.float32(face_img)) for face_img in face_imgs])

        return face_imgs_th

    def assign_label(self, det_embeddings):
        # The input are the embeddings from detections. Assign a label (or not) to a detection
        # I want to check ASAP the results, but do this with a distance matrix in the future, and that second loop is horrible
        labels = ['0'] * det_embeddings.shape[0]  # List creation in Python, lol
        emb_dist_list = [100] * det_embeddings.shape[0]
        for idx, emb in enumerate(det_embeddings):
            min_dist = 100
            res_emb_dist = 100
            min_logprob = 970
            max_logprob = 1000
            k_pol_adapt = None
            for k_pol in self.politician_dict.keys():
                # Computing the distance (equivalent to 1-similarity)
                # If cosine similarity = 0, vectors are orthogonal
                # https://stackoverflow.com/questions/58381092/difference-between-cosine-similarity-and-cosine-distance
                emb_dist = 1 - torch.cosine_similarity(self.politician_dict[k_pol]['emb'], emb.unsqueeze(0))
                # Here put the combination of features using different approaches
                if self.mod_feat.find('knn') > -1:
                    # e.g. knn_1_adapt
                    # Get minimum between the specified K and the amount of embeddings per politician
                    k_opt = self.mod_feat.split('_')[1]
                    if k_opt == 'max':
                        k_neighs = len(self.politician_dict[k_pol]['emb'])
                    else:
                        k_neighs = min(int(k_opt), len(self.politician_dict[k_pol]['emb']))
                    num_close_elems = sum(emb_dist < self.dist)  # Check all the distances lower than the threshold
                    if num_close_elems >= k_neighs:
                        res_emb_dist = torch.mean(
                            emb_dist)  # 2 neighbors with 0.2 and 0.2 should be closer to 0.15 0.3 (?)
                    else:
                        res_emb_dist = 100

                elif self.mod_feat.find('kde') > -1:
                    emb_numpy = emb.unsqueeze(0).detach().cpu().numpy()
                    logprob = self.politician_dict[k_pol]['pdf'].score_samples(emb_numpy)  # sklearn
                    if logprob >= min_logprob and logprob >= self.logprob:
                        # print(f'{k_pol}: {logprob}')
                        # print(f'{k_pol}: {min(emb_dist)}')
                        min_logprob = logprob
                        labels[idx] = k_pol
                        emb_dist_list[idx] = -min_logprob.item()
                        # if self.mod_feat.find('adapt') > -1 and logprob >= max_logprob:
                        if self.mod_feat.find('adapt') > -1 and min_logprob >= max_logprob:
                            # Add point to KDE
                            max_logprob = min_logprob
                            k_pol_adapt = k_pol
                            # pol_embs = np.array([emb.detach().cpu().numpy() for emb in self.politician_dict[k_pol]['emb']])
                            # X = np.vstack([pol_embs, emb_numpy])
                            # self.politician_dict[k_pol]['pdf'].fit(X)

                    # likelihood = self.politician_dict[k_pol]['pdf'].evaluate(emb_numpy)  # KDEpy
                    # print(f'{k_pol}: {likelihood}')
                    # print(f'{k_pol}: {min(emb_dist)}')
                    # res_emb_dist = min(emb_dist)

                if res_emb_dist < min_dist and res_emb_dist < 0.3:
                    min_dist = res_emb_dist
                    labels[idx] = k_pol
                    emb_dist_list[idx] = res_emb_dist.item()
                    if self.mod_feat.find('adapt') > -1 and min_dist < 0.15:
                        k_pol_adapt = copy.copy(k_pol)

            # Add embedding to politician model
            if k_pol_adapt is not None:
                self.politician_dict[k_pol_adapt]['emb'] = torch.cat(
                    [self.politician_dict[k_pol_adapt]['emb'], emb.unsqueeze(0)])
                if self.mod_feat.find('kde') > -1:
                    X = self.politician_dict[k_pol_adapt]['emb'].detach().cpu().numpy()
                    self.politician_dict[k_pol_adapt]['pdf'].fit(X)

        return labels, emb_dist_list

    def assign_gender(self, probs_gender, prob_thresh=0.8):
        labels = []
        for p in probs_gender:
            if p[0] >= prob_thresh:
                labels.append('female')
            elif p[1] >= prob_thresh:
                labels.append('male')
            else:
                labels.append('unknown')

        prob = [p[0] if p[0] > 0.5 else p[1] for p in probs_gender]
        return labels, prob

    def crop_and_get_gender(self, img, bboxes):
        # Input: image and bounding boxes
        # Output: probability of male / female
        probs_gender = []
        for i, _ in enumerate(bboxes):
            # Male: 1, Female: 0
            # This patch is for quick testing
            prob_male = self.face_detector.faces_frame[i].gender
            probs_gender.append((1 - prob_male, prob_male))

        return -1, probs_gender

    def crop_and_get_gender_old(self, img, bboxes):
        # Input: image and bounding boxes
        # Output: probability of male / female
        # Left here to test RGB vs. BGR inputs
        img_in = np.array(img)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)  # Input has to be BGR (coming image is RGB)
        # img_in = img_in[:, :, ::-1]
        img_in = Image.fromarray(img_in)
        face_imgs = self.crop_images(img_in, bboxes, 64, tolerance=3)  # With 0 tolerance works worse
        face_imgs_th = torch.stack([self.gender_transforms(face_img) for face_img in face_imgs]).to(self.device)
        embs, prob_gender = self.model_gender(face_imgs_th)
        return embs, prob_gender

    def crop_and_get_embeddings(self, img, bboxes):
        if self.feat_extractor_name == 'magface':
            # Magface has BGR inputs
            img_in = np.array(img)
            img_in = img_in[:, :, ::-1]
            img_in = Image.fromarray(img_in)
        else:
            img_in = img
        # Get cropped and "normalized" image tensor
        opt_norm = True if self.feat_extractor_name == 'magface' else False
        face_imgs_th = self.crop_and_batch(img_in, bboxes, output_size=self.feat_input_size, opt_norm=opt_norm)

        if self.feat_extractor_name == 'magface':
            face_imgs = F.normalize(face_imgs_th, mean=[0., 0., 0.], std=[1., 1., 1.])
        else:
            face_imgs = fixed_image_standardization(face_imgs_th)

        img_embedding = self.get_embeddings(face_imgs)
        return img_embedding

    @torch.inference_mode()
    def get_embeddings(self, face_imgs):
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = self.model(face_imgs.to(self.device))
        if self.feat_extractor_name == 'magface':
            # Normalize feature vector w.r.t the magnitude
            img_embedding = th_norm(img_embedding)

        return img_embedding


class FaceDetectorMTCNN(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self, input_size=160):
        # Detection model
        self.model_detection = MTCNN(image_size=input_size, margin=0, min_face_size=20, keep_all=True, device=self.device).eval()  # keep_all --> returns all bboxes in video

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        boxes, probs = self.model_detection.detect(img)
        return boxes, probs


class FaceDetectorDFSD(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self):
        # Detection model
        self.model_detection = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device=self.device)

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        boxes_probs = self.model_detection.detect(open_cv_image)
        boxes = boxes_probs[:, :4]
        probs = boxes_probs[:, -1]
        if len(boxes) == 0:
            boxes = None
            probs = None
        return boxes, probs


class FaceDetectorYOLO5(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self):
        # Detection model
        self.model_detection = model_yolo5.YOLO5FACE(conf_thresh=.5, nms_iou_thresh=.3, device=self.device)

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        open_cv_image = np.array(img)
        boxes_probs, _ = self.model_detection.detect(open_cv_image)

        if len(boxes_probs) == 0:
            boxes = None
            probs = None
        else:
            boxes = boxes_probs[:, :4]
            probs = boxes_probs[:, -1]

        return boxes, probs


class FaceDetectorInsightFace(FaceDetectorBase):
    def __init__(self, det_path=None, res_path='data/results', video_name='demo', video_fps=29.97, device='cpu', **kwargs):
        FaceDetectorBase.__init__(self, det_path, res_path, video_name, video_fps, device, **kwargs)

    @torch.no_grad()
    def _init_detection_model_(self, input_size=640):
        # Detection model
        import insightface
        from insightface.app import FaceAnalysis
        self.input_size = input_size
        self.model_detection = FaceAnalysis()
        self.model_detection.prepare(ctx_id=0, det_size=(input_size, input_size))

    def detect(self, img):
        # Get cropped and prewhitened image tensor
        open_cv_image = np.array(img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)  # Input has to be BGR (coming image is RGB)
        # Apply padding if the image is small (to not to rescale the face too much) --> for images from politicians
        h, w, c = open_cv_image.shape
        top_bot_pad = 0
        left_right_pad = 0
        if h < self.input_size/2:
            top_bot_pad = int((self.input_size - h) / 4)
            open_cv_image = add_padding(open_cv_image, top_bot_pad, top_bot_pad, 0, 0)
        if w < self.input_size/2:
            left_right_pad = int((self.input_size - w) / 4)
            open_cv_image = add_padding(open_cv_image, 0, 0, left_right_pad, left_right_pad)

        # Detect faces and other attributes
        faces = self.model_detection.get(open_cv_image)

        # Unpad bboxes
        boxes = []
        probs = []
        for f in faces:
            bbox = f.bbox
            bbox_unpad = unpad_bbox(bbox, top_bot_pad, top_bot_pad, left_right_pad, left_right_pad)
            boxes.append(bbox_unpad)
            probs.append(f.det_score)

        self.faces_frame = faces

        return boxes, probs


class FeatureExtractorTF(FeatureExtractor):
    # Specific child class for tensorflow embeddings (original FaceNet)
    def __init__(self, gt_politicians_path, feat_extractor='facenet', feat_input_size=160, device='cpu', **kwargs):
        self.model_path = 'src/external/facenet/20170511-185253'
        self.sess = None
        self.input_tensor = None
        self.embs_tensor = None
        self.phase_train = None
        FeatureExtractor.__init__(self, gt_politicians_path, feat_extractor='facenet', feat_input_size=160, device='cpu', **kwargs)

    def _init_feats_model_(self):
        # Feature extractor model
        if self.feat_extractor_name == 'facenet':
            # self.model_path = 'src/external/facenet/20170511-185253'
            self.graph, self.saver = self.load_model(self.model_path)
            self.input_tensor = self.graph.get_tensor_by_name("input:0")
            self.embs_tensor = self.graph.get_tensor_by_name("embeddings:0")
            self.phase_train = self.graph.get_tensor_by_name("phase_train:0")

    def load_model(self, model, input_map=None):
        # Source: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
        # Check if the model is a model directory (containing a metagraph and a checkpoint file)
        #  or if it is a protobuf file with a frozen graph
        model_exp = os.path.expanduser(model)
        if (os.path.isfile(model_exp)):
            print('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
        else:
            with tf.Graph().as_default() as graph:
                with tf.Session() as sess:
                    print('Model directory: %s' % model_exp)
                    meta_file, ckpt_file = self.get_model_filenames(model_exp)

                    print('Metagraph file: %s' % meta_file)
                    print('Checkpoint file: %s' % ckpt_file)

                    saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
                    saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

        return graph, saver

    def initialize(self):
        self.sess = tf.Session(graph=self.graph)
        meta_file, ckpt_file = self.get_model_filenames(self.model_path)
        self.saver.restore(self.sess, os.path.join(self.model_path, ckpt_file))

    @staticmethod
    def get_model_filenames(model_dir):
        # Source: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def crop_and_batch(self, img, bboxes, output_size=160, tolerance=0, opt_norm=False):
        # Crop face images and put them into a torch tensor (not normalized)
        face_imgs = self.crop_images(img, bboxes, output_size, tolerance)
        # face_imgs_np = np.stack([np.array(face_img_) for face_img_ in face_imgs])
        face_imgs_np = np.stack([self.prewhiten(np.array(face_img_)) for face_img_ in face_imgs])  # This works the best
        # face_imgs_np = np.stack([self.prewhiten(np.array(self.swap_red_blue_cannels(face_img_))) for face_img_ in face_imgs])

        return face_imgs_np

    def crop_and_get_embeddings(self, img, bboxes):
        img_in = img
        face_imgs_np = self.crop_and_batch(img_in, bboxes, output_size=self.feat_input_size, opt_norm=False)
        img_embedding = self.get_embeddings(face_imgs_np)
        return img_embedding

    @staticmethod
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    @staticmethod
    def swap_red_blue_cannels(img_pil):
        r, g, b = img_pil.split()
        new_image = Image.merge("RGB", (b, g, r))
        return new_image

    def get_embeddings(self, face_imgs):
        if self.sess is None:
            raise ValueError("Model has not been initialized. Call initialize() before using get_embeddings_batch().")
        feed_dict = {self.input_tensor: np.array(face_imgs, dtype=np.float32), self.phase_train: False}
        return self.sess.run(self.embs_tensor, feed_dict=feed_dict)
