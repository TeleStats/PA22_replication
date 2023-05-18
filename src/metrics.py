# Here we should have all the methods for the different metrics
import copy
import json
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pathlib import Path
from sklearn.metrics import average_precision_score
# import wandb

from config import get_parser
from external.coco_evaluation import get_coco_metrics_tpfpfn, get_coco_summary, get_coco_metrics, BoundingBox, BBFormat, BBType, CoordinatesType
from utils import convert_str_to_date, filter_path_list_by_date


class GTReader:
    def __init__(self, gt_path, rel_file, fuse_shots=True, time_th=2, flag_process=True):
        # GT info
        self.gt_path = gt_path
        self.gt_df = None
        self.gt_df_processed = None
        self.keys_gt_vid = None
        self.keys_gt_seg = None
        # File relating GT to the actual frames of the videos
        self.rel_file = rel_file
        self.rel_dict = None
        # Fusing time between shots
        self.time_th = time_th
        # Video information
        self.video_fps = 29.97
        # Init variables
        self._init_dataframe_()
        self.df_filter(fuse_shots=fuse_shots, time_th=time_th)  # To be able to test from outside, pass as parameter
        if rel_file is not None:
            self._read_rel_file_()
            if flag_process:
                self._process_gt_()

    def _init_dataframe_(self):
        self.gt_df = pd.read_csv(self.gt_path, sep=',')
        # The ground truth with "modified" columns == -1 are discarded
        # To replicate paper from 2019, do not drop the elements with "-1" in them. Also, use +-2 time threshold.
        idx_modified = self.gt_df[self.gt_df['Modified'] == '-1'].index
        self.gt_df.drop(idx_modified, inplace=True)
        # Drop row if the politician name is missing
        self.gt_df.dropna(subset=['Person norm.', 'Start Norm.'], inplace=True)

    def _read_rel_file_(self):
        with open(self.rel_file, 'r') as json_file:
            self.rel_dict = json.load(json_file)

    def _process_gt_(self):
        # Transform the times of the GT file to the actual times of the video (to compare it to the results).
        # Add video-related seconds columns to GT dataframe
        self.gt_df_processed = copy.copy(self.gt_df)
        self.gt_df_processed['vid_id'] = '0'
        self.gt_df_processed['start_shot'] = 0
        self.gt_df_processed['start_shot_frames'] = 0
        self.gt_df_processed['end_shot'] = 0
        self.gt_df_processed['end_shot_frames'] = 0
        for (index_label, row_series) in self.gt_df.iterrows():
            segment_id = Path(row_series['Date']).stem
            vid_id = '_'.join(segment_id.split('_'))[:-2]
            segment_key = segment_id.split('_')[-1]
            # Translate the normalised initial/end times of the ground truth to the real time in the video
            # Doing +-0.1 seconds to position inside the shot
            self.gt_df_processed['start_shot'].loc[index_label] = self.gt_df['Start Norm.'].loc[index_label] + pd.to_timedelta(self.rel_dict[vid_id][segment_key]['begin_sec'], unit='S') + pd.to_timedelta(0.1, unit='S')
            self.gt_df_processed['end_shot'].loc[index_label] = self.gt_df['End Norm.'].loc[index_label] + pd.to_timedelta(self.rel_dict[vid_id][segment_key]['begin_sec'], unit='S') - pd.to_timedelta(0.1, unit='S')
            # As we have translated the times, we can work on a video basis and not on a segment basis
            self.gt_df_processed['vid_id'].loc[index_label] = vid_id
            # Use date_to_topics.json frames
            # Could be approximated outside by doing self.gt_df_processed['start_shot_frames'] = (self.gt_df_processed['start_shot'] / np.timedelta64(1, 's')) * 29
            # But this is only +3 seconds, so let's be a little bit more accurate
            # Doing +-0.1 seconds to position inside the shot
            self.gt_df_processed['start_shot_frames'].loc[index_label] = ((self.gt_df['Start Norm.'].loc[index_label] / np.timedelta64(1, 's')) * self.video_fps) + self.rel_dict[vid_id][segment_key]['begin_frame'] + (0.1*self.video_fps)
            self.gt_df_processed['end_shot_frames'].loc[index_label] = ((self.gt_df['End Norm.'].loc[index_label] / np.timedelta64(1, 's')) * self.video_fps) + self.rel_dict[vid_id][segment_key]['begin_frame'] - (0.1*self.video_fps)
        # Get video keys
        self.keys_gt_vid = self.gt_df_processed['vid_id'].unique()
        self.keys_gt_seg = self.gt_df_processed['Date'].unique()

    def process_gt_for_detection(self, gt_df):
        # Transform the times of the GT file to the actual times of the video (to compare it to the results).
        # Add video-related seconds columns to GT dataframe
        # Use this function for face_detection using the GT
        gt_df_processed = copy.copy(gt_df)

        gt_df_processed['vid_id'] = '0'
        gt_df_processed['start_shot'] = 0
        gt_df_processed['start_shot_frames'] = 0
        gt_df_processed['end_shot'] = 0
        gt_df_processed['end_shot_frames'] = 0
        for (index_label, row_series) in gt_df.iterrows():
            segment_id = Path(row_series['Date']).stem
            vid_id = '_'.join(segment_id.split('_'))[:-2]
            segment_key = segment_id.split('_')[-1]
            # Translate the normalised initial/end times of the ground truth to the real time in the video
            # Doing +-0.1 seconds to position inside the shot
            gt_df_processed['start_shot'].loc[index_label] = gt_df['Start Norm.'].loc[index_label] + pd.to_timedelta(self.rel_dict[vid_id][segment_key]['begin_sec'], unit='S') + pd.to_timedelta(0.1, unit='S')
            gt_df_processed['end_shot'].loc[index_label] = gt_df['End Norm.'].loc[index_label] + pd.to_timedelta(self.rel_dict[vid_id][segment_key]['begin_sec'], unit='S') - pd.to_timedelta(0.1, unit='S')
            # As we have translated the times, we can work on a video basis and not on a segment basis
            gt_df_processed['vid_id'].loc[index_label] = vid_id
            # Use date_to_topics.json frames
            gt_df_processed['start_shot_frames'].loc[index_label] = ((gt_df['Start Norm.'].loc[index_label] / np.timedelta64(1, 's')) * self.video_fps) + self.rel_dict[vid_id][segment_key]['begin_frame'] + (0.1*self.video_fps)
            gt_df_processed['end_shot_frames'].loc[index_label] = ((gt_df['End Norm.'].loc[index_label] / np.timedelta64(1, 's')) * self.video_fps) + self.rel_dict[vid_id][segment_key]['begin_frame'] - (0.1*self.video_fps)

        return gt_df_processed

    def df_filter(self, fuse_shots, time_th=0):
        df = self.gt_df[['Date', 'Person norm.', 'Start Norm.', 'End Norm.', 'Duration Norm.']]
        self.gt_df = df.drop_duplicates()
        self.gt_df['Start Norm.'] = pd.to_timedelta(self.gt_df['Start Norm.'])
        self.gt_df['End Norm.'] = pd.to_timedelta(self.gt_df['End Norm.'])
        if fuse_shots:
            self.fuse_shots(time_th=time_th)

    @staticmethod
    def fuse_nodes(v_sorted_idxs):
        # Returns: indexes to be fused
        # e.g. [[8, 12, 13, 15], [16], [17, 23]]
        idxs = []  # For the TOC
        idxs_aux = []
        idxs_ret = []

        # Idea: Link start times with end times (already sorted). Fuse the rest of the elements in the middle.
        for idx in v_sorted_idxs:
            # Check if the index is already in the list (i.e. the starting time is already taken into account)
            if idx in idxs:
                # If the index is already in the list, append it to a fuse vector to be returned
                idxs_aux.append(idx)
                idxs.remove(idx)
            else:
                idxs.append(idx)

            # If the index list is empty, means that all indexes have been fused.
            if len(idxs) == 0:
                idxs_ret.append(idxs_aux)
                idxs_aux = []

        return idxs_ret

    def fuse_times(self, df, time_th=0):
        # Convert starting time, ending time and index to numpy arrays
        v_idx = df.index
        v_start = df['Start Norm.'].dt.total_seconds().values - time_th
        v_end = df['End Norm.'].dt.total_seconds().values + time_th

        # Concatenate start and ending times. Also do it for the index (will be used later).
        v_times = np.concatenate((v_start, v_end), axis=0)
        v_idxs = np.concatenate((v_idx, v_idx), axis=0)
        v_times_idxs = np.stack((v_times, v_idxs))
        # Sort the array in order to fuse the starting times and ending times.
        v_times_idxs = v_times_idxs[:, v_times_idxs[0].argsort()]
        idxs_to_fuse = self.fuse_nodes(v_times_idxs[1])

        return idxs_to_fuse

    @staticmethod
    def fuse_in_df(df, idxs_to_fuse):
        # Apply changes to dataframe. To not to change the index values, we are going to replicate the min and max and,
        # outside the function, do a unique() among the whole dataframe
        df_fused = df.copy()
        start_norm_fused = []
        end_norm_fused = []
        duration_norm_fused = []
        ii = []
        for idxs_pol in idxs_to_fuse:
            for idx in idxs_pol:
                [start_norm_fused.append(df['Start Norm.'].loc[idx].min()) for _ in range(len(idx))]
                [end_norm_fused.append(df['End Norm.'].loc[idx].max()) for _ in range(len(idx))]
                [duration_norm_fused.append(end_norm_fused[-1] - start_norm_fused[-1]) for _ in range(len(idx))]
                ii += idx

        df_fused['Start Norm.'][ii] = start_norm_fused
        df_fused['End Norm.'][ii] = end_norm_fused
        df_fused['Duration Norm.'][ii] = duration_norm_fused

        df_fused['Start Norm.'] = pd.to_timedelta(df_fused['Start Norm.'])
        df_fused['End Norm.'] = pd.to_timedelta(df_fused['End Norm.'])
        df_fused['Duration Norm.'] = pd.to_timedelta(df_fused['Duration Norm.'])

        return df_fused

    def fuse_shots(self, time_th=0):
        # Get videos as keys
        vid_keys = self.gt_df['Date'].unique()
        idxs_to_fuse_list = []
        for k_vid in vid_keys:
            df_vid = self.gt_df.loc[self.gt_df['Date'] == k_vid]
            # Take the videos with more than one shot in order to fuse them
            if len(df_vid) > 1:
                # Get politicians as keys
                politic_keys = df_vid['Person norm.'].unique()
                for k_pol in politic_keys:
                    df_vid_pol = df_vid.loc[df_vid['Person norm.'] == k_pol]
                    idxs_to_fuse = self.fuse_times(df_vid_pol, time_th)
                    # df = self.fuse_in_df(df, idxs_to_fuse)
                    # Test
                    idxs_to_fuse_list.append(idxs_to_fuse)

        df = self.fuse_in_df(self.gt_df, idxs_to_fuse_list)
        self.gt_df = df.drop_duplicates()


class FaceMetrics:
    # Class referring to all metrics regarding face detection and identification
    '''
    Here, 2 types of metrics can be made:
    1. Detections vs. GT --> To understand how well the system works.
    2. Detections --> To compute the screen time (and other metrices) of all the TV videos.
    Also, do we want a single file per video or all the results in the same file? For me, a single file per video.
    Anyway, for both we can use the same structure.
    Previous work (results file):
    video;face_name;start_frame;end_frame;total_frame;start_time;end_time;duration
    /net/per610a/home/mvp/mvp_db/news7-lv/2001/2001_03_18_19_00/2001_03_18_19_00.mpg;Takenori KANZAKI.jpg;2813;5285;35961;140;264;124
    Current work (results file):
    videoID,politicID,frame,cx,cy,w,h,prob_det,dist_ID
    /net/per610a/home/mvp/mvp_db/news7-lv/2001/2001_03_18_19_00/2001_03_18_19_00.mpg,Akihiro_Ota,100,150,150,50,50,0.9,0.25
    '''
    def __init__(self, channel, **kwargs):
        # mode='', args_from_outside=None)
        # Detection / identification results, ground truth, and relation files
        self.res_file = kwargs.get('res_file', None)
        self.gt_file = kwargs.get('gt_file', None)
        self.rel_file = kwargs.get('rel_file', None)
        self.mode = kwargs.get('mode', '')
        self.dataset_path = Path(f'data/dataset/train/{channel}')
        self.years_to_test = kwargs.get('years_to_test', None)
        # Results and GT dataframes
        self.gt_df = None
        self.gt_df_processed = None
        self.gt_df_pol_in_segments = None
        self.train_gt_df = None
        self.res_df = None
        self.res_df_processed = None
        self.res_df_pol_in_segments = None
        # Metrics results
        self.res_gt_df_metrics = None
        self.res_demo_df_metrics = None
        # Other variables
        self.from_date = kwargs.get('from_date', None)
        self.to_date = kwargs.get('to_date', None)
        self.keys_gt_vid = None
        self.keys_gt_pol = None
        self.keys_gt_seg = None
        self.keys_res_vid = None
        self.keys_res_pol = None
        self.dict_res_gt_pol = dict()
        self.sample_rate = 1  # Number of frames per second sampled from the video to perform detections
        self.video_fps = 29.97
        self.filter_threshold = 0
        self.distance_threshold = 0.3
        # Initialise variables
        self._res_gt_correspondences_()
        if self.res_file:
            self.set_res_file(self.res_file)
        if self.gt_file and self.rel_file:
            self._read_gt_file_()
            self._init_res_gt_metrics_()
        if self.mode == 'train':
            self._read_train_gt_()

    def set_res_file(self, res_file, year='-1', filter_results=True):
        self.res_file = res_file
        self._read_res_file_()
        self._process_results_(year=year, filter_results=filter_results)

    def set_filter_threshold(self, thresh):
        self.filter_threshold = thresh

    def set_distance_threshold(self, thresh):
        self.distance_threshold = thresh

    def _read_res_file_(self):
        self.res_df = pd.read_csv(self.res_file)
        self.keys_res_pol = self.res_df['ID'].unique()

    def _read_gt_file_(self):
        gt_reader = GTReader(self.gt_file, self.rel_file)
        self.gt_df = gt_reader.gt_df
        self.keys_gt_pol = self.gt_df['Person norm.'].unique()
        self.gt_df_processed = gt_reader.gt_df_processed
        # https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count
        self.gt_df_pol_in_segments = self.gt_df_processed.groupby(['Date', 'Person norm.']).size().reset_index().rename(columns={0: 'count'})
        self.keys_gt_vid = gt_reader.keys_gt_vid
        self.keys_gt_seg = gt_reader.keys_gt_seg

    def _read_train_gt_(self):
        # Read the annotation files in path and put them into a df
        # Take into account:
        # Some .jpg files don't have an associated annotation (.csv files)
        # There's a .txt with japanese characters
        # [source, year, frame, cx, cy, w, h]
        row_list = []
        # I'm sure it can be done faster but we're not going to run this code every day.
        for year_str in self.years_to_test:
            dataset_path_year = self.dataset_path / year_str
            files_annot = [f for f in dataset_path_year.iterdir() if f.suffix == '.csv']
            files_annot = filter_path_list_by_date(files_annot, self.from_date, self.to_date)
            # print(f'{year_str}: {len(files_annot)}')
            for f in files_annot:
                source = f.stem.split('-')[0]
                frame = f.stem.split('-')[1]
                df = pd.read_csv(f)
                rows = df.values.tolist()
                for row in rows:
                    row_list.append([source, year_str, frame, row[0], row[1], row[2], row[3], row[4]])

        self.train_gt_df = pd.DataFrame(data=row_list, columns=['source', 'year', 'frame', 'ID', 'cx', 'cy', 'w', 'h'])

    def _init_res_gt_metrics_(self):
        df_cols = ['vid', 'seg', 'ID', 'frame', 'TP', 'FP', 'FN', 'dist_ID']
        self.res_gt_df_metrics = pd.DataFrame(columns=df_cols)

    def _res_gt_correspondences_(self):
        # Correspondence between detection keys and ground truth keys + political affiliation
        self.dict_res_gt_pol = {
            'Junichiro_Koizumi': {'gt_id': 'Koizumi', 'party': 'LDP'},
            'Taro_Aso': {'gt_id': 'Aso', 'party': 'LDP'},
            'Yasuo_Fukuda': {'gt_id': 'Fukuda', 'party': 'LDP'},
            'Shinzo_Abe': {'gt_id': 'Abe', 'party': 'LDP'},
            'Sadakazu_Tanigaki': {'gt_id': 'Tanigaki', 'party': 'LDP'},
            'Yukio_Hatoyama': {'gt_id': 'Hatoyama', 'party': 'DPJ'},
            'Katsuya_Okada': {'gt_id': 'Okada', 'party': 'DPJ'},
            'Naoto_Kan': {'gt_id': 'Kan', 'party': 'DPJ'},
            'Ichiro_Ozawa': {'gt_id': 'Ozawa', 'party': 'DPJ'},
            'Yoshihiko_Noda': {'gt_id': 'Noda', 'party': 'DPJ'},
            'Seiji_Maehara': {'gt_id': 'Maehara', 'party': 'DPJ'},
            'Takenori_Kanzaki': {'gt_id': 'Kanzaki', 'party': 'Komeito'},
            'Natsuo_Yamaguchi': {'gt_id': 'Yamaguchi', 'party': 'Komeito'},
            'Akihiro_Ota': {'gt_id': 'Ota', 'party': 'Komeito'},
            'Mizuho_Fukushima': {'gt_id': 'Fukushima', 'party': 'SDP'},
            'Takako_Doi': {'gt_id': 'Doi', 'party': 'SDP'},
            'Kazuo_Shii': {'gt_id': 'Shii', 'party': 'CP'},
            'Yoshimi_Watanabe': {'gt_id': 'Watanabe', 'party': 'DPP'},
            'Shintaro_Ishihara': {'gt_id': 'Ishihara', 'party': 'JIP'},
            'Toru_Hashimoto': {'gt_id': 'Hashimoto', 'party': 'JIP'},
            'Yukiko_Kada': {'gt_id': '', 'party': 'LP'},
            'Yuko_Mori': {'gt_id': '', 'party': 'LP'},
            'Taro_Yamamoto': {'gt_id': '', 'party': 'Reiwa'},
            'Kenji_Eda': {'gt_id': '', 'party': 'JIP'},
            'Yorihisa_Matsuno': {'gt_id': '', 'party': 'JIP'},
            'Ichiro_Matsui': {'gt_id': '', 'party': 'IO'},
            'Banri_Kaieda': {'gt_id': '', 'party': 'DPJ'},
            'Renho_Renho': {'gt_id': '', 'party': 'DPJ'},
            'Kohei_Otsuka': {'gt_id': '', 'party': 'DPJ'},
            'Yuichiro_Tamaki': {'gt_id': '', 'party': 'DPJ'},
            'Yukio_Edano': {'gt_id': '', 'party': 'CDP'},
            'Takashi_Tachibana': {'gt_id': '', 'party': 'NHK'},
            'Shigefumi_Matsuzawa': {'gt_id': '', 'party': 'PoH'},
            'Nariaki_Nakayama': {'gt_id': '', 'party': 'PoH'},
            'Yoshihide_Suga': {'gt_id': '', 'party': 'LDP'},
            'Fumio_Kishida': {'gt_id': '', 'party': 'LDP'},
            'Kyoko_Nakayama': {'gt_id': '', 'party': 'PJK'},
            'Masashi_Nakano': {'gt_id': '', 'party': 'PJK'},
            'Takeo_Hiranuma': {'gt_id': '', 'party': 'PJK'},
            'Seiji_Mataichi': {'gt_id': '', 'party': 'SDP'},
            'Tadatomo_Yoshida': {'gt_id': '', 'party': 'SDP'}
        }

        self.dict_gt_res_pol = {
            'Hatoyama': {'res_id': 'Yukio_Hatoyama', 'party': 'DPJ'},
            'Koizumi': {'res_id': 'Junichiro_Koizumi', 'party': 'LDP'},
            'Kanzaki': {'res_id': 'Takenori_Kanzaki', 'party': 'Komeito'},
            'Aso': {'res_id': 'Taro_Aso', 'party': 'LDP'},
            'Fukuda': {'res_id': 'Yasuo_Fukuda', 'party': 'LDP'},
            'Fukushima': {'res_id': 'Mizuho_Fukushima', 'party': 'SDP'},
            'Okada': {'res_id': 'Katsuya_Okada', 'party': 'DPJ'},
            'Ota': {'res_id': 'Akihiro_Ota', 'party': 'Komeito'},
            'Kan': {'res_id': 'Naoto_Kan', 'party': 'DPJ'},
            'Abe': {'res_id': 'Shinzo_Abe', 'party': 'LDP'},
            'Doi': {'res_id': 'Takako_Doi', 'party': 'SDP'},
            'Shii': {'res_id': 'Kazuo_Shii', 'party': 'CP'},
            'Ozawa': {'res_id': 'Ichiro_Ozawa', 'party': 'DPJ'},
            'Tanigaki': {'res_id': 'Sadakazu_Tanigaki', 'party': 'LDP'},
            'Noda': {'res_id': 'Yoshihiko_Noda', 'party': 'DPJ'},
            'Yamaguchi': {'res_id': 'Natsuo_Yamaguchi', 'party': 'Komeito'},
            'Maehara': {'res_id': 'Seiji_Maehara', 'party': 'DPJ'},
            'Watanabe': {'res_id': 'Yoshimi_Watanabe', 'party': 'DPP'},
            'Ishihara': {'res_id': 'Shintaro_Ishihara', 'party': 'JIP'},
            'Hashimoto': {'res_id': 'Toru_Hashimoto', 'party': 'JIP'}
        }

    def _process_results_(self, year='-1', filter_results=True):
        self.res_df_processed = copy.copy(self.res_df)
        self.res_df_processed['vid_id'] = '0'
        self.res_df_processed['year'] = year
        source_keys = self.res_df_processed['source'].unique()
        for k_source in source_keys:
            idx_aux = self.res_df['source'] == k_source
            self.res_df_processed['vid_id'][idx_aux] = Path(k_source).stem
        # Get video keys
        self.keys_res_vid = self.res_df_processed['vid_id'].unique()
        if filter_results:
            self.filter_results()

    def filter_results(self):
        # Filter frames with no politicians detected
        mask_pol_str = self.res_df_processed['ID'] != '0'
        self.res_df_processed = self.res_df_processed[mask_pol_str]
        mask_pol_str = self.res_df_processed['ID'] != '-1'
        self.res_df_processed = self.res_df_processed[mask_pol_str]
        mask_pol_int = self.res_df_processed['ID'].apply(type) != int
        self.res_df_processed = self.res_df_processed[mask_pol_int]
        if not self.res_df_processed.empty:
            mask_distractors = self.res_df_processed['ID'].str[:10] != 'distractor'
            self.res_df_processed = self.res_df_processed[mask_distractors]
        # Filter by distance
        mask_dist = self.res_df_processed['dist_ID'] < self.distance_threshold
        self.res_df_processed = self.res_df_processed[mask_dist]
        # This function is used to filter the amount of detections per shot, to avoid spurious detections / classifications
        # CAVEAT: We should be taking into account the shot length per politician. As we don't have this information, we'll just threshold up to N appearances per politician in a segment.
        # seg_len = len(self.res_df_processed['frame'].unique())
        pol_in_segment = self.res_df_processed.groupby(['source', 'ID']).size().reset_index().rename(columns={0: 'count'})
        # pol_filtered = pol_in_segment[pol_in_segment['count'] > (seg_len / self.filter_threshold)]['ID'].unique()  # Require a politician to appear in 1 of N frames per segment
        # Define time filter masks
        mask_filter = pol_in_segment['count'] > self.filter_threshold
        pol_filtered = pol_in_segment[mask_filter]['ID'].unique()  # Require a politician to appear in 1 of N frames per segment
        mask_frames = self.res_df_processed['ID'].isin(pol_filtered)
        # Filter results with the predefined masks
        self.res_df_processed = self.res_df_processed[mask_frames]
        # if MODIFIER == 'sf':
        # Test to check performance just checking results against existing GT (as in Stanford paper)
        # Filter to get only frames inside GT
        source_gt_files = self.train_gt_df['source'].unique()
        mask_source_gt = self.res_df_processed['source'].isin(source_gt_files)
        self.res_df_processed = self.res_df_processed[mask_source_gt]
        s_file = self.res_df_processed['source'].unique()
        if s_file.size > 0:
            source_gt_frames = self.train_gt_df[self.train_gt_df['source'] == s_file[0]]['frame'].unique()
            source_gt_frames = [int(s) for s in source_gt_frames]
            mask_source_gt_frames = self.res_df_processed['frame'].isin(source_gt_frames)

            self.res_df_processed = self.res_df_processed[mask_source_gt_frames]

    def fuse_res_df(self):
        if self.res_demo_df_metrics is None:
            self.res_demo_df_metrics = copy.copy(self.res_df_processed)
        else:
            self.res_demo_df_metrics = pd.concat([self.res_demo_df_metrics, self.res_df_processed])

    def results_vs_gt(self):
        # 1. Filter results by GT video ID
        for k_vid in self.keys_gt_vid:
            # Debugging
            # if k_vid != '2003_05_15_19_00':
            #     continue

            # Check if results have that video processed
            if k_vid not in self.keys_res_vid:
                # print(f'No detections found for video: {k_vid}')
                continue
            # Filter by using only the results from the GT video
            # print(f'Detections found for video: {k_vid}')
            self.compute_tp_fp_fn(k_vid)

    def compute_tp_fp_fn(self, k_vid):
        gt_df = self.gt_df_processed[self.gt_df_processed['vid_id'] == k_vid]
        res_df = self.res_df_processed[self.res_df_processed['vid_id'] == k_vid]
        res_df['segment_id'] = '-1'  # Used for FPs
        frames_gt = self.res_df['frame'].unique()  # Without filtering (we are comparing with the whole amount of frames in a segment)
        year = k_vid.split('_')[0]
        for (index_label, row_series) in gt_df.iterrows():
            # Current segment being tested
            k_seg = row_series['Date']
            # Check shot by shot with respect to the ground truth
            dets_inside_shot = res_df['frame'].between(int(gt_df['start_shot_frames'].loc[index_label]), gt_df['end_shot_frames'].loc[index_label])
            res_df['segment_id'][dets_inside_shot] = k_seg
            # To compute FNs, take all the frames inside the shot and subtract the TP ones.
            frames_shot = [f for f in frames_gt if int(gt_df['start_shot_frames'].loc[index_label]) <= f <= gt_df['end_shot_frames'].loc[index_label]]
            # To know the overall accuracy of the detector + classifier compute the metrics for the whole system.
            # Check the detected politician keys
            key_politician_gt = gt_df['Person norm.'].loc[index_label]  # Only 1 politician (we go shot by shot in GT)
            # keys_politician_res = res_df['ID'][dets_inside_shot].unique()
            # Add to metrics dataframe
            k_pol = self.dict_gt_res_pol[key_politician_gt]['res_id']
            # for k in keys_politician_res:
            #     if str(k) == '0':  # Frame with no detections
            #         continue
            #     elif k not in self.dict_res_gt_pol.keys():  # Politician not in GT (previous detection versions)
            #         continue
            #     elif self.dict_res_gt_pol[k]['gt_id'] == key_politician_gt:
            #         k_pol = k
            #         break

            mask_dets_pol_inside_shot = res_df['ID'][dets_inside_shot] == k_pol
            # Append to dataframe
            # Here, we count only if the politician is detected inside a frame (ignore multiple detections) to match the
            # GT (which doesn't have locations of the politician in the frame).
            frames_tp, idx_frame = np.unique(res_df['frame'][mask_dets_pol_inside_shot[mask_dets_pol_inside_shot].index].to_numpy(), return_index=True)
            frames_fn = [f for f in frames_shot if f not in frames_tp]
            # If one politicians is two times in the same frame, take one of the distances (the first that was detected)
            dist_tp = res_df['dist_ID'][mask_dets_pol_inside_shot[mask_dets_pol_inside_shot].index].to_numpy()[idx_frame]
            df_metric_row_tp = [{'vid': k_vid, 'seg': k_seg, 'year': year, 'ID': k_pol, 'frame': frame_tp,
                                 'cx': res_df.iloc[idx]['cx'], 'cy': res_df.iloc[idx]['cy'], 'w': res_df.iloc[idx]['w'], 'h': res_df.iloc[idx]['h'],
                                 'TP': 1, 'FP': 0, 'FN': 0, 'dist_ID': dist}
                                for idx, frame_tp, dist in zip(idx_frame, frames_tp, dist_tp)]
            df_metric_row_fn = [{'vid': k_vid, 'seg': k_seg, 'year': year, 'ID': k_pol, 'frame': frame_fn,
                                 'cx': -1, 'cy': -1, 'w': -1, 'h': -1,
                                 'TP': 0, 'FP': 0, 'FN': 1, 'dist_ID': 100}
                                for frame_fn in frames_fn]
            self.res_gt_df_metrics = self.res_gt_df_metrics.append(df_metric_row_tp, ignore_index=True)
            self.res_gt_df_metrics = self.res_gt_df_metrics.append(df_metric_row_fn, ignore_index=True)
            # Remove true positives from dataframe (to not take them into account again)
            res_df = res_df.drop(mask_dets_pol_inside_shot[mask_dets_pol_inside_shot].index)

        # The leftovers are FPs (detections that have not been associated to any GT)
        keys_politician_res = res_df['ID'].unique()
        for k_pol in keys_politician_res:
            if k_pol not in self.dict_res_gt_pol:
                continue
            mask_dets_pol_outside_shots = res_df['ID'] == k_pol
            frames_fp = res_df['frame'][mask_dets_pol_outside_shots[mask_dets_pol_outside_shots].index].to_numpy()
            k_segments = res_df['segment_id'][mask_dets_pol_outside_shots[mask_dets_pol_outside_shots].index].to_numpy()
            dist_fp = res_df['dist_ID'][mask_dets_pol_outside_shots[mask_dets_pol_outside_shots].index]
            df_metric_row_fp = [{'vid': k_vid, 'seg': k_seg, 'year': year, 'ID': k_pol, 'frame': frame_fp,
                                 'cx': res_df.iloc[idx]['cx'], 'cy': res_df.iloc[idx]['cy'], 'w': res_df.iloc[idx]['w'], 'h': res_df.iloc[idx]['h'],
                                 'TP': 0, 'FP': 1, 'FN': 0, 'dist_ID': dist}
                                for idx, (frame_fp, k_seg, dist) in enumerate(zip(frames_fp, k_segments, dist_fp))]
            self.res_gt_df_metrics = self.res_gt_df_metrics.append(df_metric_row_fp, ignore_index=True)

    @staticmethod
    def compute_average_precision(df_pol):
        df_pol_sorted = df_pol.sort_values(by=['dist_ID'])
        mask_tp_fp = df_pol_sorted['dist_ID'] < 100  # Drop FNs
        tp_list = df_pol_sorted[mask_tp_fp]['TP'].to_numpy(dtype=bool)
        dist_list = df_pol_sorted[mask_tp_fp]['dist_ID'].to_numpy()
        if not any(tp_list):
            return 0
        return average_precision_score(tp_list, dist_list)

    def count_detections(self, year='-1'):
        # Count the amount of detections per politician
        if year == '-1':
            # Cumulative sum for all years
            df_pol_count = self.res_demo_df_metrics.groupby(['ID']).size().reset_index().rename(columns={0: 'count'})
        else:
            # Data for the current year
            mask_year = self.res_demo_df_metrics['year'] == year
            df_pol_count = self.res_demo_df_metrics[mask_year].groupby(['ID']).size().reset_index().rename(columns={0: 'count'})

        print(df_pol_count)

    def prepare_metrics(self):
        pass

    @staticmethod
    def transform_df_demo_to_gt(df):
        # Function to transform a demo-like df to a gt-like df for further processing (e.g. statistics-producer)
        # vid_id --> vid
        # TP=1, FP=0, FN=0
        df_new = df.copy()
        df_new.rename(columns={'vid_id': 'vid'}, inplace=True)
        df_new['TP'] = 1
        df_new['FP'] = 0
        df_new['FN'] = 0
        return df_new

    @staticmethod
    def transform_coco_out_to_df(coco_pol):
        # Build new df based on the output of the (custom) COCO thing to match the desired df
        rows_df = []
        for i in range(len(coco_pol['source'])):
            col_source = coco_pol['source'][i].split('-')[0]
            col_ID = coco_pol['ID'][i]
            col_frame = int(coco_pol['source'][i].split('-')[1])
            col_cx = int(coco_pol['bbox'][i][0])
            col_cy = int(coco_pol['bbox'][i][1])
            col_w = int(coco_pol['bbox'][i][2])
            col_h = int(coco_pol['bbox'][i][3])
            col_prob = 0.99
            col_dist = round(coco_pol['dist_ID'][i], 4)
            col_vid = col_source
            col_year = col_source.split('_')[0]
            # Info from detections (contains TPs and FPs)
            col_tp = 1 if coco_pol['matched'][i] else 0  # Has matched
            col_fp = 1 if (not coco_pol['matched'][i]) and (not coco_pol['gt'][i]) else 0  # Has not matched and it's not a FN
            # Info from GT (only contains FNs). For this frame-politician I have X FN that I didn't detect
            col_fn = 1 if (not coco_pol['matched'][i]) and (coco_pol['gt'][i]) else 0  # Has not matched and it's a FN
            rows_df.append([col_source, col_ID, col_frame, col_cx, col_cy, col_w, col_h, col_prob, col_dist, col_vid, col_year, col_tp, col_fp, col_fn])

        df = pd.DataFrame(data=rows_df, columns=['source', 'ID', 'frame', 'cx', 'cy', 'w', 'h', 'prob_det', 'dist_ID', 'vid_id', 'year', 'TP', 'FP', 'FN'])
        return df

    def transform_df_train_to_gt(self, df, gt_boxes, det_boxes):
        # Function to transform a demo-like df to a gt-like df for further processing (e.g. statistics-producer)
        # vid_id --> vid
        # Compute TP, FP, FN and add them to df using the indexes from dets_idx_df (to not re-match bboxes)

        coco_pol = get_coco_metrics_tpfpfn(gt_boxes, det_boxes)
        df_new = self.transform_coco_out_to_df(coco_pol)
        df_new.rename(columns={'vid_id': 'vid'}, inplace=True)
        df_new.sort_values(by=['source', 'frame'], inplace=True)
        return df_new

    def print_metrics_per_video_and_politician(self, k_vids=None):
        tp_overall = 0
        fp_overall = 0
        fn_overall = 0

        if k_vids is None:
            k_vids = self.res_gt_df_metrics['vid'].unique()
        for k_vid in k_vids:
            print(f"\n{k_vid}")
            for k_pol in self.keys_res_pol:
                df_vid = self.res_gt_df_metrics[['TP', 'FP', 'FN']][(self.res_gt_df_metrics['vid'] == k_vid) & (self.res_gt_df_metrics['ID'] == k_pol)]
                tp_pol = df_vid['TP'].sum()
                fp_pol = df_vid['FP'].sum()
                fn_pol = df_vid['FN'].sum()
                tp_overall += tp_pol
                fp_overall += fp_pol
                fn_overall += fn_pol

                print(f"{k_pol}, TP:{tp_pol}, FP:{fp_pol}, FN:{fn_pol}")

        print(f"Overall: TP:{tp_overall}, FP:{fp_overall}, FN:{fn_overall}")

    def print_metrics_per_politician_sample_resolution(self):
        # Print amount of TP, FP, FN per sample (e.g. 1fps --> metrics at a second resolution)
        tp_overall = 0
        fp_overall = 0
        fn_overall = 0
        ap_overall = 0

        print('Resolution: sample')
        keys_res_pol = self.res_gt_df_metrics['ID'].unique()
        for k_pol in keys_res_pol:
            mask_pol = self.res_gt_df_metrics['ID'] == k_pol
            df_pol = self.res_gt_df_metrics[['TP', 'FP', 'FN', 'dist_ID']][mask_pol]
            tp_pol = df_pol['TP'].sum()
            fp_pol = df_pol['FP'].sum()
            fn_pol = df_pol['FN'].sum()
            if df_pol.empty:
                f1_pol = 0
            else:
                f1_pol = tp_pol / (tp_pol + (0.5 * (fp_pol + fn_pol)))

            ap_pol = self.compute_average_precision(df_pol)

            tp_overall += tp_pol
            fp_overall += fp_pol
            fn_overall += fn_pol
            ap_overall += ap_pol

            print(f"{k_pol}: TP:{tp_pol}, FP:{fp_pol}, FN:{fn_pol}, F1:{round(f1_pol, 3)}, AP:{round(ap_pol, 2)}")

        f1_overall = tp_overall / (tp_overall + (0.5 * (fp_overall + fn_overall)))
        map_ = ap_overall / len(self.keys_gt_pol)
        print(f"Overall: TP:{tp_overall}, FP:{fp_overall}, FN:{fn_overall}, F1:{round(f1_overall, 3)}, mAP:{round(map_, 2)}")

    def print_metrics_per_politician_sample_resolution_per_segment(self):
        # For debugging
        # Print amount of TP, FP, FN per sample (e.g. 1fps --> metrics at a second resolution)
        tp_overall = 0
        fp_overall = 0
        fn_overall = 0
        ap_overall = 0

        print('Resolution: sample_per_segment')
        keys_res_pol = self.res_gt_df_metrics['ID'].unique()
        for k_pol in keys_res_pol:
            mask_pol = self.res_gt_df_metrics['ID'] == k_pol
            df_pol = self.res_gt_df_metrics[['TP', 'FP', 'FN', 'seg', 'dist_ID']][mask_pol]
            for k_seg in self.keys_gt_seg:
                df_pol_seg = df_pol[['TP', 'FP', 'FN', 'dist_ID']][df_pol['seg'] == k_seg]
                tp_pol_seg = df_pol_seg['TP'].sum()
                fp_pol_seg = df_pol_seg['FP'].sum()
                fn_pol_seg = df_pol_seg['FN'].sum()
                ap_pol_seg = self.compute_average_precision(df_pol_seg)
                if (tp_pol_seg + (0.5 * (fp_pol_seg + fn_pol_seg))) == 0:
                    f1_pol_seg = 0
                else:
                    f1_pol_seg = tp_pol_seg / (tp_pol_seg + (0.5 * (fp_pol_seg + fn_pol_seg)))

                print(f"{k_pol}, {k_seg}: TP:{tp_pol_seg}, FP:{fp_pol_seg}, FN:{fn_pol_seg}, F1:{round(f1_pol_seg, 3)}, AP:{round(ap_pol_seg, 2)}")

            tp_pol = df_pol['TP'].sum()
            fp_pol = df_pol['FP'].sum()
            fn_pol = df_pol['FN'].sum()
            ap_pol = self.compute_average_precision(df_pol)
            tp_overall += tp_pol
            fp_overall += fp_pol
            fn_overall += fn_pol
            ap_overall += ap_pol
            if (tp_pol + (0.5 * (fp_pol + fn_pol))) == 0:
                f1_pol = 0
            else:
                f1_pol = tp_pol / (tp_pol + (0.5 * (fp_pol + fn_pol)))

            print(f"{k_pol}, overall: TP:{tp_pol}, FP:{fp_pol}, FN:{fn_pol}, F1:{round(f1_pol, 3)}, AP:{round(ap_pol, 2)}")

        f1_overall = tp_overall / (tp_overall + (0.5 * (fp_overall + fn_overall)))
        map = ap_overall / len(self.keys_gt_pol)
        print(f"Overall: TP:{tp_overall}, FP:{fp_overall}, FN:{fn_overall}, F1:{round(f1_overall, 3)}, mAP:{round(map, 2)}")

    def print_metrics_per_politician_segment_resolution(self):
        # Print amount of TP, FP, FN per segment resolution
        pol_dict = dict()
        tp_overall = 0
        fp_overall = 0
        fn_overall = 0

        print('Resolution: segment')
        # Use the dataframes to filter data
        df_tp = self.res_gt_df_metrics[self.res_gt_df_metrics['TP'] == 1]
        df_fp = self.res_gt_df_metrics[self.res_gt_df_metrics['FP'] == 1]
        df_fn = self.res_gt_df_metrics[self.res_gt_df_metrics['FN'] == 1]
        df_tp_grouped = df_tp.groupby(['seg', 'ID']).size().reset_index().rename(columns={0: 'count'})
        df_fp_grouped = df_fp.groupby(['seg', 'ID']).size().reset_index().rename(columns={0: 'count'})
        df_fn_grouped = df_fn.groupby(['seg', 'ID']).size().reset_index().rename(columns={0: 'count'})
        # We have to filter the FN to count only when there's not the politician inside the same segment (now it's on samples)
        df_compare_tp_fn = pd.merge(df_fn_grouped[['seg', 'ID']], df_tp_grouped[['seg', 'ID']], how='outer', indicator='Exist')
        df_fn_grouped = df_fn_grouped[df_compare_tp_fn['Exist'] == 'left_only']
        keys_res_pol = self.res_gt_df_metrics['ID'].unique()
        for k_pol in keys_res_pol:
            pol_dict[k_pol] = dict()
            pol_dict[k_pol]['TP'] = len(df_tp_grouped[df_tp_grouped['ID'] == k_pol])
            pol_dict[k_pol]['FP'] = len(df_fp_grouped[df_fp_grouped['ID'] == k_pol])
            pol_dict[k_pol]['FN'] = len(df_fn_grouped[df_fn_grouped['ID'] == k_pol])
            if (pol_dict[k_pol]['TP'] + (0.5 * (pol_dict[k_pol]['FP'] + pol_dict[k_pol]['FN']))) == 0:
                f1_pol = 0
            else:
                f1_pol = pol_dict[k_pol]['TP'] / (pol_dict[k_pol]['TP'] + (0.5 * (pol_dict[k_pol]['FP'] + pol_dict[k_pol]['FN'])))

            tp_overall += pol_dict[k_pol]['TP']
            fp_overall += pol_dict[k_pol]['FP']
            fn_overall += pol_dict[k_pol]['FN']

            print(f"{k_pol}: TP:{pol_dict[k_pol]['TP']}, FP:{pol_dict[k_pol]['FP']}, FN:{pol_dict[k_pol]['FN']}, F1:{round(f1_pol, 3)}")

        f1_overall = tp_overall / (tp_overall + (0.5 * (fp_overall + fn_overall)))
        print(f"Overall per {len(self.keys_gt_seg)} segments: TP:{tp_overall}, FP:{fp_overall}, FN:{fn_overall}, F1:{round(f1_overall, 3)}")

    def print_detection_percentages(self):
        mask_face = self.res_demo_df_metrics['prob_det'] > 0.1
        df_faces = self.res_demo_df_metrics[mask_face].groupby(['source', 'frame']).size().reset_index().rename(columns={0: 'count'})
        screen_seconds = len(self.res_demo_df_metrics.groupby(['source', 'frame']).size())  # Grouping of frames and videos (total seconds)
        face_screen_seconds = len(df_faces)  # Grouping of frames and videos (total seconds)
        print(f"Face in screen: {round((face_screen_seconds / screen_seconds) * 100, 1)}%")
        print(f"Faces total: {df_faces['count'].sum()}")
        print(f"{len(df_faces[df_faces['count'] == 1])}, {round((len(df_faces[df_faces['count'] == 1]) / screen_seconds) * 100, 1)}%")
        print(f"{len(df_faces[df_faces['count'] == 2])}, {round((len(df_faces[df_faces['count'] == 2]) / screen_seconds) * 100, 1)}%")
        print(f"{len(df_faces[df_faces['count'] == 3])}, {round((len(df_faces[df_faces['count'] == 3]) / screen_seconds) * 100, 1)}%")
        print(f"{len(df_faces[df_faces['count'] == 4])}, {round((len(df_faces[df_faces['count'] == 4]) / screen_seconds) * 100, 1)}%")
        print(f"{len(df_faces[df_faces['count'] >= 5])}, {round((len(df_faces[df_faces['count'] >= 5]) / screen_seconds) * 100, 1)}%")

        df_faces_years = self.res_demo_df_metrics[mask_face].groupby(['source', 'frame', 'year']).size().reset_index().rename(columns={0: 'count'})
        years = df_faces_years['year'].unique()
        for year in years:
            print(f'year {year}')
            mask_year = self.res_demo_df_metrics['year'] == year
            mask_faces_year = df_faces_years['year'] == year
            df_faces_year = df_faces_years[mask_faces_year]
            screen_seconds = len(self.res_demo_df_metrics[mask_year].groupby(['source', 'frame']).size())
            face_screen_seconds = len(df_faces_year)  # Grouping of frames and videos (total seconds)
            print(f"Face in screen: {round((face_screen_seconds / screen_seconds) * 100, 1)}%")
            print(f"Faces total: {df_faces_year['count'].sum()}")
            print(f"{len(df_faces_year[df_faces_year['count'] == 1])}, {round((len(df_faces_year[df_faces_year['count'] == 1]) / screen_seconds) * 100, 1)}%")
            print(f"{len(df_faces_year[df_faces_year['count'] == 2])}, {round((len(df_faces_year[df_faces_year['count'] == 2]) / screen_seconds) * 100, 1)}%")
            print(f"{len(df_faces_year[df_faces_year['count'] == 3])}, {round((len(df_faces_year[df_faces_year['count'] == 3]) / screen_seconds) * 100, 1)}%")
            print(f"{len(df_faces_year[df_faces_year['count'] == 4])}, {round((len(df_faces_year[df_faces_year['count'] == 4]) / screen_seconds) * 100, 1)}%")
            print(f"{len(df_faces_year[df_faces_year['count'] >= 5])}, {round((len(df_faces_year[df_faces_year['count'] >= 5]) / screen_seconds) * 100, 1)}%")

    def print_bboxes_size(self):
        pols_in_res = self.res_demo_df_metrics['ID'].unique()
        mask_prob = self.res_demo_df_metrics['prob_det'] > 0.1
        for k_pol in pols_in_res:
            mask_pol = self.res_demo_df_metrics[mask_prob]['ID'] == k_pol
            df_pol = self.res_demo_df_metrics[mask_prob][mask_pol]
            bbox_sizes = df_pol['w'] * df_pol['h'] * 100 / (352 * 240)
            print(f"Average bbox for {k_pol}: {round(bbox_sizes.mean(), 1)}")

    def build_train_metrics(self, flag_det=False):
        assert args.mode == 'train'

        # Creating the object of the class BoundingBoxes
        gt_boxes = []
        # GT loop
        pol_gt = self.train_gt_df['ID'].unique()
        for (index_label, row_series) in self.train_gt_df.iterrows():
            img_name = '-'.join((row_series['source'], str(row_series['frame'])))
            id = row_series['ID'] if flag_det is False else 1
            gt_bbox = BoundingBox(image_name=img_name,
                                  class_id=id,
                                  coordinates=(row_series['cx'], row_series['cy'], row_series['w'], row_series['h']),
                                  type_coordinates=CoordinatesType.ABSOLUTE,
                                  confidence=1.0,
                                  format=BBFormat.XYWH)
            gt_boxes.append(gt_bbox)

        det_boxes = []
        # Results loop
        for (index_label, row_series) in self.res_demo_df_metrics.iterrows():
            # if (row_series['ID'] not in pol_gt) and not flag_det:
            #     continue

            id = row_series['ID'] if flag_det is False else 1
            img_name = '-'.join((row_series['source'], str(row_series['frame'])))
            det_bbox = BoundingBox(image_name=img_name,
                                   class_id=id,
                                   coordinates=(row_series['cx'], row_series['cy'], row_series['w'], row_series['h']),
                                   type_coordinates=CoordinatesType.ABSOLUTE,
                                   confidence=1 - row_series['dist_ID'],
                                   format=BBFormat.XYWH,
                                   bb_type=BBType.DETECTED)

            det_boxes.append(det_bbox)

        return gt_boxes, det_boxes

    @staticmethod
    def compute_AP(gt_boxes, det_boxes):
        # Create an evaluator object in order to obtain the metrics
        coco_ap_glob = get_coco_summary(gt_boxes, det_boxes)
        coco_pol = get_coco_metrics(gt_boxes, det_boxes)
        # Correction for missing GT and Dets
        for pol in coco_pol.keys():
            if coco_pol[pol]['AP'] is None:
                coco_pol[pol]['AP'] = 0.0
                coco_pol[pol]['total positives'] = 0
                coco_pol[pol]['TP'] = 0
                coco_pol[pol]['FP'] = 0

        # Sort the results based on AP (sort in descending order)
        # idx_sorted = list(np.argsort(-np.asarray([coco_pol[pol]['AP'] for pol in coco_pol])))
        # Sort the results by alphabetical order
        idx_sorted = list(np.argsort(np.asarray(list(coco_pol.keys()))))
        keys_pol = [pol for pol in coco_pol]
        keys_pol_sorted = [keys_pol[i] for i in idx_sorted]

        row_metrics = []
        tp_all = 0
        fp_all = 0
        fn_all = 0
        ap_all = 0
        gt_all = 0
        for pol in keys_pol_sorted:
            tp = coco_pol[pol]['TP']
            fp = coco_pol[pol]['FP']
            fn = coco_pol[pol]['total positives'] - tp
            tp_all += tp
            fp_all += fp
            fn_all += fn
            ap_all += coco_pol[pol]['AP']
            gt_all += coco_pol[pol]['total positives']
            if tp == 0:
                prec, rec, f1 = 0, 0, 0
            else:
                prec = tp / (fp + tp)
                rec = tp / (fn + tp)
                f1 = tp / (tp + (0.5 * (fp + fn)))

            # For wandb logs
            row_metrics.append([pol, round(prec, 2), round(rec, 2), round(f1, 3), round(coco_pol[pol]['AP']*100, 1), coco_pol[pol]['total positives']])

            if flag_det:
                print(f"{coco_pol[pol]['class']}: TP={tp_all}, FN={fn_all}, "
                      f"R={round(rec, 3)}, GT={coco_pol[pol]['total positives']}")
            else:
                print(f"{coco_pol[pol]['class']}: AP={round(coco_pol[pol]['AP']*100, 1)}, P={round(prec, 2)}, "
                      f"R={round(rec, 2)}, F1={round(f1, 3)}, GT={coco_pol[pol]['total positives']}")

        prec_all = tp_all / (fp_all + tp_all)
        rec_all = tp_all / (fn_all + tp_all)
        f1_all = tp_all / (tp_all + (0.5 * (fp_all + fn_all)))
        map_all = ap_all / len(coco_pol)

        # For wandb logs
        # Insert to be the first in the list (for visualization in wandb)
        # Trick to have "All" the first and then order by F1 score
        col_metrics = ['ID', 'P', 'R', 'F1', 'AP', 'GT']
        df_metrics = pd.DataFrame(data=row_metrics, columns=col_metrics).sort_values(by='F1', ascending=False)
        cols_wandb = ['ID', 'P', 'R', 'F1', 'AP', 'APS', 'APM', 'APL', 'GT']
        df_all = pd.DataFrame(data=[['All', round(prec_all, 2), round(rec_all, 2), round(f1_all, 3),
                                     round(coco_ap_glob['AP50']*100, 1), round(coco_ap_glob['APsmall']*100, 1),
                                     round(coco_ap_glob['APmedium']*100, 1), round(coco_ap_glob['APlarge']*100, 1), gt_all]],
                              columns=cols_wandb)
        df_metrics = pd.concat([df_all, df_metrics], ignore_index=True)

        print(f"Overall: AP50={round(coco_ap_glob['AP50']*100, 1)}, APS={round(coco_ap_glob['APsmall']*100, 1)}, "
              f"APM={round(coco_ap_glob['APmedium']*100, 1)}, APL={round(coco_ap_glob['APlarge']*100, 1)}, "
              f"P={round(prec_all, 2)}, "f"R={round(rec_all, 2)}, F1={round(f1_all, 3)}, GT={gt_all} \n")

        return df_metrics

    def save_df_all_info(self, df):
        # Save df with all info for further analysis
        if args.modifier == "":
            save_path = SAVE_METRICS_PATH / f'{args.detector}-{args.feats}-{args.mod_feat}.pkl'
        else:
            save_path = SAVE_METRICS_PATH / f'{args.detector}-{args.feats}-{args.mod_feat}-{args.modifier}.pkl'

        os.makedirs(str(save_path.parent), exist_ok=True)
        # Add extra data
        # Political affiliation
        df_new = df.copy()
        df_new['party'] = '-1'
        politicians = df['ID'].unique()
        for pol in politicians:
            if pol not in self.dict_res_gt_pol.keys():
                continue
            mask_pol = df['ID'] == pol
            df_new['party'][mask_pol] = self.dict_res_gt_pol[pol]['party']

        pd.to_pickle(df_new, str(save_path))


def main_gt():
    # Comparing with GT
    # face_metrics = FaceMetrics(RES_FILE, gt_file=GT_PATH, rel_file=REL_PATH)
    # face_metrics.results_vs_gt()
    # Without GT
    # face_metrics = FaceMetrics(RES_FILE)
    # face_metrics.count_detections()

    # Testing for the videos in the GT. These are located in server per920a
    face_metrics = FaceMetrics(channel=args.channel, res_file=None, gt_file=GT_PATH, rel_file=REL_PATH)
    face_metrics.set_filter_threshold(args.filter)
    face_metrics.set_distance_threshold(args.dist)
    for year_str in YEARS_TO_TEST:
        results_year_path = RES_PATH / year_str
        results_year_day_path = [x for x in results_year_path.iterdir() if x.is_file() and x.suffix == '.csv']

        for result_file in results_year_day_path:
            face_metrics.set_res_file(str(result_file), year=year_str)
            face_metrics.results_vs_gt()
            face_metrics.fuse_res_df()

    print(f'Detector: {args.detector}')
    print(f'Mod_feat: {args.mod_feat}')
    print(f'Filter: {face_metrics.filter_threshold}')

    if args.resolution == 'sample':
        face_metrics.print_metrics_per_politician_sample_resolution()
        # face_metrics.print_metrics_per_politician_sample_resolution_per_segment()
    elif args.resolution == 'segment':
        face_metrics.print_metrics_per_politician_segment_resolution()
    elif args.resolution == 'all':
        face_metrics.print_metrics_per_politician_sample_resolution()
        face_metrics.print_metrics_per_politician_segment_resolution()

    # face_metrics.print_metrics_per_video_and_politician()
    face_metrics.print_bboxes_size()
    if len(args.year) == 0:
        # Only save df when all the info is available
        # Parser in res_gt_statistics.py
        face_metrics.save_df_all_info(face_metrics.res_gt_df_metrics)


def main_demo():
    # Testing for the videos in the whole dataset.
    face_metrics = FaceMetrics(channel=args.channel, res_file=None, years_to_test=YEARS_TO_TEST)
    face_metrics.set_filter_threshold(args.filter)
    face_metrics.set_distance_threshold(args.dist)
    # Testing
    # files_to_test = ['2001_03_16_19_00', '2001_03_17_19_00', '2001_03_18_19_00', '2001_03_19_19_00', '2001_03_20_19_00',
    #                  '2001_03_21_19_00', '2001_10_05_19_00', '2001_10_06_19_00', '2001_10_07_19_00', '2001_10_08_19_00',
    #                  '2001_10_09_19_00', '2001_10_10_19_00', '2001_10_11_19_00']

    files_to_test = ['2003_04_13_19_00']

    print(f'Detector: {args.detector}')
    print(f'Mod_feat: {args.mod_feat}')
    print(f'Filter: {face_metrics.filter_threshold}')
    print(f'Distance: {face_metrics.distance_threshold}')

    for year_str in YEARS_TO_TEST:
        print(f'Year: {year_str}')
        results_year_path = RES_PATH / year_str
        results_year_day_path = [x for x in results_year_path.iterdir() if x.is_file() and x.suffix == '.csv']
        results_year_day_path = filter_path_list_by_date(results_year_day_path, FROM_DATE, TO_DATE)

        if len(results_year_day_path) == 0:
            continue

        for result_file in results_year_day_path:
            # if result_file.stem not in files_to_test:
            #     continue
            face_metrics.set_res_file(str(result_file), year=year_str)
            face_metrics.fuse_res_df()

        # Count per year
        face_metrics.count_detections(year=year_str)

    # Count overall
    print(f'Overall: ')
    face_metrics.count_detections()
    face_metrics.print_bboxes_size()
    if len(args.year) == 0:
        # Only save df when all the info is available
        df_save = face_metrics.transform_df_demo_to_gt(face_metrics.res_demo_df_metrics)
        face_metrics.save_df_all_info(df_save)


def main_dets():
    face_metrics = FaceMetrics(channel=args.channel, res_file=None, years_to_test=YEARS_TO_TEST)
    face_metrics.set_filter_threshold(args.filter)
    face_metrics.set_distance_threshold(args.dist)

    for year_str in YEARS_TO_TEST:
        print(f'Year: {year_str}')
        results_year_path = DET_PATH / year_str
        results_year_day_path = [x for x in results_year_path.iterdir() if x.is_file() and x.suffix == '.csv']
        results_year_day_path = filter_path_list_by_date(results_year_day_path, FROM_DATE, TO_DATE)

        if len(results_year_day_path) == 0:
            continue

        for result_file in results_year_day_path:
            # if result_file.stem not in files_to_test:
            #     continue
            face_metrics.set_res_file(str(result_file), year=year_str, filter_results=False)
            face_metrics.fuse_res_df()

    face_metrics.print_detection_percentages()
    face_metrics.print_bboxes_size()


def main_train(flag_det=False):
    face_metrics = FaceMetrics(channel=args.channel, res_file=None, mode='train', from_date=FROM_DATE, to_date=TO_DATE, years_to_test=YEARS_TO_TEST)
    face_metrics.set_filter_threshold(args.filter)
    face_metrics.set_distance_threshold(args.dist)

    for year_str in YEARS_TO_TEST:
        print(f'Year: {year_str}')
        results_year_path = RES_PATH / year_str
        results_year_day_path = [x for x in results_year_path.iterdir() if x.is_file() and x.suffix == '.csv']
        results_year_day_path = filter_path_list_by_date(results_year_day_path, FROM_DATE, TO_DATE)

        for result_file in results_year_day_path:
            # if result_file.stem not in files_to_test:
            #     continue
            if flag_det:
                face_metrics.set_res_file(str(result_file), year=year_str, filter_results=False)
            else:
                face_metrics.set_res_file(str(result_file), year=year_str, filter_results=True)
            face_metrics.fuse_res_df()

    face_metrics.print_detection_percentages()
    face_metrics.print_bboxes_size()
    gt_boxes, det_boxes = face_metrics.build_train_metrics(flag_det=flag_det)
    df_metrics = face_metrics.compute_AP(gt_boxes, det_boxes)

    # # wandb logs
    if not flag_det:
    #     min_date = '/'.join(face_metrics.res_demo_df_metrics['source'].min().split('_')[:3])
    #     max_date = '/'.join(face_metrics.res_demo_df_metrics['source'].max().split('_')[:3])
    #     # run.log({f"Metrics from {min_date}-{max_date}": df_metrics})
    #     if args.modifier == "":
    #         run.log({f"{args.channel}": df_metrics})
    #     else:
    #         run.log({f"{args.channel}-{args.modifier}": df_metrics})
    #
        if len(args.year) == 0:
            # Only save df when all the info is available
            df_save = face_metrics.transform_df_train_to_gt(face_metrics.res_demo_df_metrics, gt_boxes, det_boxes)
            face_metrics.save_df_all_info(df_save)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    FROM_DATE = convert_str_to_date(args.from_date)
    TO_DATE = convert_str_to_date(args.to_date)
    MODIFIER = args.modifier
    # RES_FILE = Path('data/results/results_sample.csv')
    RES_FILE = Path('data/results/results_haolin_segment_newformat.csv')
    # RES_FILE = Path('data/results/2012_12_04_19_00.csv')

    GT_PATH = Path(args.gt_path)
    REL_PATH = Path(args.rel_path)
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / Path(args.detector + '-' + args.feats + '-' + args.mod_feat)
    SAVE_METRICS_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / 'results'
    DATASET_TRAIN_PATH = Path(f'data/dataset/train/{args.channel}')
    DET_PATH = None

    if args.use_dets:
        DET_PATH = Path(args.res_path) / Path(args.mode) / Path(args.detector + '-' + 'detections')

    # Do the detection for a specific year
    if len(args.year) > 0:
        YEARS_TO_TEST = args.year
    else:
        YEARS_TO_TEST = [str(year_path.stem) for year_path in RES_PATH.iterdir() if year_path.is_dir()]

    # Weights and biases logger
    # group_name = args.mod_feat.upper()
    # run = wandb.init(project="TeleStats", entity="agirbau", group=f'{group_name}', job_type='eval')
    # os.environ["WANDB_RUN_GROUP"] = group_name
    # if args.modifier != '':
    #     run.name = f'{args.channel}-{args.mode}-{args.detector}-{args.feats}-{args.mod_feat}-{args.modifier}'
    # else:
    #     run.name = f'{args.channel}-{args.mode}-{args.detector}-{args.feats}-{args.mod_feat}'

    if args.mode == 'gt':
        if DET_PATH:
            main_dets()
        else:
            main_gt()
    elif args.mode == 'demo':
        if DET_PATH:
            main_dets()
        else:
            main_demo()
    elif args.mode == 'train':
        flag_det = False
        if args.use_dets:
            flag_det = True

        main_train(flag_det=flag_det)
