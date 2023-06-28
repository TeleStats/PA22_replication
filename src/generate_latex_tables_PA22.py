# Set of scripts to generate Political Analysis 2022 latex tables
# Imports
import pandas as pd
from pathlib import Path
import random

from src.statistics_producer import ResStatistics
from src.utils import bbox_cx_cy_w_h_to_x1_y1_x2_y2, compute_ious_bboxes


class MetricsGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_statistics_object(detector, channel, **kwargs):
        res_path = kwargs.get("res_path", "data/results")
        mode = kwargs.get("mode", "train")
        mod_feat = kwargs.get("mod_feat", "fcg_average_vote-sf")
        feats = kwargs.get("feats", "resnetv1")
        FROM_DATE = kwargs.get("from_date", "2000_01_01")
        TO_DATE = kwargs.get("to_date", "2100_01_01")

        DATA_PATH = Path(f'data/dataset/train/{channel}')
        RES_PATH = Path(res_path) / Path(channel) / Path(mode) / 'results'
        res_statistics = ResStatistics(RES_PATH, mode=mode, channel=channel, detector=detector, mod_feat=mod_feat,
                                       feats=feats, from_date_str=FROM_DATE, to_date_str=TO_DATE)
        df_res = res_statistics.res_df_all_info_from_to

        return res_statistics

    @staticmethod
    def print_metrics_per_politician_sample_resolution(df_res):
        # Print amount of TP, FP, FN per sample (e.g. 1fps --> metrics at a second resolution)
        tp_overall = 0
        fp_overall = 0
        fn_overall = 0

        print('Resolution: sample')
        keys_res_pol = sorted(df_res['ID'].unique())
        for k_pol in keys_res_pol:
            if k_pol == "Overall":
                continue
            mask_pol = df_res['ID'] == k_pol
            # df_pol = df_res[['TP', 'FP', 'FN', 'dist_ID']][mask_pol]
            df_pol = df_res[mask_pol]
            tp_pol = df_pol['TP'].sum()
            fp_pol = df_pol['FP'].sum()
            fn_pol = df_pol['FN'].sum()
            if df_pol.empty:
                f1_pol = 0
            else:
                f1_pol = tp_pol / (tp_pol + (0.5 * (fp_pol + fn_pol)))

            tp_overall += tp_pol
            fp_overall += fp_pol
            fn_overall += fn_pol

            prec_pol = tp_pol / (tp_pol + fp_pol)
            rec_pol = tp_pol / (tp_pol + fn_pol)

            print(f"{k_pol}: P:{round(prec_pol, 2)}, R:{round(rec_pol, 2)}, F1:{round(f1_pol, 3)}")
            # print(f"{k_pol}: TP:{tp_pol}, FP:{fp_pol}, FN:{fn_pol}")

        prec_overall = tp_overall / (tp_overall + fp_overall)
        rec_overall = tp_overall / (tp_overall + fn_overall)
        f1_overall = tp_overall / (tp_overall + (0.5 * (fp_overall + fn_overall)))
        print(f"Overall: P:{round(prec_overall, 2)}, R:{round(rec_overall, 2)}, F1:{round(f1_overall, 3)}")

    def compute_metrics(self, df_res, key_filter='Overall', opt_correct=False):
        # Compute precision, recall and F1 score based on the "corrected" FPs
        df_res_masked = df_res[df_res['ID'] == key_filter]
        tp_ovr = df_res_masked['TP'].sum()
        fn_ovr = df_res_masked['FN'].sum()
        if opt_correct:
            df_fp_sf = self.filter_fps_based_on_iou(df_res_masked)
            fp_ovr = df_fp_sf['FP'].sum()
        else:
            fp_ovr = df_res_masked['FP'].sum()

        prec_ovr = tp_ovr / (tp_ovr + fp_ovr)
        rec_ovr = tp_ovr / (tp_ovr + fn_ovr)
        f1_ovr = tp_ovr / (tp_ovr + (0.5 * (fp_ovr + fn_ovr)))

        return prec_ovr, rec_ovr, f1_ovr

    @staticmethod
    def filter_fps_based_on_iou(df_res):
        mask_tp = df_res['TP'] == 1
        mask_fp = df_res['FP'] == 1
        mask_fn = df_res['FN'] == 1
        df_tp = df_res[mask_tp]
        df_fp = df_res[mask_fp]
        df_fn = df_res[mask_fn]

        # If we want to evaluate as SF paper we have to take every FP, compare it with every FN, and match
        # those that do not share the same class to have the "true FPs", as the dataset is not fully annotated
        # First, filter by source and frame
        # Intersection of the sources in FPs and FNs
        sources_fp = df_fp['source'].unique()
        sources_fn = df_fn['source'].unique()
        sources_list = list(set(sources_fp) & set(sources_fn))
        sf_fps = 0
        row_sf_fp = []
        for k_source in sources_list:
            mask_source_fp = df_fp['source'] == k_source
            mask_source_fn = df_fn['source'] == k_source
            df_fp_src = df_fp[mask_source_fp]
            df_fn_src = df_fn[mask_source_fn]
            # Intersection of the frames within the same source
            frames_fp = df_fp_src['frame'].unique()
            frames_fn = df_fn_src['frame'].unique()
            frames_list = list(set(frames_fp) & set(frames_fn))
            # Now compute the ious per frame between the FNs and FPs. If an IoU > thresh, it is a true FP
            for k_frame in frames_list:
                mask_frame_fp = df_fp_src['frame'] == k_frame
                mask_frame_fn = df_fn_src['frame'] == k_frame
                df_fp_frame = df_fp_src[mask_frame_fp]
                df_fn_frame = df_fn_src[mask_frame_fn]
                bboxes_fp = df_fp_frame[['cx', 'cy', 'w', 'h']].to_numpy()
                bboxes_fn = df_fn_frame[['cx', 'cy', 'w', 'h']].to_numpy()
                # From cx, cy, w, h to x1,y1,x2,y2
                bboxes_fp = [bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox) for bbox in bboxes_fp]
                bboxes_fn = [bbox_cx_cy_w_h_to_x1_y1_x2_y2(bbox) for bbox in bboxes_fn]
                ious = compute_ious_bboxes(bboxes_fp, bboxes_fn)
                # Finally, check if the iou is > 0.5 and add them to the "true_fps" list
                for i, _ in enumerate(bboxes_fp):
                    for j, _ in enumerate(bboxes_fn):
                        if ious[i, j] > 0.5:
                            actual_fp_serie = df_fp_frame.iloc[i]
                            actual_fn_serie = df_fn_frame.iloc[j]
                            row_sf_fp.append(actual_fp_serie.to_list())
                            sf_fps += 1

        df_fp_sf = pd.DataFrame(data=row_sf_fp, columns=df_fp.columns.to_list())
        # print(sf_fps)
        # df_fp_sf.head()

        return df_fp_sf


def generate_df_from_dict(table_dict):
    rows_df = []
    cols_df = ['Channel', 'Detector', 'Classifier', 'P', 'R', 'F1']
    for channel in table_dict.keys():
        c_dict = table_dict[channel]
        for detector in c_dict.keys():
            d_dict = c_dict[detector]
            for feat in d_dict.keys():
                f_dict = d_dict[feat]
                for classifier in f_dict.keys():
                    prec_ = f_dict[classifier]['P']
                    rec_ = f_dict[classifier]['R']
                    f1_ = f_dict[classifier]['F1']

                    if classifier == 'fcg_average_vote':
                        classifier_latex_name = 'vote' if not FLAG_TRACKING_TABLE else 'Yes'  # Reuse this for "Tracking vs. no Tracking table"
                    elif classifier == 'fcg_average_centroid':
                        classifier_latex_name = 'centroid'
                    elif classifier == 'knn_3':
                        classifier_latex_name = 'KNN'
                    elif classifier == 'fcgNT_average_vote':
                        classifier_latex_name = 'voteNT' if not FLAG_TRACKING_TABLE else 'No'  # Reuse this for "Tracking vs. no Tracking table"
                    else:
                        classifier_latex_name = classifier

                    row_ = [channel, detector.upper(), classifier_latex_name, prec_, rec_, f1_]
                    rows_df.append(row_)

    df = pd.DataFrame(rows_df, columns=cols_df)

    # Calculate average values
    avg_values = df.groupby(['Detector', 'Classifier']).mean().reset_index()
    # Specify the decimal places for each column
    if FLAG_DETECTION_TABLE:
        decimal_places = {
            'P': 4,
            'R': 4,
            'F1': 4
        }
    else:
        decimal_places = {
            'P': 2,
            'R': 2,
            'F1': 3
        }
    # Round the mean values based on the specified decimal places
    avg_values = avg_values.round(decimal_places)
    # Add "Overall" channel
    avg_values['Channel'] = 'Overall'
    # Reorder columns
    avg_values = avg_values[['Channel', 'Detector', 'Classifier', 'P', 'R', 'F1']]
    # Concatenate with original DataFrame
    df = pd.concat([df, avg_values])
    # Sort for desired column order
    df = df.sort_values(by=['Channel'])

    return df


def reorder_df_dets(df):
    # Give order to the df based on the Latex table 3 (detection error) in PA22
    # Extract the channel names from column names
    channel_columns = sorted(list(set([col.split('_')[0] for col in df.columns if '_' in col])))
    # Move "Overall" to the beginning of the list
    channel_columns.remove('Overall')
    channel_columns.insert(0, 'Overall')
    # For quick modification
    # channel_columns = ['Overall', 'news7-lv', 'hodost-lv']
    # channel_columns = ['CNNW', 'FOXNEWSW', 'MSNBCW', 'news7-lv', 'hodost-lv']  # For tracking vs. no tracking performance
    channel_columns = ['CNNW', 'FOXNEWSW', 'MSNBCW', 'news7-lv', 'hodost-lv', 'Overall']  # For detector performance
    # Specify column order
    chan_list = [[f'{chan}_R'] for chan in channel_columns]
    column_order = ['Detector'] + [item for sublist in chan_list for item in sublist]

    # Reindex the DataFrame with the desired column order
    df_ = df.reindex(columns=column_order)
    df_[[item for sublist in chan_list for item in sublist]] = (1 - df_[[item for sublist in chan_list for item in sublist]]) * 100

    return df_.round(2)

def reorder_df(df):
    # Give order to the df based on the Latex tables order
    # Column order
    # Detector, Classifier, Overall_P, Overall_R, Overall_F1, Channel_P, Channel_R, Channel_F1...

    # Extract the channel names from column names
    channel_columns = sorted(list(set([col.split('_')[0] for col in df.columns if '_' in col])))
    # Move "Overall" to the beginning of the list
    channel_columns.remove('Overall')
    channel_columns.insert(0, 'Overall')
    # For quick modification
    # channel_columns = ['Overall', 'news7-lv', 'hodost-lv']
    # channel_columns = ['CNNW', 'FOXNEWSW', 'MSNBCW', 'news7-lv', 'hodost-lv']  # For tracking vs. no tracking performance
    # Specify column order
    chan_list = [[f'{chan}_P', f'{chan}_R', f'{chan}_F1'] for chan in channel_columns]
    column_order = ['Detector', 'Classifier'] + [item for sublist in chan_list for item in sublist]

    # Reindex the DataFrame with the desired column order
    df_ = df.reindex(columns=column_order)

    return df_


def generate_table_from_dict(table_dict):
    df = generate_df_from_dict(table_dict)

    # Pivot the DataFrame
    if FLAG_DETECTION_TABLE:
        pivot_df = df.pivot_table(index=['Detector'], columns='Channel', values=['R']).reset_index()
    else:
        pivot_df = df.pivot_table(index=['Detector', 'Classifier'], columns='Channel', values=['P', 'R', 'F1']).reset_index()

    # For tracking vs. no tracking performance
    if FLAG_TRACKING_TABLE:
        pivot_df = pivot_df.sort_values(by=['Detector', 'Classifier'])
    elif FLAG_DETECTION_TABLE:
        pivot_df = pivot_df.sort_values(by='Detector')
    else:
        pivot_df = pivot_df.sort_values(by='Classifier')

    pivot_df = pivot_df.sort_values(by='Channel', axis=1)
    # Flatten column names
    pivot_df.columns = [f'{col[1]}_{col[0]}' if col[1] != '' else col[0] for col in pivot_df.columns]
    if FLAG_DETECTION_TABLE:
        pivot_df = reorder_df_dets(pivot_df)
    else:
        pivot_df = reorder_df(pivot_df)
    # Generate LaTeX table
    latex_table = pivot_df.to_latex(index=False)

    print(latex_table)


def main():
    metrics_generator = MetricsGenerator()
    table_dict = {}

    for channel in CHANNELS:
        if channel not in table_dict.keys():
            table_dict[channel] = {}

        for detector in DETECTORS:
            if detector not in table_dict[channel].keys():
                table_dict[channel][detector] = {}

            for feat in FEATS:
                if feat not in table_dict[channel][detector].keys():
                    table_dict[channel][detector][feat] = {}

                for classifier in CLASSIFIERS:
                    if classifier not in table_dict[channel][detector][feat].keys():
                        table_dict[channel][detector][feat][classifier] = {}

                    if MODIFIER:
                        mod_feat = f"{classifier}-{MODIFIER}"
                    else:
                        mod_feat = classifier

                    kwargs = {
                        "mod_feat": mod_feat,
                        "feats": feat,
                        "from_date": "2000_01_01" if channel != "news7-lv" else "2013_01_01"
                    }
                    res_statistics = metrics_generator.get_statistics_object(detector, channel, **kwargs)
                    df_res = res_statistics.res_df_all_info_from_to

                    print(f'{channel}-{detector}-{feat}-{classifier}:')

                    opt_correct = True if res_statistics.channel in ['CNNW', 'FOXNEWSW', 'MSNBCW'] else False
                    prec_, rec_, f1_ = metrics_generator.compute_metrics(df_res, opt_correct=opt_correct)
                    # Put information in dictionary
                    # table_dict[channel][detector][feat][classifier]['P'] = round(prec_, 2)
                    # table_dict[channel][detector][feat][classifier]['R'] = round(rec_, 3)
                    # table_dict[channel][detector][feat][classifier]['F1'] = round(f1_, 3)
                    table_dict[channel][detector][feat][classifier]['P'] = prec_
                    table_dict[channel][detector][feat][classifier]['R'] = rec_
                    table_dict[channel][detector][feat][classifier]['F1'] = f1_

                    print(f"P={round(prec_, 2)}, R={round(rec_, 2)}, F1={round(f1_, 3)}")
                    # res_latex = f"{round(prec_, 2)} & {round(rec_, 2)} & {round(f1_, 3)}"

    generate_table_from_dict(table_dict)


if __name__ == "__main__":
    CHANNELS = ["news7-lv", "hodost-lv"]
    # CHANNELS = ["CNNW", "FOXNEWSW", "MSNBCW"]
    # CHANNELS = ["CNNW", "FOXNEWSW", "MSNBCW", "news7-lv", "hodost-lv"]
    DETECTORS = ["yolo", "dfsd", "mtcnn"]
    # DETECTORS = ["yolo"]
    FEATS = ["resnetv1"]
    # CLASSIFIERS = ["fcg_average_vote"]
    CLASSIFIERS = ["fcg_average_vote", "fcg_average_centroid", "knn_3"]
    # CLASSIFIERS = ["fcg_average_vote", "fcgNT_average_vote"]
    # MODIFIER = ''
    MODIFIER = 'sf'
    FLAG_TRACKING_TABLE = False
    FLAG_DETECTION_TABLE = False

    main()
