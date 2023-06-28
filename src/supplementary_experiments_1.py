# Imports
import pandas as pd
from pathlib import Path
import random
# Data initialization
from argparse import Namespace
from types import SimpleNamespace
# Plots
import matplotlib.pyplot as plt

from statistics_producer import ResStatistics
from utils import bbox_cx_cy_w_h_to_x1_y1_x2_y2, compute_ious_bboxes


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


def compute_metrics(df_res, key_filter='Overall', opt_correct=False):
    # Compute precision, recall and F1 score based on the "corrected" FPs
    df_res_masked = df_res[df_res['ID'] == key_filter]
    tp_ovr = df_res_masked['TP'].sum()
    fn_ovr = df_res_masked['FN'].sum()
    if opt_correct:
        df_fp_sf = filter_fps_based_on_iou(df_res_masked)
        fp_ovr = df_fp_sf['FP'].sum()
    else:
        fp_ovr = df_res_masked['FP'].sum()

    prec_ovr = tp_ovr / (tp_ovr + fp_ovr)
    rec_ovr = tp_ovr / (tp_ovr + fn_ovr)
    f1_ovr = tp_ovr / (tp_ovr + (0.5 * (fp_ovr + fn_ovr)))

    return prec_ovr, rec_ovr, f1_ovr



def get_statistics_object(detector, channel, mod_feat="fcg_average_vote-sf", feats='resnetv1'):
    args = {
        'res_path': 'data/results',
        'channel': channel,
        'mode': 'train',
        'detector': detector,
        'feats': feats,
        'mod_feat': mod_feat
    }

    args = SimpleNamespace(**args)

    DATA_PATH = Path(f'data/dataset/train/{args.channel}')
    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / 'results'
    FROM_DATE = "2000_01_01"
    TO_DATE = "2100_01_01"
    res_statistics = ResStatistics(RES_PATH, mode='train', channel=args.channel, detector=args.detector,
                                   feats=args.feats, mod_feat=args.mod_feat, args_from_outside=args,
                                   from_date_str=FROM_DATE, to_date_str=TO_DATE)
    df_res = res_statistics.res_df_all_info_from_to

    return res_statistics


def generate_threshs_df(channel, opt_correct=False):
    cols_df = ['Threshold', 'P', 'R', 'F1']
    rows_df = []

    for vv, ff in zip(perc_votes_list, mod_feat_list):
        mod_feat = ff + '-sf'
        res_statistics = get_statistics_object('yolo', channel, mod_feat=mod_feat)
        df_res = res_statistics.res_df_all_info_from_to
        prec_ov, rec_ov, f1_ov = compute_metrics(df_res, key_filter='Overall', opt_correct=opt_correct)
        df_met = res_statistics.compute_metrics(df_res)
        df_ov = df_met[df_met["ID"] == "Overall"]
        # print(f"{vv}: P={round(prec_ov, 2)}, R={round(rec_ov, 2)}, F1={round(f1_ov, 3)}")
        # print(f"{vv},{round(prec_ov, 2)},{round(rec_ov, 2)},{round(f1_ov, 3)}")
        row_df_ = [vv, round(prec_ov, 2), round(rec_ov, 2), round(f1_ov, 3)]
        rows_df.append(row_df_)

    df = pd.DataFrame(rows_df, columns=cols_df)
    return df

def sm_fig1(dict_df):
    # Fig1 (Precision)
    for k in dict_df.keys():
        # Plot the data
        plt.plot(dict_df[k]['df']['Threshold'].values, dict_df[k]['df']['P'].values, marker=dict_df[k]['marker'], markersize=2, color=dict_df[k]['color'], label=k)

    plt.xlabel('Voting threshold')
    # Axis things
    plt.ylim(0, 1.01)
    # Rotate and align the x-axis labels
    plt.title('Precision')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.legend()
    # plt.show()
    out_file = PATH_SAVE / 'SM_fig1.pdf'
    plt.savefig(out_file, dpi=300)
    plt.close()


def sm_fig2(dict_df):
    # Fig1 (Precision)
    for k in dict_df.keys():
        # Plot the data
        plt.plot(dict_df[k]['df']['Threshold'].values, dict_df[k]['df']['R'].values, marker=dict_df[k]['marker'], markersize=2, color=dict_df[k]['color'], label=k)

    plt.xlabel('Voting threshold')
    # Axis things
    plt.ylim(0, 1.01)
    # Rotate and align the x-axis labels
    plt.title('Recall')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.legend()
    # plt.show()
    out_file = PATH_SAVE / 'SM_fig2.pdf'
    plt.savefig(out_file, dpi=300)
    plt.close()


def sm_fig3(dict_df):
    # Fig3 (F1-score)
    for k in dict_df.keys():
        # Plot the data
        plt.plot(dict_df[k]['df']['Threshold'].values, dict_df[k]['df']['F1'].values, marker=dict_df[k]['marker'], markersize=2, color=dict_df[k]['color'], label=k)

    plt.xlabel('Voting threshold')
    # Axis things
    plt.ylim(0, 1.01)
    # Rotate and align the x-axis labels
    plt.title('F1-score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.legend()
    # plt.show()
    out_file = PATH_SAVE / 'SM_fig3.pdf'
    plt.savefig(out_file, dpi=300)
    plt.close()


def sm_tab1(dict_df):
    out_file = PATH_SAVE / 'SM_tab1.txt'

    with open(out_file, 'w', newline='\n') as w_file:
        for k in US_CHANNELS:
            df = dict_df[k]['df']

            w_file.write(f'{k}: \n')
            w_file.write(f'Threshold   P    R    F1\n')

            for (index_label, row_series) in df.iterrows():
                thresh_ = row_series['Threshold']
                p_ = round(row_series['P'], 2)
                r_ = round(row_series['R'], 2)
                f1_ = round(row_series['F1'], 3)

                row_w_ = f'{thresh_:<12}{p_:<5}{r_:<5}{f1_:<5}'
                w_file.write(row_w_ + '\n')

            w_file.write('\n')


def sm_tab2(dict_df):
    out_file = PATH_SAVE / 'SM_tab2.txt'

    with open(out_file, 'w', newline='\n') as w_file:
        for k in JP_CHANNELS:
            df = dict_df[k]['df']

            w_file.write(f'{k}: \n')
            w_file.write(f'Threshold   P    R    F1\n')

            for (index_label, row_series) in df.iterrows():
                thresh_ = row_series['Threshold']
                p_ = round(row_series['P'], 2)
                r_ = round(row_series['R'], 2)
                f1_ = round(row_series['F1'], 3)

                row_w_ = f'{thresh_:<12}{p_:<5}{r_:<5}{f1_:<5}'
                w_file.write(row_w_ + '\n')

            w_file.write('\n')


def main():
    dict_df = dict()

    # US data
    for chn, chn_re, color in zip(US_CHANNELS_W, US_CHANNELS, US_COLORS):
        dict_df[chn_re] = dict()
        dict_df[chn_re]['df'] = generate_threshs_df(chn, opt_correct=True)
        dict_df[chn_re]['color'] = color
        dict_df[chn_re]['marker'] = None

    # JP data
    for chn, chn_re, color in zip(JP_CHANNELS_W, JP_CHANNELS, JP_COLORS):
        dict_df[chn_re] = dict()
        dict_df[chn_re]['df'] = generate_threshs_df(chn, opt_correct=False)
        dict_df[chn_re]['color'] = color
        dict_df[chn_re]['marker'] = 'o'

    # Generate figures
    sm_fig1(dict_df)
    sm_fig2(dict_df)
    sm_fig3(dict_df)

    # Generate tables
    sm_tab1(dict_df)
    sm_tab2(dict_df)


if __name__ == "__main__":
    US_CHANNELS = ['CNN', 'FOX', 'MSNBC']
    US_CHANNELS_W = ['CNNW', 'FOXNEWSW', 'MSNBCW']
    US_COLORS = ['#f0e929', '#519e1e', '#e07b22']
    JP_CHANNELS = ['NHK', 'HODO']
    JP_CHANNELS_W = ['news7-lv', 'hodost-lv']
    JP_COLORS = ['#3264a8', '#e80c0c']

    perc_votes_list = ["0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5", "0.55", "0.6", "0.65", "0.7",
                       "0.75", "0.8", "0.85", "0.9", "0.95", "0.99"]
    mod_feat_list = ["fcg_average_vote_01", "fcg_average_vote_015", "fcg_average_vote_02", "fcg_average_vote_025",
                     "fcg_average_vote_03", "fcg_average_vote_035", "fcg_average_vote_04", "fcg_average_vote_045",
                     "fcg_average_vote_05", "fcg_average_vote_055", "fcg_average_vote_06", "fcg_average_vote_065",
                     "fcg_average_vote_07", "fcg_average_vote_075", "fcg_average_vote_08", "fcg_average_vote_085",
                     "fcg_average_vote_09", "fcg_average_vote_095", "fcg_average_vote_099"]
    modifier = 'sf'

    PATH_SAVE = Path('.')

    main()
