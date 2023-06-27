# Script to parse the /output results to the corresponding tables
import csv
import pandas as pd
from pathlib import Path


def generate_tab3():
    # Super ad hoc
    rows_df = []
    cols = ['chn', 'det', 'res']
    flag_found_header = False

    with open(output_file, 'r') as r_file:
        reader = csv.reader(r_file, delimiter=':')
        for row in reader:
            if len(row) == 0:
                continue

            if row[0] in HEADERS:
                if row[0] == HEADERS[0]:
                    flag_found_header = True
                else:
                    flag_found_header = False
                    continue

            else:
                if flag_found_header:
                    # Missed detections experiment
                    info_config = row[0].split('-')
                    if len(info_config) < 2 or info_config[0] == "Metrics for channel":
                        continue

                    info_res = float(row[1].split('=')[1].split('%')[0])
                    row_df_ = [info_config[0], info_config[1], info_res]  # [chn, det, res]

                    rows_df.append(row_df_)

    df = pd.DataFrame(rows_df, columns=cols)
    df = df.sort_values(by='det')

    # Process df to compute overall results
    overall = df.groupby('det')['res'].mean()

    # Save to file
    out_file = PATH_SAVE / FILES_NAMES[0]

    with open(out_file, 'w', newline='\n') as w_file:
        det_prev = ''
        for (index_label, row_series) in df.iterrows():
            k_chn_ = row_series['chn']
            k_det_ = row_series['det']
            res_ = row_series['res']

            if det_prev != '' and det_prev != k_det_:
                # Write overall results after all channels
                overall_row = ['Overall', det_prev]
                row_w_ = '-'.join(overall_row) + f': {str(round(overall[det_prev], 2))}%'
                w_file.write(row_w_ + '\n')
                w_file.write('\n')

            det_prev = k_det_
            row_w = [k_chn_, k_det_]
            row_w_ = '-'.join(row_w) + f': {str(round(res_, 2))}%'
            w_file.write(row_w_ + '\n')

        # Last overall results
        # Write overall results after all channels
        overall_row = ['Overall', det_prev]
        row_w_ = '-'.join(overall_row) + f': {str(round(overall[k_det_], 2))}%'
        w_file.write(row_w_)


def df_tabs_456():
    # Super ad hoc
    rows_df = []
    cols = ['chn', 'det', 'cls', 'P', 'R', 'F1']
    flag_found_header = False

    with open(output_file, 'r') as r_file:
        reader = csv.reader(r_file, delimiter=':')
        for row in reader:
            if len(row) == 0:
                continue

            if row[0] in HEADERS:
                if row[0] == HEADERS[1]:
                    flag_found_header = True
                else:
                    flag_found_header = False
                    continue

            else:
                if flag_found_header:
                    # Missed detections experiment
                    info_config = row[0].split('-')
                    if info_config[0] == "Metrics for channel":
                        continue

                    info_res = row[1].split(',')
                    p_ = float(info_res[0].split('P=')[-1])
                    r_ = float(info_res[1].split('R=')[-1])
                    f1_ = float(info_res[2].split('F1=')[-1])
                    row_df_ = [info_config[0], info_config[1], info_config[2], p_, r_, f1_]  # [chn, det, res]

                    rows_df.append(row_df_)

    df = pd.DataFrame(rows_df, columns=cols)
    df = df.sort_values(by=['cls', 'det'])

    return df


def generate_tab4(df_all):
    # US data
    mask_us = df_all['chn'].isin(US_CHANNELS)
    df = df_all[mask_us]
    # classifiers
    mask_cls = df['cls'].isin(CLASSIFIERS[:3])  # Filter "no tracking"
    df = df[mask_cls]

    # Process df to compute overall results
    overall = df.groupby(['cls', 'det']).mean().reset_index()

    # Save to file
    out_file = PATH_SAVE / FILES_NAMES[1]

    with open(out_file, 'w', newline='\n') as w_file:
        det_prev = ''
        cls_prev = ''

        for (index_label, row_series) in df.iterrows():
            k_chn_ = row_series['chn']
            k_det_ = row_series['det']
            k_cls_ = row_series['cls']
            p_ = round(row_series['P'], 2)
            r_ = round(row_series['R'], 2)
            f1_ = round(row_series['F1'], 3)

            if det_prev != '' and det_prev != k_det_:
                # if cls_prev != '' and cls_prev != k_cls_:
                # Write overall results after all channels
                mask_overall = (overall['cls'] == cls_prev) & (overall['det'] == det_prev)
                p_overall_ = round(overall[mask_overall]['P'].to_list()[0], 2)
                r_overall_ = round(overall[mask_overall]['R'].to_list()[0], 2)
                f1_overall_ = round(overall[mask_overall]['F1'].to_list()[0], 3)
                overall_row = ['Overall', det_prev, cls_prev]
                row_w_ = '-'.join(overall_row) + f': P={p_overall_}, R={r_overall_}, F1={f1_overall_}'
                w_file.write(row_w_ + '\n')
                w_file.write('\n')

            det_prev = k_det_
            cls_prev = k_cls_

            row_w = [k_chn_, k_det_, k_cls_]
            row_w_ = '-'.join(row_w) + f': P={p_}, R={r_}, F1={f1_}'
            w_file.write(row_w_ + '\n')

        # Last overall results
        # Write overall results after all channels
        mask_overall = (overall['cls'] == cls_prev) & (overall['det'] == det_prev)
        p_overall_ = round(overall[mask_overall]['P'].to_list()[0], 2)
        r_overall_ = round(overall[mask_overall]['R'].to_list()[0], 2)
        f1_overall_ = round(overall[mask_overall]['F1'].to_list()[0], 3)
        overall_row = ['Overall', det_prev, cls_prev]
        row_w_ = '-'.join(overall_row) + f': P={p_overall_}, R={r_overall_}, F1={f1_overall_}'
        w_file.write(row_w_)


def generate_tab5(df_all):
    # JP data
    mask_jp = df_all['chn'].isin(JP_CHANNELS)
    df = df_all[mask_jp]
    # classifiers
    mask_cls = df['cls'].isin(CLASSIFIERS[:3])  # Filter "no tracking"
    df = df[mask_cls]

    # Process df to compute overall results
    overall = df.groupby(['cls', 'det']).mean().reset_index()

    # Save to file
    out_file = PATH_SAVE / FILES_NAMES[2]

    with open(out_file, 'w', newline='\n') as w_file:
        det_prev = ''
        cls_prev = ''

        for (index_label, row_series) in df.iterrows():
            k_chn_ = row_series['chn']
            k_det_ = row_series['det']
            k_cls_ = row_series['cls']
            p_ = round(row_series['P'], 2)
            r_ = round(row_series['R'], 2)
            f1_ = round(row_series['F1'], 3)

            if det_prev != '' and det_prev != k_det_:
                # if cls_prev != '' and cls_prev != k_cls_:
                # Write overall results after all channels
                mask_overall = (overall['cls'] == cls_prev) & (overall['det'] == det_prev)
                p_overall_ = round(overall[mask_overall]['P'].to_list()[0], 2)
                r_overall_ = round(overall[mask_overall]['R'].to_list()[0], 2)
                f1_overall_ = round(overall[mask_overall]['F1'].to_list()[0], 3)
                overall_row = ['Overall', det_prev, cls_prev]
                row_w_ = '-'.join(overall_row) + f': P={p_overall_}, R={r_overall_}, F1={f1_overall_}'
                w_file.write(row_w_ + '\n')
                w_file.write('\n')

            det_prev = k_det_
            cls_prev = k_cls_

            row_w = [k_chn_, k_det_, k_cls_]
            row_w_ = '-'.join(row_w) + f': P={p_}, R={r_}, F1={f1_}'
            w_file.write(row_w_ + '\n')

        # Last overall results
        # Write overall results after all channels
        mask_overall = (overall['cls'] == cls_prev) & (overall['det'] == det_prev)
        p_overall_ = round(overall[mask_overall]['P'].to_list()[0], 2)
        r_overall_ = round(overall[mask_overall]['R'].to_list()[0], 2)
        f1_overall_ = round(overall[mask_overall]['F1'].to_list()[0], 3)
        overall_row = ['Overall', det_prev, cls_prev]
        row_w_ = '-'.join(overall_row) + f': P={p_overall_}, R={r_overall_}, F1={f1_overall_}'
        w_file.write(row_w_)


def generate_tab6(df_all):
    # classifiers
    mask_cls = df_all['cls'].isin(CLASSIFIERS[2:])  # Filter for vote and "no tracking"
    df = df_all[mask_cls]
    # Reorder to be no tracking, tracking as in Table 6
    df = df.sort_values(by=['det', 'cls'], ascending=[True, False])

    # Save to file
    out_file = PATH_SAVE / FILES_NAMES[3]


    with open(out_file, 'w', newline='\n') as w_file:
        det_prev = ''

        for (index_label, row_series) in df.iterrows():
            k_chn_ = row_series['chn']
            k_det_ = row_series['det']
            k_cls_ = row_series['cls']
            p_ = round(row_series['P'], 2)
            r_ = round(row_series['R'], 2)
            f1_ = round(row_series['F1'], 3)

            if det_prev != '' and det_prev != k_det_:
                # Breakline after change of detector
                w_file.write('\n')

            det_prev = k_det_

            row_w = [k_chn_, k_det_, k_cls_]
            row_w_ = '-'.join(row_w) + f': P={p_}, R={r_}, F1={f1_}'
            w_file.write(row_w_ + '\n')


def generate_tab7():
    # Super ad hoc
    rows_df = []
    cols = ['chn', 'ID', 'res']
    flag_found_header = False
    current_election = ''
    current_chn = ''
    # Save to file
    out_file = PATH_SAVE / FILES_NAMES[4]

    with open(out_file, 'w', newline='\n') as w_file:
        with open(output_file, 'r') as r_file:
            reader = csv.reader(r_file, delimiter=':')
            for row in reader:
                if len(row) == 0:
                    continue

                if row[0] in HEADERS:
                    if row[0] in HEADERS[2:]:
                        # Build overall if current election is different than previous
                        if flag_found_header:
                            df = pd.DataFrame(rows_df, columns=cols)
                            overall = df.groupby('ID')['res'].sum().reset_index().sort_values(['res'], ascending=False)

                            # Write to file
                            prev_chn = ''
                            w_file.write(current_election + '\n')
                            for (index_label, row_series) in df.iterrows():
                                k_chn_ = row_series['chn']
                                k_id_ = row_series['ID']
                                res_ = row_series['res']

                                if prev_chn != k_chn_:
                                    # Write channel
                                    w_file.write(k_chn_ + '\n')
                                    prev_chn = k_chn_

                                row_w_ = f'{k_id_}: {res_} seconds'
                                w_file.write(row_w_ + '\n')

                            # Write overall info
                            w_file.write('Overall' + '\n')
                            for (index_label, row_series) in overall.iterrows():
                                k_id_ = row_series['ID']
                                res_ = row_series['res']
                                row_w_ = f'{k_id_}: {res_} seconds'
                                w_file.write(row_w_ + '\n')

                            w_file.write('\n')

                        # Reset values
                        rows_df = []
                        # Capture header
                        flag_found_header = True
                        current_election = row[0]
                    else:
                        flag_found_header = False
                        continue

                else:
                    if flag_found_header:
                        # Missed detections experiment
                        info_config = row[0]
                        if info_config in US_CHANNELS_W:
                            idx_chn = US_CHANNELS_W.index(info_config)
                            current_chn = US_CHANNELS[idx_chn]  # In the form of CNN instead of CNNW
                            continue

                        id_ = row[0]
                        secs_ = int(row[1].split(' ')[1])  # _N_seconds

                        row_df_ = [current_chn, id_, secs_]  # [chn, id, res]
                        rows_df.append(row_df_)

            # Last results
            if flag_found_header:
                df = pd.DataFrame(rows_df, columns=cols)
                overall = df.groupby('ID')['res'].sum().reset_index().sort_values(['res'], ascending=False)

                # Write to file
                prev_chn = ''
                w_file.write(current_election + '\n')
                for (index_label, row_series) in df.iterrows():
                    k_chn_ = row_series['chn']
                    k_id_ = row_series['ID']
                    res_ = row_series['res']

                    if prev_chn != k_chn_:
                        # Write channel
                        w_file.write(k_chn_ + '\n')
                        prev_chn = k_chn_

                    row_w_ = f'{k_id_}: {res_} seconds'
                    w_file.write(row_w_ + '\n')

                # Write overall info
                w_file.write('Overall' + '\n')
                for (index_label, row_series) in overall.iterrows():
                    k_id_ = row_series['ID']
                    res_ = row_series['res']
                    row_w_ = f'{k_id_}: {res_} seconds'
                    w_file.write(row_w_ + '\n')


def main():
    generate_tab3()
    df_456 = df_tabs_456()
    generate_tab4(df_456)
    generate_tab5(df_456)
    generate_tab6(df_456)
    generate_tab7()


if __name__ == "__main__":
    output_file = Path('output_example')
    US_CHANNELS = ['CNN', 'FOX', 'MSNBC']
    US_CHANNELS_W = ['CNNW', 'FOXNEWSW', 'MSNBCW']
    JP_CHANNELS = ['NHK', 'HODO']
    DETECTORS = ['DFSD', 'MTCNN', 'YOLO']
    CLASSIFIERS = ['KNN', 'centroid', 'vote', 'vote (no tracking)']

    # Output info
    HEADERS = ['Table 3', 'Tables 4, 5, 6', 'Republican primary (2016-02-01 to 2016-06-07)',
               'Democratic primary (2015-04-12 to 2016-06-02)', 'General election (2016-06-02 to 2016-11-08)']
    PATH_SAVE = Path('.')
    FILES_NAMES = ['tab3', 'tab4', 'tab5', 'tab6', 'tab7']

    main()
