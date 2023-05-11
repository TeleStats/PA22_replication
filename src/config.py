import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Face detection and classification for politicians in Japanese TV.')

    # Face detection mode (GT or detection for the whole video)
    parser.add_argument('mode', type=str, help='Face detector　mode. gt, demo.')
    # TV channels to test. Make it mandatory to not to mistake results or something.
    # news7-lv (NHK news 7), hodost-lv (hodo station)
    parser.add_argument('channel', type=str, help='Channel to test or detect. news7-lv, hodost-lv')

    # Paths
    parser.add_argument('--gt_path', type=str, default='data/printNews.csv', help='Ground truth path.')
    parser.add_argument('--rel_path', type=str, default='data/date_to_topics.json',
                        help='Path to file relating ground truth to videos.')
    parser.add_argument('--res_path', type=str, default='data/results', help='Results base path.')
    parser.add_argument('--models_path', type=str, default='data/faces_politicians', help='Path where the images of the individuals to detect is located.')
    parser.add_argument('--channels_path', type=str, default='video_database', help='Path where the videos are located.')

    # Flags
    parser.add_argument('--whole_video', action='store_true', help='Compute the whole video or just the shots in date_to_topics.json.')
    parser.add_argument('--save_video', action='store_true', help='Save the frames with detections / classifications overriden.')
    parser.add_argument('--use_dets', action='store_true', help='Use pre-computed detections.')
    parser.add_argument('--extract_dets', action='store_true', help='Extract the detections of the video for not having to compute them every time.')
    parser.add_argument('--extract_embs', action='store_true', help='Extract the embeddings of the video for not having to compute them every time.')
    parser.add_argument('--gender', action='store_true', help='Do gender classification instead of person classification.')
    parser.add_argument('--post_2012', action='store_true', help='Do detection and classification post year 2012.')

    # Detector and feature extractor
    parser.add_argument('--detector', type=str, default='yolo', help='Face detector. MTCNN, DFSD, YOLO-face.')
    parser.add_argument('--feats', type=str, default='resnetv1', help='Feature extractor. ResnetV1.')
    parser.add_argument('--mod_feat', type=str, default='fcg_average_vote', help='Modify feature combination. single, min_dist, all_dist, mean_dist, min_plus_mean_dist, all_plus_mean_dist')

    # Metrics
    parser.add_argument('--resolution', type=str, default='sample', help='Resolution to present the metrics. sample, segment.')
    parser.add_argument('--filter', type=int, default=0, help='Filter the results to have at least a politician detected N times per segment. +inf most restrictive, 1 less restrictive.')
    parser.add_argument('--dist', type=float, default=0.3, help='Filter the results with the required minimum distance to a politician.')
    parser.add_argument('--logprob', type=float, default=970, help='Filter the results with the required minimum log likelihood.')
    parser.add_argument('--perc_votes', type=float, default=0.7, help='Threshold of votes for a tracklet to be assigned a certain ID.')

    # Experiments
    parser.add_argument('--from_date', type=str, default='1900_01_01', help='Specific X to Y (min, max) dates for politician appearances. Use with --to. E.g. 2001_06_30 (yyyy_mm_dd)')
    parser.add_argument('--to_date', type=str, default='2100_01_01', help='Specific X to Y (min, max) dates for politician appearances. Use with --from. E.g. 2001_06_30 (yyyy_mm_dd)')
    parser.add_argument('--modifier', type=str, default='', help='Name modifier for experiments.')

    # Multiprocessing
    parser.add_argument('--process', type=int, default=0, help='Amount of processes to run in parallel.')

    # Testing
    parser.add_argument('--year', type=str, nargs='+', default=[], help='Test specific year.')
    parser.add_argument('--who', type=str, nargs='+', default=[], help='Test for specific politicians.')
    parser.add_argument('--party', type=str, nargs='+', default=[], help='Test for specific parties.')
    parser.add_argument('--top', type=int, default=0, help='Show the top-K politicians in a period.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
