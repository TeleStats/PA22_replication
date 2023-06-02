# File to extract data for the analysis paper (Japan TV)
import argparse
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

from statistics_producer import ResStatistics
from utils import POLITICIANS, PRIME_MINISTERS, PM_DATES_MONTH, PM_DATES_DAY, PARTIES, OP_LEADERS, REPRESENTATIVES_DATES_DAY
from utils import convert_str_to_date, cm_to_inch


class Analyzer:
    def __init__(self, **kwargs):
        # Attributes
        self.res_dict = dict()
        self.from_date = kwargs.get('from_date', "2000_01_01")
        self.to_date = kwargs.get('to_date', "2100_01_01")
        self.flag_uncertainty = kwargs.get('flag_uncertainty', False)

    @staticmethod
    def _read_metric_results_(dataset_path: Path, dataset_id: str) -> ResStatistics:
        # Read train results to be able to generate "corrected" TV seconds
        train_res_path = dataset_path.parents[2] / 'train' / 'results'  # data/results/news7-lv/train/results
        # train_res_path = dataset_path.parents[2] / 'train' / 'old' / 'results'  # data/results/news7-lv/train/results
        res_statistics_train = ResStatistics(train_res_path, channel=dataset_id, mode='train', from_date_str=FROM_DATE,
                                             detector=DETECTOR, feats=FEATS, mod_feat=CLASSIFIER, to_date_str=TO_DATE)
        # res_statistics_train = ResStatistics(train_res_path, channel=dataset_id, mode='train', from_date_str=FROM_DATE,
        #                                      detector=DETECTOR, feats=FEATS, mod_feat=CLASSIFIER, to_date_str=TO_DATE,
        #                                      mod_metrics="sf")

        res_statistics_train.print_prec_rec()
        return res_statistics_train

    def add_res_dataset(self, dataset_path: Path, dataset_id: str) -> None:
        # Read the results from a dataset and include them in the dataset dictionary
        self.res_dict[dataset_id] = dict()
        self.res_dict[dataset_id]['data_path'] = dataset_path
        self.res_dict[dataset_id]['statistics'] = ResStatistics(dataset_path.parent, channel=dataset_id, mode=MODE,
                                                                detector=DETECTOR, feats=FEATS, mod_feat=CLASSIFIER,
                                                                from_date_str=FROM_DATE, to_date_str=TO_DATE)

        if self.flag_uncertainty:
            # Read train results to be able to generate "corrected" TV seconds
            res_statistics_train = self._read_metric_results_(dataset_path, dataset_id)
            self.res_dict[dataset_id]['statistics'].gt_correction_factor = self.res_dict[dataset_id]['statistics'].compute_correction_factor(
                res_statistics_train.res_df_all_info_from_to)
            print(f"Correction factors\n{self.res_dict[dataset_id]['statistics'].gt_correction_factor}")

    def set_dates_to_analyse(self, from_date_str, to_date_str):
        for dataset_id in self.res_dict.keys():
            self.res_dict[dataset_id]['statistics'].filter_df_from_to(from_date_str, to_date_str)

    def reset_dates_to_analyse(self):
        for dataset_id in self.res_dict.keys():
            self.res_dict[dataset_id]['statistics'].filter_df_from_to(FROM_DATE, TO_DATE)

    @staticmethod
    def _init_plot_(w_cm=20, h_cm=10):
        sns.set_theme(style="white")
        fig, ax = plt.subplots()
        fig.set_size_inches(cm_to_inch(w_cm), cm_to_inch(h_cm))
        return fig, ax

    @staticmethod
    def _purge_legend_(ax):
        handles, labels = ax.get_legend_handles_labels()
        labels_legend = set(labels) & (set(POLITICIANS) | set(PARTIES))
        idxs = [labels.index(pol) for pol in labels_legend]
        if len(idxs) == 0:
            return -1, -1
        else:
            handles_list = [handles[idx] for idx in idxs]
            labels_list = [labels[idx] for idx in idxs]
            return handles_list, labels_list

    @staticmethod
    def _format_xaxis_(ax):
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/tick-locators.html
        majors = [(loc, ax_tick) for loc, ax_tick in enumerate(ax.xaxis.get_major_formatter().seq) if ax_tick != ' ']
        major_axis = [major[0] for major in majors]
        major_str = [major[1] for major in majors]
        ax.xaxis.set_major_locator(ticker.FixedLocator(major_axis))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(major_str))
        ax.tick_params(axis='x', bottom=True)

    def format_plot(self, ax):
        self._format_xaxis_(ax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        handles_list, labels_list = self._purge_legend_(ax)
        if handles_list == -1:
            ax.legend().remove()
        else:
            ax.legend(handles=handles_list, labels=labels_list, loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)

    @staticmethod
    def filter_df_by_id(df: pd.DataFrame, id_list: list) -> pd.DataFrame:
        return df[df['ID'].isin(id_list)] if id_list is not None else df

    @staticmethod
    def filter_df_by_pm(df: pd.DataFrame) -> pd.DataFrame:
        # Filter the df to get only the PM for each period
        pminister_dates = [convert_str_to_date(x) for x in PM_DATES_DAY]

        df_list = []
        for i, (pm_id, pm_date) in enumerate(zip(PRIME_MINISTERS, pminister_dates)):
            mask_pm = df['ID'] == pm_id
            df_pm = df[mask_pm]

            if i == 0:
                # First case: the date has to be less or equal than the one we have annotated
                mask_date = df_pm['date'] <= pm_date
            elif i < len(PRIME_MINISTERS):
                # Second case: the date has to be between mandates
                mask_date = (df_pm['date'] > pminister_dates[i-1]) & (df_pm['date'] <= pm_date)
            else:
                # Third case: The date has to be greater than the last election
                mask_date = df_pm['date'] > pm_date

            df_pm_date = df_pm[mask_date]
            df_list.append(df_pm_date)

        df_res = pd.concat(df_list)

        return df_res

    @staticmethod
    def fuse_df_list(df_list: list) -> pd.DataFrame:
        df_res = pd.concat(df_list)
        # Remove hours and keep only year-month-day
        df_res['date'] = df_res['date'].dt.date
        df_res = df_res.groupby(by=['date', 'ID']).sum().reset_index()

        return df_res

    @staticmethod
    def fill_with_zeros(df: pd.DataFrame) -> pd.DataFrame:
        # Get a dataframe and fill the information for dates where we don't have information
        # From: https://stackoverflow.com/questions/59875404/fill-in-missing-dates-with-0-zero-in-pandas
        dtr = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        df_daterange = pd.DataFrame(data=dtr, columns=['date'])
        df_daterange['date'] = df_daterange['date'].dt.date
        # Mask the dates and only use the ones that are not in "df"
        mask_date = ~df_daterange.isin(df['date'].to_list())
        df_daterange = df_daterange[mask_date]
        # Concatenate the two dataframes and return the dataframe
        df_res = pd.concat([df, df_daterange]).sort_values(by=['date'])
        df_res = df_res.fillna(0)
        return df_res

    def get_actors_screen_time(self, actors_list: list = None) -> pd.DataFrame:
        # Return screen time for a list of specific key actors
        # Get a dataframe returning the daily screen time in the form of:
        # Date (day), ID (PM), News7, Hodost
        df_pm_channels = []
        for dataset_id in self.res_dict.keys():
            df = self.res_dict[dataset_id]['statistics'].res_df_all_info_from_to
            df_pm = self.filter_df_by_id(df, actors_list)
            df_pm_channel = df_pm.groupby(by=['date', 'ID'])['TP'].sum().reset_index().rename(columns={'TP': f'{dataset_id}'})
            df_pm_channels.append(df_pm_channel)

        df_res = self.fuse_df_list(df_pm_channels)

        return df_res

    def get_incumbent_pm_screen_time(self) -> pd.DataFrame:
        # Return screen time per PM only when they were PM
        # Get a dataframe returning the daily screen time in the form of:
        # Date (day), ID (PM), News7, Hodost
        df_pm_channels = []
        for dataset_id in self.res_dict.keys():
            df = self.res_dict[dataset_id]['statistics'].res_df_all_info_from_to
            df_pm = self.filter_df_by_pm(df)
            df_pm_channel = df_pm.groupby(by=['date', 'ID'])['TP'].sum().reset_index().rename(columns={'TP': f'{dataset_id}'})
            df_pm_channels.append(df_pm_channel)

        df_res = self.fuse_df_list(df_pm_channels)

        return df_res

    @staticmethod
    def append_to_df(df, rows, var_name='TP_d', dummy=None):
        df_new = df.copy()
        new_rows = []
        for row in rows:
            new_rows.append(pd.DataFrame(data=[[dummy, row, var_name, -1000]], columns=df.keys().to_list()))

        df_new = df_new.append(new_rows, ignore_index=True)
        return df_new

    # def scale_values_with_correction_factor(self, df_plot, dataset_id):
    #     df_plot_res = df_plot.copy()
    #     pol_ids_correction = self.res_dict[dataset_id]['statistics'].gt_correction_factor['ID'].unique()
    #     pol_ids_plot = df_plot['ID'].unique()
    #
    #     for pol_id in pol_ids_correction:
    #         if pol_id not in pol_ids_plot:
    #             continue
    #
    #         mask_id_corr = self.res_dict[dataset_id]['statistics'].gt_correction_factor['ID'] == pol_id
    #         mask_id_plot = df_plot_res['ID'] == pol_id
    #
    #         correction_factor = self.res_dict[dataset_id]['statistics'].gt_correction_factor[mask_id_corr]['c_factor'].item()
    #         df_plot_res[mask_id_plot]['value'] *= correction_factor
    #
    #     return df_plot_res

    def plot_channels_times(self, **kwargs):
        df_plot_list = []  # Dataframes for all datasets
        # fig, ax = self._init_plot_()
        # Read and prepare dataframes for all channels
        date_res = kwargs.get('date_res', 'month')  # month or day
        name_mod = kwargs.get('name_mod', None)
        smooth_method = kwargs.get('smooth_method', 'gauss')
        win_size = kwargs.get('win_size', None)
        span = kwargs.get('span', None)
        sigma = kwargs.get('sigma', None)
        flag_perc = kwargs.get('flag_perc', False)
        flag_smooth = kwargs.get('flag_smooth', False)
        flag_save = kwargs.get('flag_save', False)

        for dataset_id in self.res_dict.keys():
            if self.res_dict[dataset_id]['statistics'].res_df_all_info_from_to.empty:
                print(f'{dataset_id} is empty for {FROM_DATE} to {TO_DATE}')
                continue

            df_plot = self.res_dict[dataset_id]['statistics'].generate_df_plot(**kwargs)
            df_plot_list.append(df_plot)

        # Plot the dataframes in the same temporal scale
        dates_in_plot = []
        for df_plot in df_plot_list:
            dates_in_plot += df_plot['year'].to_list()

        dates_in_plot = sorted(list(set(dates_in_plot)))  # Get unique dates for all dataframes

        for dataset_id, df_plot in zip(self.res_dict.keys(), df_plot_list):
            df_plot = self.append_to_df(df_plot, dates_in_plot, dummy='.')
            df_plot = df_plot.sort_values(by=['year', df_plot.columns[0]])  # columns[0] should be ID or party
            # Name to save pdf
            count_mag = 'secs'
            if flag_perc:
                count_mag = 'perc'

            smooth_val = sigma if smooth_method == 'gauss' else span
            common_name = f'{str(dates_in_plot[0])}-{str(dates_in_plot[-1])}.pdf'
            if not flag_save:
                save_name = None
            elif not flag_smooth:
                save_name = f'{dataset_id}-{date_res}-{count_mag}-{name_mod}-{common_name}'
            elif name_mod is not None:
                save_name = f'{dataset_id}-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{name_mod}-{common_name}'
            else:
                save_name = f'{dataset_id}-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{common_name}'

            # Plot here in the same scale for all datasets
            if date_res == 'month':
                self.res_dict[dataset_id]['statistics'].plot_screen_time_timeline_month(
                    df_plot=df_plot, save_name=save_name, **kwargs)
            elif date_res == 'day':
                self.res_dict[dataset_id]['statistics'].plot_screen_time_timeline_day(
                    df_plot=df_plot, save_name=save_name, **kwargs)

    def plot_gov_opp_screen_ratio(self, **kwargs):
        df_plot_list = []  # Dataframes for all datasets
        kwargs['politicians'] = []  # The filtering happens inside the functions
        kwargs['max_value'] = 100  # The filtering happens inside the functions
        # fig, ax = self._init_plot_()
        # Read and prepare dataframes for all channels
        date_res = kwargs.get('date_res', 'month')  # month or day
        name_mod = kwargs.get('name_mod', None)
        smooth_method = kwargs.get('smooth_method', 'gauss')
        win_size = kwargs.get('win_size', None)
        span = kwargs.get('span', None)
        sigma = kwargs.get('sigma', None)
        flag_perc = kwargs.get('flag_perc', False)
        flag_smooth = kwargs.get('flag_smooth', False)
        flag_save = kwargs.get('flag_save', False)

        for dataset_id in self.res_dict.keys():
            if self.res_dict[dataset_id]['statistics'].res_df_all_info_from_to.empty:
                print(f'{dataset_id} is empty for {FROM_DATE} to {TO_DATE}')
                continue

            df_plot = self.res_dict[dataset_id]['statistics'].generate_df_plot(**kwargs)
            df_plot_list.append(df_plot)

        # Plot the dataframes in the same temporal scale
        dates_in_plot = []
        for df_plot in df_plot_list:
            dates_in_plot += df_plot['year'].to_list()

        dates_in_plot = sorted(list(set(dates_in_plot)))  # Get unique dates for all dataframes

        for dataset_id, df_plot in zip(self.res_dict.keys(), df_plot_list):
            df_plot = self.append_to_df(df_plot, dates_in_plot, dummy='.')
            df_plot = df_plot.sort_values(by=['year', df_plot.columns[0]])  # columns[0] should be ID or party
            # Name to save pdf
            count_mag = 'secs'
            if flag_perc:
                count_mag = 'perc'

            smooth_val = sigma if smooth_method == 'gauss' else span
            # common_name = f'{str(dates_in_plot[0])}-{str(dates_in_plot[-1])}_ratio.pdf'
            common_name = f'ratio.pdf'
            chann_ = ''
            if dataset_id == 'news7-lv':
                chann_ = 'NHK'
            else:
                chann_ = 'HODO'

            if not flag_save:
                save_name = None
            elif not flag_smooth:
                save_name = f'fig7a-{chann_}-{smooth_val}-{common_name}'
                # save_name = f'{dataset_id}-{date_res}-{count_mag}-{name_mod}-{common_name}'
            elif name_mod is not None:
                save_name = f'fig7a-{chann_}-{smooth_val}-{name_mod}-{common_name}'
                # save_name = f'{dataset_id}-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{name_mod}-{common_name}'
            else:
                save_name = f'fig7a-{chann_}-{smooth_val}-{common_name}'
                # save_name = f'{dataset_id}-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{common_name}'

            # Plot here in the same scale for all datasets
            self.res_dict[dataset_id]['statistics'].plot_pm_opp(df_plot=df_plot, save_name=save_name, **kwargs)

    def plot_gov_opp_screen_ratio_nhk_vs_hodo(self, **kwargs):
        df_plot_list = []  # Dataframes for all datasets
        kwargs['politicians'] = []  # The filtering happens inside the functions
        # kwargs['max_value'] = 100  # The filtering happens inside the functions
        kwargs['max_value'] = None  # The filtering happens inside the functions
        # fig, ax = self._init_plot_()
        # Read and prepare dataframes for all channels
        date_res = kwargs.get('date_res', 'month')  # month or day
        name_mod = kwargs.get('name_mod', None)
        smooth_method = kwargs.get('smooth_method', 'gauss')
        win_size = kwargs.get('win_size', None)
        span = kwargs.get('span', None)
        sigma = kwargs.get('sigma', None)
        flag_perc = kwargs.get('flag_perc', False)
        flag_smooth = kwargs.get('flag_smooth', False)
        flag_save = kwargs.get('flag_save', False)
        kwargs['max_value'] = 3.5
        # kwargs['max_value'] = 200

        for dataset_id in self.res_dict.keys():
            if self.res_dict[dataset_id]['statistics'].res_df_all_info_from_to.empty:
                print(f'{dataset_id} is empty for {FROM_DATE} to {TO_DATE}')
                continue

            df_plot = self.res_dict[dataset_id]['statistics'].generate_df_plot(**kwargs)
            df_plot_list.append(df_plot)

        # Plot the dataframes in the same temporal scale
        dates_in_plot = []
        for df_plot in df_plot_list:
            dates_in_plot += df_plot['year'].to_list()

        dates_in_plot = sorted(list(set(dates_in_plot)))  # Get unique dates for all dataframes
        df_plot_list_union = []

        for dataset_id, df_plot_ in zip(self.res_dict.keys(), df_plot_list):
            df_plot_ = self.append_to_df(df_plot_, dates_in_plot, dummy='.')
            df_plot_ = df_plot_.sort_values(by=['year', df_plot_.columns[0]])  # columns[0] should be ID or party
            df_plot_["dataset"] = dataset_id
            df_plot_list_union.append(df_plot_)

        df_plot_relation = df_plot_.copy()
        df_plot_relation['value'] = df_plot_list_union[0]['value'] / df_plot_list_union[1]['value']
        # df_plot_relation['value'] = df_plot_list_union[0]['value'] - df_plot_list_union[1]['value']
        # df_plot_relation['value'] = df_plot_list_union[0]['value'] / (df_plot_list_union[0]['value'] + df_plot_list_union[1]['value'])

        # Name to save pdf
        count_mag = 'secs'
        if flag_perc:
            count_mag = 'perc'

        smooth_val = sigma if smooth_method == 'gauss' else span
        # common_name = f'{str(dates_in_plot[0])}-{str(dates_in_plot[-1])}_ratio.pdf'
        common_name = f'ratio.pdf'
        if not flag_save:
            save_name = None
        elif not flag_smooth:
            save_name = f'fig7b-NHKvsHODO-{name_mod}-{common_name}'
            # save_name = f'NHKvsHODO-{date_res}-{count_mag}-{name_mod}-{common_name}'
        elif name_mod is not None:
            save_name = f'fig7b-NHKvsHODO-{smooth_val}-{name_mod}-{common_name}'
            # save_name = f'NHKvsHODO-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{name_mod}-{common_name}'
        else:
            save_name = f'fig7b-NHKvsHODO-{smooth_val}-{common_name}'
            # save_name = f'NHKvsHODO-{date_res}-{smooth_method}-{smooth_val}-{count_mag}-{common_name}'

        # Plot here in the same scale for all datasets
        # self.res_dict['news7-lv']['statistics'].plot_pm_opp(df_plot=df_plot_relation, save_name=save_name, **kwargs)
        self.res_dict['news7-lv']['statistics'].plot_pm_opp(df_plot=df_plot_list_union, flag_tv1_vs_tv2=True, save_name=save_name, **kwargs)


def generate_csv_pm_channels(analyzer: Analyzer):
    # Generate csv file with PM info for NHK and HODO station
    # df_pm_screen_time = analyzer.get_pm_screen_time()
    df_pm_screen_time = analyzer.get_incumbent_pm_screen_time()
    df_pm_screen_time = analyzer.fill_with_zeros(df_pm_screen_time)
    df_pm_screen_time.to_csv(SAVE_PATH, index=False)
    print(df_pm_screen_time.head())


def plot_zoomed_by_date(analyzer: Analyzer, date_str: str, prev_weeks: int = 2, **kwargs):
    # Function to provide plots zoomed in a certain date for fine-grained analysis
    days_prev = datetime.timedelta(prev_weeks * 7)
    days_margin = datetime.timedelta(5)
    from_date = convert_str_to_date(date_str) - days_prev
    to_date = convert_str_to_date(date_str) + days_margin
    from_date_str = from_date.strftime("%Y_%m_%d")
    to_date_str = to_date.strftime("%Y_%m_%d")

    # Filter around the date to analyse
    analyzer.set_dates_to_analyse(from_date_str, to_date_str)
    # analyzer.set_dates_to_analyse(FROM_DATE, TO_DATE)

    # Generate plots
    analyzer.plot_channels_times(**kwargs)


def main_big(analyzer, **kwargs):
    # Plot in a big magnitude (20 years~)
    plot_args = kwargs.copy()
    # analyzer.plot_channels_times(**plot_args)
    # Gov. vs. opposition ratio plot
    plot_args['flag_prev_pm'] = False
    plot_args['flag_party'] = False
    analyzer.plot_gov_opp_screen_ratio(**plot_args)
    analyzer.plot_gov_opp_screen_ratio_nhk_vs_hodo(**plot_args)


def main_fine(analyzer, **kwargs):
    # dates to be analysed
    date_list = REPRESENTATIVES_DATES_DAY
    # date_list = [REPRESENTATIVES_DATES_DAY[-1]]
    # Plot in a smaller magnitude for fine-grained analysis
    plot_args = kwargs.copy()
    # Set attributes for this kind of plot
    plot_args['flag_pm'] = False
    plot_args['topk'] = 6
    plot_args['flag_perc'] = False
    plot_args['flag_smooth'] = False
    plot_args['sigma'] = 2
    plot_args['flag_uncertainty'] = False
    plot_args['max_value'] = 2500
    plot_args['name_mod'] = 'election_repr'

    for date_str in date_list:
        plot_zoomed_by_date(analyzer, date_str, prev_weeks=6, **plot_args)


def main_print_screen_time(analyzer, dataset_id, k_pols=None):
    # Show screen time for specified politicians in a certain dataset
    print(dataset_id)
    if k_pols == None:
        keys_pol = analyzer.res_dict[dataset_id]['statistics'].res_corrected['ID'].unique()
    else:
        keys_pol = k_pols

    for k_pol in keys_pol:
        mask_pol = analyzer.res_dict[dataset_id]['statistics'].res_corrected['ID'] == k_pol
        df_pol = analyzer.res_dict[dataset_id]['statistics'].res_corrected[mask_pol]
        print(f"{k_pol}: {df_pol['TP_d'].sum()} seconds")

def analysis_init(sigma=15, flag_unc=False):
    flag_uncertainty = flag_unc  # De moment aqui, posar dins la main function
    analyzer = Analyzer(flag_uncertainty=flag_uncertainty)
    # Add NHK news7 and hodo station for analysis
    for channel in CHANNELS:
        # print(channel)
        dataset_path = Path(ROOT_RES_PATH) / channel / MODE / 'results' / METHOD_ID
        analyzer.add_res_dataset(dataset_path, channel)

    # generate_csv_pm_channels(analyzer)
    # Args for generating plots
    plot_args = {'politicians': POLS_INTEREST,
                 'parties': [],
                 'topk': 0,
                 'date_res': 'day',  # month or day
                 'win_size': 90,  # window size for rolling average (e.g. 2 for month and 5 for day)
                 'alpha': 0.3,
                 'span': 180,  # 30 sigma corresponds to ~180 days (3 sigma) --> 90 days in the past for exp
                 'sigma': sigma,  # 15 for day, 1(?) per month
                 'smooth_method': 'gauss',  # gauss, exp, rolling
                 'name_mod': 'gov_opp',  # gov_opp_ind_legend
                 'max_value': 9,  # 7000 for secs and 10 for perc is fine
                 'flag_pm': False,
                 'flag_op': False,
                 'flag_gov_team': True,  # To plot screen time of politicians within the government
                 'flag_op_team': True,
                 'flag_gov_op_color': True,  # To plot government and opposition with a single color. # CAVEAT: if False, screen_ratio is PM vs. OP leader
                 'flag_uncertainty': flag_uncertainty,
                 'flag_prev_pm': False,
                 'flag_party': False,
                 'flag_perc': True,
                 'flag_smooth': True,
                 'flag_dates': True,
                 'flag_legend': True,
                 'flag_save': True}

    return analyzer, plot_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis for replicating table 7 and figure 7')
    # Face detection mode (GT or detection for the whole video)
    parser.add_argument('exp', type=str, help='Which experiments to run (tab7, fig7)')
    args = parser.parse_args()

    # Paths and stuff
    ROOT_RES_PATH = 'data/results'
    # Variables
    MODE = 'demo'
    DETECTOR = 'yolo'
    FEATS = 'resnetv1'
    CLASSIFIER = 'fcg_average_vote'
    METHOD_ID = f"{DETECTOR}-{FEATS}-{CLASSIFIER}"

    # Experiments
    # Table 7
    if args.exp == 'tab7':
        CHANNELS = ['CNNW', 'FOXNEWSW', 'MSNBCW']
        # CHANNELS = ['FOXNEWSW', 'MSNBCW']
        # 1. Republican primary
        FROM_DATE = "2016_02_01"
        TO_DATE = "2016_06_07"
        POLS_INTEREST = ['Donald_Trump', 'Ted_Cruz', 'Marco_Rubio', 'John_Kasich']
        # Initialize analyzer depending on options
        print("Republican primary (2016-02-01 to 2016-06-07)")
        analyzer_, _ = analysis_init()
        for channel in CHANNELS:
            main_print_screen_time(analyzer_, channel, POLS_INTEREST)
        print('\n')

        # 2. Democratic primary
        FROM_DATE = "2015_04_12"
        TO_DATE = "2016_06_02"
        POLS_INTEREST = ['Hillary_Clinton', 'Bernie_Sanders']
        # Initialize analyzer depending on options
        print("Democratic primary (2015-04-12 to 2016-06-02)")
        analyzer_, _ = analysis_init()
        for channel in CHANNELS:
            main_print_screen_time(analyzer_, channel, POLS_INTEREST)
        print('\n')

        # 3. General election
        FROM_DATE = "2016_06_02"
        TO_DATE = "2016_11_08"
        POLS_INTEREST = ['Donald_Trump', 'Hillary_Clinton']
        # Initialize analyzer depending on options
        print("General election (2016-06-02 to 2016-11-08)")
        analyzer_, _ = analysis_init()
        for channel in CHANNELS:
            main_print_screen_time(analyzer_, channel, POLS_INTEREST)
        print('\n')

    elif args.exp == 'fig7':
        FROM_DATE = "2014_01_05"
        TO_DATE = "2100_01_01"
        POLS_INTEREST = []
        # 1. NHK and HODO (15-days)
        CHANNELS = ['news7-lv', 'hodost-lv']
        analyzer_, plot_args_ = analysis_init(sigma=15, flag_unc=True)
        main_big(analyzer_, **plot_args_)
        # 6. NHK and HODO (90-days)
        CHANNELS = ['news7-lv', 'hodost-lv']
        analyzer_, plot_args_ = analysis_init(sigma=90, flag_unc=True)
        main_big(analyzer_, **plot_args_)



    # main_big(analyzer_, **plot_args_)
    # main_fine(analyzer_, **plot_args_)


    #
    # # CHANNELS = ['news7-lv', 'hodost-lv']
    # CHANNELS = ['MSNBCW']
    # # POLS_INTEREST = []  # [] for all
    # POLS_INTEREST = ['Hillary_Clinton', 'Bernie_Sanders']  # DEMS
    # # POLS_INTEREST = ['Carly_Fiorina', 'Mike_Huckabee', 'Jim_Gilmore', 'Rick_Santorum']  # REPS: Donald Trump, Ted Cruz, Marco Rubio, John Kasich
    # # POLS_INTEREST = ['Donald_Trump', 'Hillary_Clinton']  # REPS: Donald Trump, Ted Cruz, Marco Rubio, John Kasich
    # # POLS_INTEREST = ['Donald_Trump', 'Ted_Cruz', 'Marco_Rubio', 'John_Kasich']  # REPS: Donald Trump, Ted Cruz, Marco Rubio, John Kasich
    # # SAVE_PATH = 'results_CNN_RepPrimary.csv'
    # SAVE_PATH = 'results_incumbent_pm_nhk.csv'
    # # Variables
    # MODE = 'demo'
    # DETECTOR = 'yolo'
    # FEATS = 'resnetv1'
    # CLASSIFIER = 'fcg_average_vote'
    # METHOD_ID = f"{DETECTOR}-{FEATS}-{CLASSIFIER}"
    # # FROM_DATE = "2014_01_05"
    # # TO_DATE = "2100_01_01"
    # # Tests
    # # FROM_DATE = "2016_09_06"
    # # TO_DATE = "2016_09_06"
    # #US general elections
    # # FROM_DATE = "2016_06_02"
    # # TO_DATE = "2016_11_08"
    # #Dem. primaries
    # FROM_DATE = "2015_04_12"
    # TO_DATE = "2016_06_02"
    # # Rep. primaries
    # # FROM_DATE = "2016_02_01"
    # # TO_DATE = "2016_06_07"
    # # Rep. primaries 2
    # # FROM_DATE = "2015_03_23"
    # # TO_DATE = "2016_05_26"
