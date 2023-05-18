# 2. Do ROC / plot curves from results
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns

from config import get_parser
from utils import cm_to_inch, modify_palette, get_politician_palette, convert_str_to_date, filter_list_by_date
from utils import POLITICIANS, PRIME_MINISTERS, PM_DATES_MONTH, PM_DATES_DAY, PARTIES, OP_LEADERS, OP_DATES_MONTH, OP_DATES_DAY
from utils import REPRESENTATIVES_DATES_MONTH, REPRESENTATIVES_DATES_DAY, COUNCILLORS_DATES_MONTH, COUNCILLORS_DATES_DAY
from utils import LDP_DATES_MONTH, LDP_DATES_DAY, PARLIAMENT_DISSOLUTION_DAY, PARL_CAMPAIGN_START_DAY, COUNC_CAMPAIGN_START_DAY
from utils import US_ACTORS

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class ResStatistics:
    def __init__(self, res_path, channel, mode='gt', **kwargs):
        # Results data
        self.res_path = res_path
        self.res_df_all_info = None
        self.res_df_all_info_from_to = None
        self.gt_correction_factor = kwargs.get('gt_correction_factor', None)  # In case we have it precomputed for demo files
        self.res_corrected = None
        self.train_res_df_all_info_from_to = None
        self.valid_res_df_all_info_from_to = None
        # Metadata for videos
        self.video_metadata = None
        self.video_metadata_from_to = None
        # Other variables
        self.channel = channel
        self.detector = kwargs.get('detector', 'yolo')
        self.feats = kwargs.get('feats', 'resnetv1')
        self.mod_feat = kwargs.get('mod_feat', 'fcg_average_vote')
        self.from_date = None
        self.to_date = None
        # Initialise variables
        self._read_df_all_info_()
        self._read_video_metadata_()
        from_date_str = kwargs.get('from_date_str', None)  # In case we have it precomputed for demo files
        to_date_str = kwargs.get('to_date_str', None)  # In case we have it precomputed for demo files
        self.filter_df_from_to(from_date_str, to_date_str)
        self.filter_metadata_from_to()
        self._compute_overall_info_()
        if mode == 'gt':
            self.gt_correction_factor = self.compute_correction_factor_year(self.res_df_all_info_from_to)
        if mode == 'gt' or mode == 'train':
            self.res_df_metrics_overall = self.compute_metrics(self.res_df_all_info_from_to)
        if self.gt_correction_factor is not None or mode == 'gt':
            self.res_corrected = self.compute_screen_seconds(self.res_df_all_info_from_to)
        else:
            self.res_corrected = self.compute_screen_seconds(self.res_df_all_info_from_to, flag_correct=False)

    def _read_df_all_info_(self):
        read_path = Path(self.res_path) / f'{self.detector}-{self.feats}-{self.mod_feat}.pkl'
        if read_path.is_file():
            self.res_df_all_info = pd.read_pickle(str(read_path))

    def filter_df_from_to(self, from_date_str, to_date_str):
        # It actually works like this, lol
        # vid --> 2001_06_30_19_00
        # from_date --> 2001_00_00
        # to_date --> 2002_05_30
        from_date = pd.to_datetime(from_date_str, format='%Y_%m_%d')
        to_date = pd.to_datetime(to_date_str, format='%Y_%m_%d') + pd.DateOffset(days=1)  # Actually add the day to analyse (hours are at 00:00:00)
        df_date = pd.to_datetime(self.res_df_all_info['vid'], format='%Y_%m_%d_%H_%M')
        mask_dates = (df_date >= from_date) & (df_date <= to_date)
        self.res_df_all_info_from_to = self.res_df_all_info[mask_dates]
        self.res_df_all_info_from_to['date'] = df_date[mask_dates]
        # Once filtered, take the minimum and maximum dates of the results
        self.from_date = self.res_df_all_info_from_to['date'].min().replace(hour=00, minute=00, second=00)  # From the beginning of the day
        self.to_date = self.res_df_all_info_from_to['date'].max().replace(hour=23, minute=59, second=59)  # To the end of the day
        # self._filter_special_news_()
        self._filter_special_politicians_()
        # self._filter_small_faces_()  # Test filtering small faces for plotting

    def _filter_special_news_(self):
        # Filter some special programs (special COVID programs) --> longer (1:15-1:45), focusing on the PM (Suga or Abe)
        dates_to_filter = ['2020_04_07_19_00', '2021_01_13_19_00', '2021_02_02_19_00', '2021_03_18_19_00', '2021_05_07_19_00',
                           '2021_06_17_19_00', '2021_07_08_19_00', '2021_07_30_19_00', '2021_09_09_19_00', '2021_09_28_19_00']
        mask_dates = ~self.res_df_all_info_from_to['vid'].isin(dates_to_filter)
        self.res_df_all_info_from_to = self.res_df_all_info_from_to[mask_dates]

    def _filter_special_politicians_(self):
        # Filter some politicians that we suspect that have many false positives (looking at you, Hiranuma)
        pols_to_filter = ['Takeo_Hiranuma', 'Takako_Doi']
        mask_pols = ~self.res_df_all_info_from_to['ID'].isin(pols_to_filter)
        self.res_df_all_info_from_to = self.res_df_all_info_from_to[mask_pols]

    def _filter_small_faces_(self):
        sr_area = self.res_df_all_info_from_to['w'] * self.res_df_all_info_from_to['h']
        mask_small_area = sr_area > (32*32)
        self.res_df_all_info_from_to = self.res_df_all_info_from_to[mask_small_area]

    def _compute_overall_info_(self):
        years_res = self.res_df_all_info_from_to['year'].unique()
        for year in years_res:
            mask_year = self.res_df_all_info_from_to['year'] == year
            df = self.res_df_all_info_from_to[mask_year]
            row_overall = pd.DataFrame({'vid': f'{year}_01_01_00_00', 'seg': f'{year}_01_01_00_00_1.webm',
                                        'ID': 'Overall', 'frame': -1,
                                        'TP': df['TP'].sum(), 'FP': df['FP'].sum(), 'FN': df['FN'].sum(),
                                        'dist_ID': -1, 'year': year, 'cx': -1, 'cy': -1, 'w': -1, 'h': -1}, index=[0])
            self.res_df_all_info_from_to = pd.concat((self.res_df_all_info_from_to, row_overall), axis=0)

    def _read_video_metadata_(self):
        metadata_path = Path('/home/agirbau/work/politics/data/videos_metadata') / f'metadata_{self.channel}.csv'
        if metadata_path.exists():
            self.video_metadata = pd.read_csv(metadata_path)

    def filter_metadata_from_to(self):
        # Filter by dates
        if self.video_metadata is not None:
            df_date = pd.to_datetime(self.video_metadata['source'], format='%Y_%m_%d_%H_%M')
            mask_dates = (df_date >= self.from_date) & (df_date <= self.to_date)
            self.video_metadata_from_to = self.video_metadata[mask_dates]

    @staticmethod
    def _init_plot_(w_cm=20, h_cm=10):
        sns.set_theme(style="white")
        fig, ax = plt.subplots()
        fig.set_size_inches(cm_to_inch(w_cm), cm_to_inch(h_cm))
        return fig, ax

    @staticmethod
    def _purge_legend_(ax):
        handles, labels = ax.get_legend_handles_labels()
        labels_legend = set(labels) & (set(POLITICIANS) | set(PARTIES) | set(US_ACTORS))
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
    def filter_df_by_politicians(df: pd.DataFrame, politicians_list: list) -> pd.DataFrame:
        if politicians_list is None:
            return df

        if len(politicians_list) > 0:
            mask_politicians = df['ID'].isin(politicians_list)
            return df[mask_politicians]
        else:
            return df

    @staticmethod
    def filter_df_by_parties(df: pd.DataFrame, parties_list: list) -> pd.DataFrame:
        if parties_list is None:
            return df

        if len(parties_list) > 0:
            mask_politicians = df['party'].isin(parties_list)
            return df[mask_politicians]
        else:
            return df

    @staticmethod
    def filter_df_top_k(df: pd.DataFrame, k: int, flag_local: bool = False) -> pd.DataFrame:
        if k == 0:
            return df
        else:
            if flag_local:
                # Locally (per month or day)
                ss_grouped = df.groupby(['year'], group_keys=False)['value'].nlargest(k)
                return df.iloc[ss_grouped.index]
            else:
                # Globally
                top_pol_list = list(df.groupby(['ID'])['value'].sum().nlargest(k).keys())
                return df[df['ID'].isin(top_pol_list)]

    @staticmethod
    def append_to_df(df, rows, var_name='TP_d', dummy=None):
        df_new = df.copy()
        new_rows = []
        for row in rows:
            new_rows.append(pd.DataFrame(data=[[dummy, row, var_name, -1000]], columns=df.keys().to_list()))

        df_new = df_new.append(new_rows, ignore_index=True)
        return df_new

    @staticmethod
    def add_zero_values_to_df(df, var_name='TP_d'):
        # Add 0 values to politicians for representation
        df_new = df.copy()
        # years = set(df_new['year'].unique())
        date_min = df_new['year'].min()
        date_max = df_new['year'].max()
        if len(date_min.split('_')) > 2:
            # Daily
            date_format = "%Y_%m_%d"
            date_freq = 'D'
        else:
            # Monthly
            date_format = "%Y_%m"
            date_freq = 'M'

        sdate = datetime.strptime(date_min, date_format).date()
        edate = datetime.strptime(date_max, date_format).date()
        years_datetime = pd.date_range(sdate, edate - timedelta(days=1), freq=date_freq).to_list()
        years = set([datetime.strftime(date_str, date_format) for date_str in years_datetime])
        pols = df_new['ID'].unique()
        set_list = [set(df_new[df_new['ID'] == pol]['year']) for pol in pols]
        years_no_pol_list = [years - set_pol for set_pol in set_list]
        rows = []
        for pol, years_no_pol in zip(pols, years_no_pol_list):
            for year in years_no_pol:
                rows.append([pol, year, var_name, 0])

        if len(rows) > 0:
            # If empty the elements of the df get re-defined as "object" and we have problems afterwards.
            df_new_rows = pd.DataFrame(data=rows, columns=['ID', 'year', 'variable', 'value'])
            df_new = df_new.append(df_new_rows)

        return df_new

    def split_train_valid(self, per=0.8):
        msk = np.random.rand(len(self.res_df_all_info_from_to)) < per
        self.train_res_df_all_info_from_to = self.res_df_all_info_from_to[msk]
        self.valid_res_df_all_info_from_to = self.res_df_all_info_from_to[~msk]

    def compute_correction_factor(self, df):
        # Estimate correction ratio globally
        # Correction factor: t = t' * c  --> c = P/R  --> TP_corrected = TP * P/R
        df_metrics = self.compute_metrics(df)
        df_c_factor = (df_metrics['prec'] / df_metrics['rec']).fillna(1)
        list_corr_factor = [[id_pol, c_factor] for id_pol, c_factor in zip(df_metrics['ID'].to_list(), df_c_factor.to_list())]
        df_correction_factor = pd.DataFrame(data=list_corr_factor, columns=['ID', 'c_factor'])
        return df_correction_factor

    def compute_correction_factor_year(self, df):
        # Estimate correction ratio per year
        # Correction factor: t = t' * c  --> c = P/R  --> TP_corrected = TP * P/R
        list_corr_factor = []
        years = df['year'].unique()
        for year in years:
            mask_year = df['year'] == year
            df_year = df[mask_year]
            df_metrics = self.compute_metrics(df_year)
            df_c_factor = (df_metrics['prec'] / df_metrics['rec']).fillna(1)
            list_corr_factor += [[id_pol, year, c_factor] for id_pol, c_factor in zip(df_metrics['ID'].to_list(), df_c_factor.to_list())]

        df_correction_factor = pd.DataFrame(data=list_corr_factor, columns=['ID', 'year', 'c_factor'])
        return df_correction_factor

    def compute_screen_seconds(self, df, flag_correct=True):
        # Returns a dataframe: [ID, year, TP, TP_corrected, c_factor]
        if not flag_correct:
            df_fast = df.groupby(['ID', 'party', 'year']).size().reset_index().rename(columns={0: 'TP_d'})
            df_fast[['TP_t', 'TP_c', 'c_factor']] = [df_fast['TP_d'], df_fast['TP_d'], -1]
            return df_fast

        list_corr_seconds = []
        keys_pol = df['ID'].unique()
        for k_pol in keys_pol:
            mask_pol = df['ID'] == k_pol
            mask_pol_corr = self.gt_correction_factor['ID'] == k_pol
            df_pol = df[mask_pol]
            df_pol_corr = self.gt_correction_factor[mask_pol_corr]
            party = df_pol['party'].unique().item()
            years = df_pol['year'].unique()
            true_seconds_pol_all = 0
            tp_pol_all = 0
            tp_c_pol_all = 0
            for year in years:
                mask_year = df_pol['year'] == year
                mask_year_corr = df_pol_corr['year'] == year[:4]  # Only get the year for the correcting factor
                df_pol_year = df_pol[mask_year]
                true_seconds_pol = df_pol_year['TP'].sum() + df_pol_year['FN'].sum()
                det_seconds_pol = df_pol_year['TP'].sum() + df_pol_year['FP'].sum()  # FPs are also detections
                c_factor = df_pol_corr[mask_year_corr]['c_factor'].item() if any(mask_year_corr) else 1
                corr_seconds_pol = det_seconds_pol * c_factor
                list_corr_seconds += [[k_pol, party, year, true_seconds_pol, det_seconds_pol, corr_seconds_pol, c_factor]]
                true_seconds_pol_all += true_seconds_pol
                tp_pol_all += det_seconds_pol
                tp_c_pol_all += corr_seconds_pol

            list_corr_seconds += [[k_pol, party, 'all', true_seconds_pol_all, tp_pol_all, tp_c_pol_all, -1]]

        res_corrected = pd.DataFrame(data=list_corr_seconds, columns=['ID', 'party', 'year', 'TP_t', 'TP_d', 'TP_c', 'c_factor'])
        return res_corrected

    @staticmethod
    def compute_percentage_screen_time(df, df_metadata):
        # Reuse "year" as monthly or daily dates
        df_meta_group = df_metadata.groupby(['year'])['seconds'].agg({np.sum, np.mean}).reset_index()
        df_meta_group.columns = ['year', 'avg_secs', 'total_secs']

        df_list = []
        dates_list = df['year'].unique()
        for date in dates_list:
            df_date = df[df['year'] == date]
            avg_secs_meta_date = df_meta_group[df_meta_group['year'] == date]['total_secs'].to_numpy()[-1]
            df_date.loc[:, 'perc'] = (df_date['TP_d'] / avg_secs_meta_date) * 100  # Percentage of screen time per date (monthly / daily)
            df_list.append(df_date)

        df_res = pd.concat(df_list)
        return df_res

    @staticmethod
    def compute_smoothed_time(df, method='gauss', **kwargs):
        # I tried to do this with groupy but it was taking so much time.
        # df_res = df.groupby(['ID'])['TP_d'].rolling(15).mean().reset_index()
        win = kwargs.get('win', 15)
        alpha = kwargs.get('alpha', 0.5)
        span = kwargs.get('span', 7)
        sigma = kwargs.get('sigma', 3)

        df_res = []
        pols = df['ID'].unique()
        for pol in pols:
            mask_pol = df['ID'] == pol
            df_pol = df[mask_pol]
            if method == 'rolling':
                df_pol['avg'] = df_pol['TP_d'].rolling(win).mean()
                df_pol['perc_avg'] = df_pol['perc'].rolling(win).mean()
            elif method == 'exp':
                df_pol['avg'] = df_pol['TP_d'].ewm(span=span).mean()
                df_pol['perc_avg'] = df_pol['perc'].ewm(span=span).mean()
                # df_pol['avg'] = df_pol['TP_d'].ewm(alpha=alpha).mean()
                # df_pol['perc_avg'] = df_pol['perc'].ewm(alpha=alpha).mean()
            elif method == 'gauss':
                # From https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
                det_np = df_pol['TP_d'].to_numpy(dtype=np.float)
                perc_np = df_pol['perc'].to_numpy(dtype=np.float)
                df_pol.loc[:, 'avg'] = gaussian_filter1d(det_np, sigma=sigma)
                df_pol.loc[:, 'perc_avg'] = gaussian_filter1d(perc_np, sigma=sigma)

            # elif method == 'poly':
            #     x = np.arange(0, df_pol.shape[0])
            #     det_np = df_pol['TP_d'].to_numpy()
            #     perc_np = df_pol['perc'].to_numpy()
            #     if len(x) <= 1:
            #         df_pol['avg'] = df_pol['TP_d']
            #         df_pol['perc_avg'] = df_pol['perc']
            #         continue
            #
            #     det_model = np.poly1d(np.polyfit(x, det_np, 3))
            #     perc_model = np.poly1d(np.polyfit(x, perc_np, 3))
            #     df_pol['avg'] = det_model(x)
            #     df_pol['perc_avg'] = perc_model(x)
            #     df_pol['avg'][df_pol['avg'] < 0] = 0.0
            #     df_pol['perc_avg'][df_pol['perc_avg'] < 0] = 0.0

            df_res.append(df_pol)
        df_res = pd.concat(df_res)

        return df_res

    @staticmethod
    def smooth_plot_lines(df_plot, method='gauss', **kwargs):
        # Smooth lines to be plotted
        df = df_plot.sort_values(by=['ID', 'year'])
        win = kwargs.get('win', 15)
        alpha = kwargs.get('alpha', 0.5)
        span = kwargs.get('span', 1)
        sigma = kwargs.get('sigma', 3)

        df_res = []
        pols = df['ID'].unique()
        for pol in pols:
            mask_pol = df['ID'] == pol
            df_pol = df[mask_pol]
            if method == 'rolling':
                df_pol['value'] = df_pol['value'].rolling(win).mean()
            elif method == 'exp':
                df_pol['value'] = df_pol['value'].ewm(span=span).mean()
            elif method == 'gauss':
                # From https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
                det_np = df_pol['value'].to_numpy()
                df_pol.loc[:, 'value'] = gaussian_filter1d(det_np, sigma=sigma)

            df_res.append(df_pol)

        df_res = pd.concat(df_res)

        return df_res

    @staticmethod
    def compute_metrics(df):
        keys_pol = df['ID'].unique()
        tp_fp_fn_list = []
        for k_pol in keys_pol:
            mask_pol = df['ID'] == k_pol
            df_pol = df[mask_pol]
            tp_pol = df_pol['TP'].sum()
            fp_pol = df_pol['FP'].sum()
            fn_pol = df_pol['FN'].sum()
            prec_pol = tp_pol / (tp_pol + fp_pol) if tp_pol > 0 else 0
            rec_pol = tp_pol / (tp_pol + fn_pol) if tp_pol > 0 else 0
            f1_pol = tp_pol / (tp_pol + (0.5 * (fp_pol + fn_pol)))

            tp_fp_fn_list.append([k_pol, tp_pol, fp_pol, fn_pol, prec_pol, rec_pol, f1_pol])

        df_metrics = pd.DataFrame(data=tp_fp_fn_list, columns=['ID', 'TP', 'FP', 'FN', 'prec', 'rec', 'f1'])
        return df_metrics

    def compute_correction_error(self):
        keys_pol = self.res_corrected['ID'].unique()
        err_list = []
        err_d_total = 0
        err_c_total = 0
        for k_pol in keys_pol:
            mask_pol = self.res_corrected['ID'] == k_pol
            df_pol = self.res_corrected[mask_pol]
            years = df_pol['year'].unique()
            err_d = 0
            err_c = 0
            # Compute error per year and sum it all up (doing it globally may not work as expected)
            for year in years:
                mask_year = df_pol['year'] == year
                df_pol_year = df_pol[mask_year]
                err_d += abs(df_pol_year['TP_t'].sum() - df_pol_year['TP_d'].sum())
                err_c += round(abs(df_pol_year['TP_t'].sum() - df_pol_year['TP_c'].sum()), 0)

            err_d_per = round((err_d / (df_pol['TP_t'].sum())) * 100, 1)
            err_c_per = round((err_c / (df_pol['TP_t'].sum())) * 100, 1)

            err_d_total += err_d
            err_c_total += err_c

            err_list.append([k_pol, err_d, err_c, err_d_per, err_c_per])

        err_d_per_total = round((err_d_total / (self.res_corrected['TP_t'].sum())) * 100, 1)
        err_c_per_total = round((err_c_total / (self.res_corrected['TP_t'].sum())) * 100, 1)
        err_list.append(['Overall_err', err_d_total, err_c_total, err_d_per_total, err_c_per_total])

        df_err = pd.DataFrame(data=err_list, columns=['ID', 'err_d', 'err_c', 'err_d_per', 'err_c_per'])
        return df_err

    def print_prec_rec(self):
        keys_pol = self.res_df_metrics_overall['ID'].unique()
        for k_pol in keys_pol:
            mask_pol = self.res_df_metrics_overall['ID'] == k_pol
            df_pol = self.res_df_metrics_overall[mask_pol]
            print(f"{k_pol}: P:{round(df_pol['prec'].item(), 2)}, R:{round(df_pol['rec'].item(), 2)}, F1:{round(df_pol['f1'].item(), 2)}")

    def print_screen_time(self):
        keys_pol = self.res_corrected['ID'].unique()
        for k_pol in keys_pol:
            mask_pol = self.res_corrected['ID'] == k_pol
            df_pol = self.res_corrected[mask_pol]
            print(f"{k_pol}: {df_pol['TP_d'].sum()} seconds, {round(df_pol['TP_c'].sum(), 0)} seconds corrected")

    @staticmethod
    def print_screen_time_from_plot(df: pd.DataFrame) -> None:
        df_pol = df.groupby(['ID'])['value'].sum().sort_values(ascending=False)
        for row in df_pol.iteritems():
            print(f"{row[0]}: {row[1]} seconds")
            # print(f"{row[0]} {row[1]}")  # For excel

    def print_correction_error(self):
        df_err = self.compute_correction_error()
        keys_pol = df_err['ID'].unique()
        for k_pol in keys_pol:
            # if k_pol != 'Overall_err':
            #     continue
            mask_pol = df_err['ID'] == k_pol
            df_err_pol = df_err[mask_pol]
            print(f"{k_pol}: det_err: {df_err_pol['err_d'].item()}, det_err_per: {df_err_pol['err_d_per'].item()}%, "
                  f"corr_err: {df_err_pol['err_c'].item()}, cor_err_per: {df_err_pol['err_c_per'].item()}%")

    @staticmethod
    def get_top_pol_appearances(df, pol, topk=5):
        mask_pol = df['ID'] == pol
        df_pol = df[mask_pol].groupby(['source'])['TP'].sum().sort_values(ascending=False).nlargest(topk)
        return df_pol

    def print_top_appearances(self, politicians_list, topk=5):
        # If "--who" is specified, print the videos where that/those politicians appear the most (for now top 5)
        if len(politicians_list) > 0:
            df = self.filter_df_by_politicians(self.res_df_all_info_from_to, politicians_list)
            for pol in politicians_list:
                df_pol = self.get_top_pol_appearances(df, pol, topk)
                dict_pol = df_pol.to_dict()
                print(f'{pol}:')
                print(f'{dict_pol}')

    def plot_screen_time(self):
        sns.set_theme(style="whitegrid")
        _ = plt.figure()

        mask_no_overall = self.res_corrected['ID'] != 'Overall'
        mask_year = self.res_corrected[mask_no_overall]['year'] == 'all'
        df_plot = self.res_corrected[mask_no_overall][mask_year].sort_values(by='ID')
        df_plot = pd.melt(df_plot, id_vars=['ID'], value_vars=['TP_d', 'TP_c'])
        # df_plot = pd.melt(df_plot, id_vars=['ID'], value_vars=['TP_d'])

        bar_secs = sns.barplot(data=df_plot, y='value', x='ID', hue='variable')
        bar_secs.set_xticklabels(bar_secs.get_xticklabels(), size=7, rotation=60, ha='right')
        plt.ylabel('')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 5500)
        plt.text(0.0, 1.03, f"{self.from_date}-{self.to_date}", fontsize=10, transform=bar_secs.transAxes)
        plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
        plt.tight_layout()
        plt.show()

    def plot_screen_time_timeline(self, flag_party=False, flag_pm=False):
        sns.set_theme(style="white")
        _ = plt.figure(figsize=(cm_to_inch(20), cm_to_inch(10)))

        mask_no_overall = self.res_corrected['ID'] != 'Overall'
        mask_year_noall = self.res_corrected[mask_no_overall]['year'] != 'all'
        df_plot = self.res_corrected[mask_no_overall][mask_year_noall].sort_values(by='year')
        # df_plot = pd.melt(df_plot, id_vars=['ID', 'year'], value_vars=['TP_d', 'TP_c'])
        df_plot = pd.melt(df_plot, id_vars=['ID', 'year'], value_vars=['TP_d'])
        # df_plot = pd.melt(df_plot, id_vars=['ID', 'year'], value_vars=['TP_t'])

        color_palette_mod = get_politician_palette(df_plot['ID'].unique())
        line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue='ID', marker='o',
                                 palette=sns.color_palette(palette=color_palette_mod, n_colors=20))
        line_secs.set_xticklabels(df_plot['year'].unique(), size=7, rotation=60, ha='right')
        line_secs.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)

        plt.ylabel('screen time (seconds)')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 4000)
        plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
        plt.tight_layout()
        plt.show()

    def scale_values_with_correction_factor(self, df):
        df_res = df.copy()
        pol_ids_correction = self.gt_correction_factor['ID'].unique()
        pol_ids_df = df['ID'].unique()

        for pol_id in pol_ids_correction:
            if pol_id not in pol_ids_df:
                continue

            mask_id_corr = self.gt_correction_factor['ID'] == pol_id
            mask_id_plot = df_res['ID'] == pol_id

            correction_factor = self.gt_correction_factor[mask_id_corr]['c_factor'].item()
            df_res.loc[mask_id_plot, ('TP_d')] = df_res.loc[mask_id_plot, ('TP_d')] * correction_factor  # Correct way of setting new values

        return df_res

    def generate_df_screen_time(self, **kwargs):
        # Function to do the first filtering of times and stuff from the original dataframe
        # kwargs
        politicians = kwargs.get('politicians', None)
        parties = kwargs.get('parties', None)
        date_res = kwargs.get('date_res', 'month')
        win_size = kwargs.get('win_size', None)
        flag_correct = kwargs.get('flag_correct', False)
        flag_uncertainty = kwargs.get('flag_uncertainty', False)

        # Reuse for month and day
        str_len = 7
        win_size = 3 if win_size is None else win_size
        if date_res == 'month':
            str_len = 7
            win_size = 3 if win_size is None else win_size
        elif date_res == 'day':
            str_len = 10
            win_size = 7 if win_size is None else win_size

        # Here we will do a hack to re-use previous functions --> change year to year_month
        df = self.res_df_all_info_from_to.copy()
        df['year'] = self.res_df_all_info_from_to['vid'].str[:str_len]
        df = self.filter_df_by_politicians(df, politicians)
        df = self.filter_df_by_parties(df, parties)
        # Same for video metadata to filter in one go
        df_metadata = self.video_metadata_from_to.copy()
        df_metadata['year'] = self.video_metadata_from_to['source'].str[:str_len]

        res_corrected = self.compute_screen_seconds(df, flag_correct=flag_correct)
        if flag_uncertainty:
            res_corrected = self.scale_values_with_correction_factor(res_corrected)
        res_corrected = self.compute_percentage_screen_time(res_corrected, df_metadata)  # To get percentage wrt program
        res_corrected = self.compute_smoothed_time(res_corrected, win=win_size, **kwargs)  # To compute rolling average (smooth trends)
        mask_no_overall = res_corrected['ID'] != 'Overall'
        mask_year_noall = res_corrected[mask_no_overall]['year'] != 'all'
        df_res = res_corrected[mask_no_overall][mask_year_noall].sort_values(by=['year'])

        return df_res

    def generate_df_plot(self, **kwargs):
        # Function returning a dataframe prepared for plotting
        # kwargs
        topk = kwargs.get('topk', 0)
        date_res = kwargs.get('date_res', 'month')  # Date resolution
        flag_party = kwargs.get('flag_party', False)
        flag_pm = kwargs.get('flag_pm', False)
        flag_op = kwargs.get('flag_op', False)
        flag_prev_pm = kwargs.get('flag_prev_pm', False)
        flag_prev_op = kwargs.get('flag_prev_op', False)
        flag_gov_team = kwargs.get('flag_gov_team', False)
        flag_op_team = kwargs.get('flag_op_team', False)
        flag_gov_op_color = kwargs.get('flag_gov_op_color', False)
        flag_perc = kwargs.get('flag_perc', False)
        flag_smooth = kwargs.get('flag_smooth', False)  # Flag for rolling average
        smooth_method = kwargs.get('smooth_method', 'gauss')  # Flag for rolling average

        # Choose whether plot the percentage or the total time
        # We smooth only the plot values now
        if flag_perc:
            var_name = 'perc'
            # if flag_smooth:
            #     var_name = 'perc_avg'
        else:
            var_name = 'TP_d'
            # if flag_smooth:
            #     var_name = 'avg'

        df_plot = self.generate_df_screen_time(**kwargs)

        # if flag_uncertainty:
        #     # Scale values depending on the performance with train (GT) data
        #     df_plot = self.scale_values_with_correction_factor(df_plot)

        # Filter and add prime ministers dates for visualization
        if date_res == 'month':
            idx_pminister_dates = filter_list_by_date(PM_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
            new_pminister_date = [dd for dd, bb in zip(PM_DATES_MONTH, idx_pminister_dates) if bb]
            idx_opleader_dates = filter_list_by_date(OP_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
            new_opleader_date = [dd for dd, bb in zip(OP_DATES_MONTH, idx_opleader_dates) if bb]
        else:
            idx_pminister_dates = filter_list_by_date(PM_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
            new_pminister_date = [dd for dd, bb in zip(PM_DATES_DAY, idx_pminister_dates) if bb]
            idx_opleader_dates = filter_list_by_date(OP_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
            new_opleader_date = [dd for dd, bb in zip(OP_DATES_DAY, idx_opleader_dates) if bb]

        new_pminister_list = [mm for mm, bb in zip(PRIME_MINISTERS, idx_pminister_dates) if bb]
        new_opleader_list = [mm for mm, bb in zip(OP_LEADERS, idx_opleader_dates) if bb]

        # Masks list that will combine to mask_glob later for filtering both PM and OP leader
        masks_list = []

        # Bug when there is no change in PM during the analyzed period (above)
        # Possible solution: redo the filtering with pandas df.truncate(before=, after=)
        if flag_pm and len(new_pminister_list) > 0:
            pm_df_list = []
            for i, pm in enumerate(new_pminister_list):
                # Case where we want to see the screen time of the current PM in the previous mandate
                if flag_prev_pm:
                    mask_pm = (df_plot['year'] >= new_pminister_date[i-1] if len(new_pminister_date) > i-1 else True) & \
                              (df_plot['year'] < new_pminister_date[i+1] if len(new_pminister_date) > i+1 else True) \
                              & (df_plot['ID'] == pm)
                # Case where we only want to see the screen time of the current PM
                else:
                    mask_pm = (df_plot['year'] >= new_pminister_date[i]) & \
                              (df_plot['year'] < new_pminister_date[i+1] if len(new_pminister_date) > i+1 else True) & \
                              (df_plot['ID'] == pm)

                pm_df_list.append(mask_pm)
            mask_pm_glob = sum(pm_df_list) > 0  # To get all the mask info at once
            masks_list.append(mask_pm_glob)

        if flag_op and len(new_opleader_list) > 0:
            op_df_list = []
            for i, op in enumerate(new_opleader_list):
                # Case where we want to see the screen time of the current PM in the previous mandate
                if flag_prev_op:
                    mask_pm = (df_plot['year'] >= new_opleader_date[i-1] if len(new_opleader_date) > i-1 else True) & \
                              (df_plot['year'] < new_opleader_date[i+1] if len(new_opleader_date) > i+1 else True) \
                              & (df_plot['ID'] == op)
                # Case where we only want to see the screen time of the current PM
                else:
                    mask_pm = (df_plot['year'] >= new_opleader_date[i]) & \
                              (df_plot['year'] < new_opleader_date[i+1] if len(new_opleader_date) > i+1 else True) & \
                              (df_plot['ID'] == op)

                op_df_list.append(mask_pm)
            mask_op_glob = sum(op_df_list) > 0
            masks_list.append(mask_op_glob)

        # Filter df plot with the information above if there's any condition to meet (flag_pm, flag_op)
        if len(masks_list) > 0:
            mask_glob = sum(masks_list) > 0
            df_plot = df_plot[mask_glob]

        # Case to plot the party leaders in the government and in the opposition
        df_leaders = pd.read_excel('/home/agirbau/work/politics/data/party_leaders_mod.xlsx')
        masks_gov_op_list = []

        if flag_gov_team or flag_op_team:
            # Add flag for politicians in government vs. politicians in the opposition
            df_plot['is_gov'] = 0

            for idx, row in df_leaders.iterrows():
                pol_id = row['name']
                term_dates = row['term'].split('-')  # [mm/dd/yyyy, mm/dd/yyy]
                new_order = [2, 0, 1]
                date_start_list = term_dates[0].split('/')
                date_end_list = term_dates[1].split('/')
                date_start = '_'.join([date_start_list[i] for i in new_order])  # [yyyy_mm_dd, yyyy_mm_dd]
                date_end = '_'.join([date_end_list[i] for i in new_order]) if len(date_end_list) > 1 else '2100_01_01'  # [yyyy_mm_dd, yyyy_mm_dd]
                mask_pol = (df_plot['ID'] == pol_id) & (df_plot['year'] >= date_start) & (df_plot['year'] < date_end)

                if row['incumbent'] == 1 and flag_gov_team:
                    # Politicans in the government within their incumbent dates
                    masks_gov_op_list.append(mask_pol)
                    df_plot.loc[mask_pol, 'is_gov'] = 1
                    # df_plot['is_gov'][mask_pol] = 1

                elif row['incumbent'] == 0 and flag_op_team:
                    # Get all the politicans in the opposition within their incumbent dates
                    masks_gov_op_list.append(mask_pol)

        # Filter df plot with the information above
        if len(masks_gov_op_list) > 0:
            mask_glob_gov_op = sum(masks_gov_op_list) > 0
            df_plot = df_plot[mask_glob_gov_op]

        # Choose what ID to show for the plots
        pol_id = 'ID'
        if flag_party:
            pol_id = 'party'
        elif flag_gov_op_color:
            # Sum the times or something for the people in the government and the people in the opposition
            df_plot = df_plot.groupby(['year', 'is_gov']).sum().reset_index()
            pol_id = 'is_gov'

        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d', 'TP_c'])
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_c'])

        df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=[var_name])
        df_plot.rename(columns={df_plot.columns[0]: 'ID'}, inplace=True)  # In case we have "party" or "is_gov"

        if not flag_party:
            df_plot = self.filter_df_top_k(df_plot, topk)
            self.print_screen_time_from_plot(df_plot)

        if flag_party:
            parties_list = []
            for party in df_plot['ID'].unique():
                mask_party = df_plot['ID'] == party
                for year in df_plot['year'][mask_party].unique():
                    mask_year = df_plot['year'][mask_party] == year
                    parties_list.append([party, year, 'TP', df_plot['value'][mask_party][mask_year].sum()])

            df_plot = pd.DataFrame(data=parties_list, columns=['ID', 'year', 'variable', 'value']).sort_values(by=['year'])

        # Add 0 values to plots
        df_plot = self.add_zero_values_to_df(df_plot, var_name=var_name)
        if flag_smooth:
            df_plot = self.smooth_plot_lines(df_plot, method=smooth_method, **kwargs)

        if date_res == 'month':
            # Add dates for plots for monthly analysis
            idx_repr = filter_list_by_date(REPRESENTATIVES_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
            idx_coun = filter_list_by_date(COUNCILLORS_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
            idx_ldp = filter_list_by_date(LDP_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
            repr_list = [dd for dd, bb in zip(REPRESENTATIVES_DATES_MONTH, idx_repr) if bb]
            coun_list = [dd for dd, bb in zip(COUNCILLORS_DATES_MONTH, idx_coun) if bb]
            ldp_list = [dd for dd, bb in zip(LDP_DATES_MONTH, idx_ldp) if bb]
            df_plot = self.append_to_df(df_plot, new_pminister_date, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, repr_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, coun_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, ldp_list, var_name=var_name, dummy='.')
        else:
            # Add dates for plots for daily analysis
            idx_repr = filter_list_by_date(REPRESENTATIVES_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
            idx_diss = filter_list_by_date(PARLIAMENT_DISSOLUTION_DAY, from_date=self.from_date, to_date=self.to_date)
            idx_repr_camp = filter_list_by_date(PARL_CAMPAIGN_START_DAY, from_date=self.from_date, to_date=self.to_date)
            idx_coun = filter_list_by_date(COUNCILLORS_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
            idx_coun_camp = filter_list_by_date(COUNC_CAMPAIGN_START_DAY, from_date=self.from_date, to_date=self.to_date)
            idx_ldp = filter_list_by_date(LDP_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
            repr_list = [dd for dd, bb in zip(REPRESENTATIVES_DATES_DAY, idx_repr) if bb]
            diss_list = [dd for dd, bb in zip(PARLIAMENT_DISSOLUTION_DAY, idx_diss) if bb]
            repr_camp_list = [dd for dd, bb in zip(PARL_CAMPAIGN_START_DAY, idx_repr_camp) if bb]
            coun_list = [dd for dd, bb in zip(COUNCILLORS_DATES_DAY, idx_coun) if bb]
            coun_camp_list = [dd for dd, bb in zip(COUNC_CAMPAIGN_START_DAY, idx_coun_camp) if bb]
            ldp_list = [dd for dd, bb in zip(LDP_DATES_DAY, idx_ldp) if bb]
            df_plot = self.append_to_df(df_plot, new_pminister_date, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, repr_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, diss_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, repr_camp_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, coun_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, coun_camp_list, var_name=var_name, dummy='.')
            df_plot = self.append_to_df(df_plot, ldp_list, var_name=var_name, dummy='.')

        df_plot = df_plot.sort_values(by=['year', 'ID'])

        return df_plot

    def plot_screen_time_timeline_month(self, **kwargs):
        # kwargs
        flag_party = kwargs.get('flag_party', False)
        flag_perc = kwargs.get('flag_perc', False)
        flag_dates = kwargs.get('flag_dates', False)
        flag_legend = kwargs.get('flag_legend', True)
        max_value = kwargs.get('max_value', None)
        save_name = kwargs.get('save_name', None)

        # fig and axis handling
        ax = kwargs.get('ax', None)  # Try to get axis from outside (for analysing several channels at the same time)
        if ax is None:
            fig, ax = self._init_plot_()

        # Generate for plotting
        df_plot = kwargs.get('df_plot', None)  # In case we want to give the plot from outside
        if df_plot is None:
            df_plot = self.generate_df_plot(**kwargs)

        # Care with parties here
        pol_id = 'ID' if not flag_party else 'party'
        color_palette_mod = get_politician_palette(df_plot[pol_id].unique())

        n_colors = len(df_plot[pol_id].unique())
        line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue=pol_id,
                                 palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)
        # New ticks corresponding to each year instead of the whole year_month (unreadable)
        new_xticks = []
        for year_month in df_plot['year'].unique():
            year = year_month.split('_')[0]
            if year not in new_xticks:
                new_xticks.append(year)
            else:
                new_xticks.append(' ')

        line_secs.set_xticklabels(new_xticks, size=8, rotation=60, ha='right')
        max_value = df_plot['value'].max() if max_value is None else max_value

        # Filter and add prime ministers dates for visualization
        idx_pminister_dates = filter_list_by_date(PM_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
        new_pminister_date = [dd for dd, bb in zip(PM_DATES_MONTH, idx_pminister_dates) if bb]
        # Add dates for plots
        idx_repr = filter_list_by_date(REPRESENTATIVES_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
        repr_list = [dd for dd, bb in zip(REPRESENTATIVES_DATES_MONTH, idx_repr) if bb]

        if flag_dates:
            plt.vlines(x=new_pminister_date, ymin=0, ymax=max_value, linestyles='dashed', colors='gray', label='new_pminister')
            plt.vlines(x=repr_list, ymin=0, ymax=max_value, linestyles='solid', colors='black', label='repr_election')
            # plt.vlines(x=coun_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            # plt.vlines(x=ldp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='ldp_election')

        self.format_plot(ax)
        if not flag_legend:
            # In case we don't want to show legend (specially when we want to compare plots for plot scaling reasons)
            ax.legend().remove()

        plt.ylim(0, max_value)
        if flag_perc:
            plt.ylabel('Percentage of screen time')
        else:
            plt.ylabel('screen time (seconds)')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 4000)
        # plt.title(f'{args.detector}-{args.mod_feat}')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f'figures/{save_name}', dpi=250)
        plt.show()

    def plot_screen_time_timeline_day(self, **kwargs):
        fig, ax = self._init_plot_()
        flag_party = kwargs.get('flag_party', False)
        flag_dates = kwargs.get('flag_dates', False)
        flag_legend = kwargs.get('flag_legend', True)
        max_value = kwargs.get('max_value', None)
        save_name = kwargs.get('save_name', None)
        flag_perc = kwargs.get('flag_perc', False)

        # Generate for plotting
        df_plot = kwargs.get('df_plot', None)  # In case we want to give the plot from outside
        if df_plot is None:
            df_plot = self.generate_df_plot(date_res='day', **kwargs)

        # Filter and add prime ministers dates for visualization
        idx_pminister_dates = filter_list_by_date(PM_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
        new_pminister_date = [dd for dd, bb in zip(PM_DATES_DAY, idx_pminister_dates) if bb]
        new_pminister_list = [mm for mm, bb in zip(PRIME_MINISTERS, idx_pminister_dates) if bb]

        # Add dates for plots
        idx_repr = filter_list_by_date(REPRESENTATIVES_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
        idx_diss = filter_list_by_date(PARLIAMENT_DISSOLUTION_DAY, from_date=self.from_date, to_date=self.to_date)
        idx_repr_camp = filter_list_by_date(PARL_CAMPAIGN_START_DAY, from_date=self.from_date, to_date=self.to_date)
        idx_coun = filter_list_by_date(COUNCILLORS_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
        idx_coun_camp = filter_list_by_date(COUNC_CAMPAIGN_START_DAY, from_date=self.from_date, to_date=self.to_date)
        idx_ldp = filter_list_by_date(LDP_DATES_DAY, from_date=self.from_date, to_date=self.to_date)
        repr_list = [dd for dd, bb in zip(REPRESENTATIVES_DATES_DAY, idx_repr) if bb]
        diss_list = [dd for dd, bb in zip(PARLIAMENT_DISSOLUTION_DAY, idx_diss) if bb]
        repr_camp_list = [dd for dd, bb in zip(PARL_CAMPAIGN_START_DAY, idx_repr_camp) if bb]
        coun_list = [dd for dd, bb in zip(COUNCILLORS_DATES_DAY, idx_coun) if bb]
        coun_camp_list = [dd for dd, bb in zip(COUNC_CAMPAIGN_START_DAY, idx_coun_camp) if bb]
        ldp_list = [dd for dd, bb in zip(LDP_DATES_DAY, idx_ldp) if bb]

        # Care here for parties
        pol_id = 'ID'
        # pol_id = 'ID' if not flag_party else 'party'
        color_palette_mod = get_politician_palette(df_plot[pol_id].unique())
        n_colors = len(df_plot[pol_id].unique())
        line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue=pol_id,
                                 palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)

        # Creating a stem plot (used for one politician)
        # line_secs = sns.scatterplot(data=df_plot, y='value', x='year', hue=pol_id,
        #                             palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)
        # plt.stem(df_plot['year'], df_plot['value'])

        # New ticks corresponding to each year instead of the whole year_month (unreadable)
        new_xticks = []
        years_month = df_plot['year'].unique()
        for year_month in years_month:
            if len(years_month) < 100:
                tick = year_month[:10]
            elif len(years_month) < 1000:
                tick = year_month[:7]
            else:
                tick = year_month[:4]
            if tick not in new_xticks:
                new_xticks.append(tick)
            else:
                new_xticks.append(' ')

        line_secs.set_xticklabels(new_xticks, size=8, rotation=60, ha='right')
        line_secs.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)
        # election_list = ['2003_11_09', '2005_09_11', '2009_08_30', '2012_12_16']
        # 2001_04_26, 2003_11_09, 2005_09_11, 2014_12_14 videos, and 2017_10_22 are not available (reflexion day?)

        max_value = df_plot['value'].max() if max_value is None else max_value

        if flag_dates:
            plt.vlines(x=new_pminister_date, ymin=0, ymax=max_value, linestyles='dashed', colors='gray', label='new_pminister')
            plt.vlines(x=repr_list, ymin=0, ymax=max_value, linestyles='solid', colors='gray', label='repr_election')
            plt.vlines(x=diss_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            plt.vlines(x=repr_camp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            # plt.vlines(x=coun_list, ymin=0, ymax=max_value, linestyles='solid', colors='black', label='coun_election')
            # plt.vlines(x=coun_camp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            ### LDP
            # plt.vlines(x=ldp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='ldp_election')

        self.format_plot(ax)
        if not flag_legend:
            # In case we don't want to show legend (specially when we want to compare plots for plot scaling reasons)
            ax.legend().remove()

        if flag_perc:
            plt.ylim(0, max_value)
            plt.ylabel('Percentage of screen time')
        else:
            plt.ylim(0, max_value + 20)
            plt.ylabel('screen time (seconds)')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 4000)
        # plt.title(f'{args.detector}-{args.mod_feat}')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f'figures/{save_name}', dpi=250)
        plt.show()

    def plot_tp_fp_fn(self):
        sns.set_theme(style="whitegrid")
        _ = plt.figure()

        mask_no_overall = self.res_df_all_info_from_to['ID'] != 'Overall'
        df_plot = self.compute_metrics(self.res_df_all_info_from_to[mask_no_overall]).sort_values(by='ID')
        df_plot['TPFN'] = df_plot['TP'] + df_plot['FN']
        df_plot['TPFNFP'] = df_plot['TP'] + df_plot['FN'] + df_plot['FP']

        sns.barplot(data=df_plot, y='TPFNFP', x='ID', color='red')
        sns.barplot(data=df_plot, y='TPFN', x='ID', color='darkblue')
        bar_secs = sns.barplot(data=df_plot, y='TP', x='ID', color='lightblue')
        bar_secs.set_xticklabels(bar_secs.get_xticklabels(), size=7, rotation=60, ha='right')
        #Legend
        top_bar = mpatches.Patch(color='lightblue', label='TP')
        mid_bar = mpatches.Patch(color='darkblue', label='FN')
        bottom_bar = mpatches.Patch(color='red', label='FP')
        plt.legend(handles=[top_bar, mid_bar, bottom_bar])

        plt.ylabel('')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 5500)
        plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
        plt.text(0.0, 1.03, f"{self.from_date}-{self.to_date}", fontsize=10, transform=bar_secs.transAxes)
        plt.tight_layout()
        plt.show()

    def plot_tp_fp_fn_pol_year(self, k_pol):
        sns.set_theme(style="whitegrid")
        _ = plt.figure()

        mask_pol = self.res_df_all_info_from_to['ID'] == k_pol
        df_pol = self.res_df_all_info_from_to[mask_pol]
        df_metrics_year_list = []
        years = df_pol['year'].unique()
        for year in years:
            df_metrics_year = self.compute_metrics(df_pol[df_pol['year'] == year])
            df_metrics_year['year'] = year
            df_metrics_year_list.append(df_metrics_year)
        df_plot = pd.concat(df_metrics_year_list)
        df_plot['TPFN'] = df_plot['TP'] + df_plot['FN']
        df_plot['TPFNFP'] = df_plot['TP'] + df_plot['FN'] + df_plot['FP']

        sns.barplot(data=df_plot, y='TPFNFP', x='year', color='red')
        sns.barplot(data=df_plot, y='TPFN', x='year', color='darkblue')
        bar_secs = sns.barplot(data=df_plot, y='TP', x='year', color='lightblue')
        bar_secs.set_xticklabels(bar_secs.get_xticklabels(), size=7, rotation=60, ha='right')
        #Legend
        top_bar = mpatches.Patch(color='lightblue', label='TP')
        mid_bar = mpatches.Patch(color='darkblue', label='FN')
        bottom_bar = mpatches.Patch(color='red', label='FP')
        plt.legend(handles=[top_bar, mid_bar, bottom_bar])

        plt.ylabel('')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 5500)
        plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
        plt.text(0.0, 1.03, f"{self.from_date}-{self.to_date}", fontsize=10, transform=bar_secs.transAxes)
        plt.tight_layout()
        plt.show()

    def plot_correction_error(self):
        sns.set_theme(style="whitegrid")
        _ = plt.figure()

        df_err = self.compute_correction_error()

        mask_no_overall = (df_err['ID'] != 'Overall') & (df_err['ID'] != 'Overall_err')
        df_plot = df_err[mask_no_overall].sort_values(by='ID')
        df_plot = pd.melt(df_plot, id_vars=['ID'], value_vars=['err_d', 'err_c'])

        bar_secs = sns.barplot(data=df_plot, y='value', x='ID', hue='variable')
        bar_secs.set_xticklabels(bar_secs.get_xticklabels(), size=7, rotation=60, ha='right')
        plt.ylabel('')
        plt.xlabel('')
        # if args.mode == 'gt':
        #     plt.ylim(0, 4000)
        plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
        plt.tight_layout()
        plt.show()

    def plot_pm_opp(self, **kwargs):
        fig, ax = self._init_plot_()
        date_res = kwargs.get('date_res', 'month')
        dummy_val = kwargs.get('dummy', '.')  # To keep temporal scales between channels
        save_name = kwargs.get('save_name', None)
        flag_party = kwargs.get('flag_party', False)
        flag_gov_op_color = kwargs.get('flag_gov_op_color', False)
        flag_pm_date = kwargs.get('flag_pm_date', False)
        flag_ratio = kwargs.get('flag_ratio', True)
        flag_dates = kwargs.get('flag_dates', False)
        max_value = kwargs.get('max_value', None)
        save_name = kwargs.get('save_name', None)
        flag_perc = kwargs.get('flag_perc', False)
        flag_tv1_vs_tv2 = kwargs.get('flag_tv1_vs_tv2', False)  # Flag to compare GOV and OPP time of tv1 against tv2 (used to generate plot of NHK vs. HODO for PA2022 rebuttal)

        # Variable initialization (they might be modified within the code)
        min_value = 0

        # Generate for plotting
        df_plot = kwargs.get('df_plot', None)  # In case we want to give the plot from outside
        if df_plot is None:
            df_plot = self.generate_df_screen_time(**kwargs)

        # Temporal resolution
        if date_res == 'month':
            pm_dates = PM_DATES_MONTH
            op_dates = OP_DATES_MONTH
            repr_dates = REPRESENTATIVES_DATES_MONTH
            counc_dates = COUNCILLORS_DATES_MONTH
        else:
            # Daily resolution
            pm_dates = PM_DATES_DAY
            op_dates = OP_DATES_DAY
            repr_dates = REPRESENTATIVES_DATES_DAY
            counc_dates = COUNCILLORS_DATES_DAY

        if flag_gov_op_color:
            # Plot the relation between politicians leaders in the government vs. opposition leaders
            if flag_tv1_vs_tv2:
                # Do NHK vs. HODO station ratios
                # Assume df_plot here is a list of df_plots
                df_ratio_pm_list_ = []
                for df_plot_ in df_plot:
                    mask_gov_ = df_plot_['ID'] == 1
                    mask_gov_opp_ = (df_plot_['ID'] == 0) | (df_plot_['ID'] == 1)
                    df_ratio_pm_ = df_plot_[mask_gov_].groupby(['year']).sum() / df_plot_[mask_gov_opp_].groupby(['year']).sum().fillna(0)
                    df_ratio_pm_list_.append(df_ratio_pm_)
                # Ugly, but we need information inside df_plot
                df_plot = df_plot[0]
                df_ratio_pm = (df_ratio_pm_list_[0]/df_ratio_pm_list_[-1]) / 100
                # df_ratio_pm = df_ratio_pm_list_[0] - df_ratio_pm_list_[-1]
                # min_value = df_ratio_pm['value'].min() * 100
                min_value = 0.5
            else:
                mask_gov = df_plot['ID'] == 1
                mask_gov_opp = (df_plot['ID'] == 0) | (df_plot['ID'] == 1)
                df_ratio_pm = df_plot[mask_gov].groupby(['year']).sum() / df_plot[mask_gov_opp].groupby(['year']).sum().fillna(0)

        else:
            # Plot the PM vs. the opposition leader
            pm_df_list = []
            opp_df_list = []
            for i, pm in enumerate(PRIME_MINISTERS):
                mask_pm = (df_plot['year'] >= pm_dates[i]) & \
                          (df_plot['year'] < pm_dates[i + 1] if len(pm_dates) > i + 1 else True) & \
                          (df_plot['ID'] == pm)
                pm_df_list.append(mask_pm)

            for i, opp in enumerate(OP_LEADERS):
                mask_opp = (df_plot['year'] >= op_dates[i]) & \
                           (df_plot['year'] < op_dates[i + 1] if len(op_dates) > i + 1 else True) & \
                           (df_plot['ID'] == opp)
                opp_df_list.append(mask_opp)

            mask_glob = sum(pm_df_list + opp_df_list) > 0
            mask_glob_pm = sum(pm_df_list) > 0
            # df_plot = df_plot[mask_glob]
            df_ratio_pm = (df_plot[mask_glob_pm].groupby(['year']).sum() / df_plot[mask_glob].groupby(['year']).sum()).fillna(0)

        # Add dummy values for temporal axis for the times that we don't have the government nor opposition
        pm_opp_years = df_ratio_pm.index.to_list()
        mask_dummy = (df_plot['ID'] == dummy_val) & (~df_plot['year'].isin(pm_opp_years))
        df_dummy = pd.DataFrame(df_plot[mask_dummy][['value', 'year']]).set_index('year')
        df_dummy['value'] = 0.5  # 50% of time screen whenever PM or opp ar not shown
        df_ratio_pm = pd.concat([df_ratio_pm, df_dummy], axis=0).sort_index()

        # Choose whether plot the percentage or the total time
        # We smooth only the plot values now
        if flag_perc:
            var_name = 'perc'
            # if flag_smooth:
            #     var_name = 'perc_avg'
        else:
            var_name = 'TP_d'
            # if flag_smooth:
            #     var_name = 'avg'

        pol_id = 'ID' if not flag_party else 'party'
        # # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d', 'TP_c'])
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=[var_name])
        # # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_t'])

        if flag_party:
            parties_list = []
            for party in df_plot['party'].unique():
                mask_party = df_plot['party'] == party
                for year in df_plot['year'][mask_party].unique():
                    mask_year = df_plot['year'][mask_party] == year
                    parties_list.append([party, year, 'TP', df_plot['value'][mask_party][mask_year].sum()])

            df_plot = pd.DataFrame(data=parties_list, columns=['party', 'year', 'variable', 'value'])

        # Care here for parties
        color_palette_mod = get_politician_palette(df_plot[pol_id].unique())
        n_colors = len(df_plot[pol_id].unique())
        if flag_ratio:
            # Should be called something like "ratio_pm_occ" but it's made to reuse posterior code
            df_ratio_pm['year'] = df_ratio_pm.index
            df_ratio_pm['value'] = df_ratio_pm['value'] * 100
            df_plot = df_ratio_pm
            line_secs = sns.lineplot(data=df_plot, y='value', x='year', ax=ax)
            max_value = 100 if max_value is None else max_value
        else:
            line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue=pol_id,
                                     palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)
            max_value = df_plot['value'].max() if max_value is None else max_value
        # New ticks corresponding to each year instead of the whole year_month (unreadable)
        new_xticks = []
        for year_month in df_plot['year'].unique():
            year = year_month.split('_')[0]
            if year not in new_xticks:
                new_xticks.append(year)
            else:
                new_xticks.append(' ')

        line_secs.set_xticklabels(new_xticks, size=8, rotation=60, ha='right')
        line_secs.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)

        idx_repr = filter_list_by_date(repr_dates, from_date=self.from_date, to_date=self.to_date)
        idx_coun = filter_list_by_date(counc_dates, from_date=self.from_date, to_date=self.to_date)
        repr_list = [dd for dd, bb in zip(repr_dates, idx_repr) if bb]
        coun_list = [dd for dd, bb in zip(counc_dates, idx_coun) if bb]

        if flag_dates:
            plt.vlines(x=pm_dates, ymin=min_value, ymax=max_value, linestyles='dashed', colors='gray', label='new_pminister')
            plt.vlines(x=repr_list, ymin=min_value, ymax=max_value, linestyles='solid', colors='black', label='repr_election')
            # plt.vlines(x=repr_list, ymin=0, ymax=max_value, linestyles='solid', colors='gray', label='repr_election')
            # plt.vlines(x=diss_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            # plt.vlines(x=repr_camp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')
            # plt.vlines(x=coun_list, ymin=0, ymax=max_value, linestyles='solid', colors='black', label='coun_election')
            # plt.vlines(x=coun_camp_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')

        self.format_plot(ax)

        if flag_ratio:
            plt.ylim(min_value, max_value)
            # plt.title(f'Screen time ratio between PM and opposition leaders')
            plt.ylabel('screen time (percentage) %')
            plt.xlabel('')
        else:
            plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
            plt.ylabel('screen time (seconds)')
            plt.xlabel('')
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(f'figures/{save_name}', dpi=250)
        plt.show()

    def plot_pm_opposition_month(self, flag_party=False, flag_ratio=False, flag_pm_date=False, max_value=None, **kwargs):
        fig, ax = self._init_plot_()

        # Here we will do a hack to re-use previous functions --> change year to year_month
        df = self.res_df_all_info_from_to.copy()
        df['year'] = self.res_df_all_info_from_to['vid'].str[:7]
        res_corrected = self.compute_screen_seconds(df, flag_correct=False)
        mask_no_overall = res_corrected['ID'] != 'Overall'
        mask_year_noall = res_corrected[mask_no_overall]['year'] != 'all'
        df_plot = res_corrected[mask_no_overall][mask_year_noall].sort_values(by=['year'])

        # Filter and add prime ministers dates for visualization
        pminister_dates = [convert_str_to_date(x) for x in PM_DATES_MONTH]
        idx_pminister_dates = [self.from_date <= dd <= self.to_date for dd in pminister_dates]
        new_pminister_date = [dd for dd, bb in zip(PM_DATES_MONTH, idx_pminister_dates) if bb]
        new_pminister_list = [mm for mm, bb in zip(PRIME_MINISTERS, idx_pminister_dates) if bb]

        # From https://en.wikipedia.org/wiki/Leader_of_the_Opposition_(Japan)
        # Ignoring gap between Renho (2016/10/01-2017/07/27) and Maehara (2017/09/01-2017/10/23)
        # Hatoyama is the opposition leader since 1999_09, but for filtering issues, let's put the same date as 1st PM
        new_oppleader_date = ['2001_05', '2002_12', '2004_08', '2005_09', '2006_04',
                              '2009_05', '2009_09', '2012_09', '2012_12', '2014_12',
                              '2016_10', '2017_09', '2017_10']
        new_oppleader_list = [['Yukio_Hatoyama'], ['Naoto_Kan'], ['Katsuya_Okada'], ['Seiji_Maehara'], ['Ichiro_Ozawa'],
                    ['Yukio_Hatoyama'], ['Sadakazu_Tanigaki'], ['Shinzo_Abe'], ['Banri_Kaieda'], ['Katsuya_Okada'],
                    ['Renho_Renho'], ['Seiji_Maehara'], ['Yukio_Edano']]
        # opp_list = [['Takenori_Kanzaki'], ['Takenori_Kanzaki'], ['Takenori_Kanzaki'], ['Takenori_Kanzaki', 'Akihiro_Ota'],
        #             ['Akihiro_Ota'], ['Akihiro_Ota'], ['Mizuho_Fukushima'], [], [], [], [], [], []]

        idx_oppleader_dates = filter_list_by_date(new_oppleader_date, from_date=self.from_date, to_date=self.to_date)
        new_oppleader_date = [dd for dd, bb in zip(new_oppleader_date, idx_oppleader_dates) if bb]
        new_oppleader_list = [mm for mm, bb in zip(new_oppleader_list, idx_oppleader_dates) if bb]

        pm_df_list = []
        opp_df_list = []
        # Bug when analysing a period without change of PM.
        for i, pm in enumerate(PRIME_MINISTERS):
            mask_pm = (df_plot['year'] >= PM_DATES_MONTH[i]) & \
                      (df_plot['year'] < PM_DATES_MONTH[i + 1] if len(PM_DATES_MONTH) > i + 1 else True) & \
                      (df_plot['ID'] == pm)
            pm_df_list.append(mask_pm)

        for i, opp in enumerate(OP_LEADERS):
            mask_opp = (df_plot['year'] >= OP_DATES_MONTH[i]) & \
                       (df_plot['year'] < OP_DATES_MONTH[i + 1] if len(OP_DATES_MONTH) > i + 1 else True) & \
                       (df_plot['ID'] == opp)
            opp_df_list.append(mask_opp)

        # for i, (pm, opp) in enumerate(zip(new_pminister_list, opp_list)):
        #     mask_pm = (df_plot['year'] >= new_pminister_date[i]) & \
        #               (df_plot['year'] <= new_pminister_date[i + 1] if len(new_pminister_date) > i + 1 else True) & \
        #               (df_plot['ID'] == pm)
        #     # mask_pm = df_plot['ID'] == pm
        #     pm_party = df_plot[df_plot['ID'] == pm]['party'].unique()[0]
        #     opp_parties = [p for p in PARTIES if p != pm_party]
        #     mask_opp = (df_plot['year'] >= new_pminister_date[i]) & \
        #                (df_plot['year'] <= new_pminister_date[i + 1] if len(new_pminister_date) > i + 1 else True) & \
        #                (df_plot['party'].isin(opp_parties))
        #     pm_df_list.append(mask_pm)
        #     opp_df_list.append(mask_opp)

        mask_glob = sum(pm_df_list + opp_df_list) > 0
        mask_glob_pm = sum(pm_df_list) > 0
        df_plot = df_plot[mask_glob]
        df_ratio_pm = (df_plot[mask_glob_pm].groupby(['year']).sum() / df_plot.groupby(['year']).sum()).fillna(0)

        pol_id = 'ID' if not flag_party else 'party'
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d', 'TP_c'])
        df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d'])
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_t'])

        if flag_party:
            parties_list = []
            for party in df_plot['party'].unique():
                mask_party = df_plot['party'] == party
                for year in df_plot['year'][mask_party].unique():
                    mask_year = df_plot['year'][mask_party] == year
                    parties_list.append([party, year, 'TP', df_plot['value'][mask_party][mask_year].sum()])

            df_plot = pd.DataFrame(data=parties_list, columns=['party', 'year', 'variable', 'value'])
            color_idx_list = [PARTIES.index(party) for party in df_plot[pol_id].unique()]

        else:
            color_idx_list = [POLITICIANS.index(pol) for pol in df_plot[pol_id].unique()]

        # Care here for parties
        color_palette_mod = get_politician_palette(df_plot[pol_id].unique())
        n_colors = len(df_plot[pol_id].unique())
        if flag_ratio:
            # Should be called something like "ratio_pm_occ" but it's made to reuse posterior code
            df_ratio_pm['year'] = df_ratio_pm.index
            df_ratio_pm['value'] = df_ratio_pm['TP_d'] * 100
            df_plot = df_ratio_pm
            line_secs = sns.lineplot(data=df_plot, y='value', x='year', ax=ax)
            max_value = 100 if max_value is None else max_value
        else:
            line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue=pol_id,
                                     palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)
            max_value = df_plot['value'].max() if max_value is None else max_value
        # New ticks corresponding to each year instead of the whole year_month (unreadable)
        new_xticks = []
        for year_month in df_plot['year'].unique():
            year = year_month.split('_')[0]
            if year not in new_xticks:
                new_xticks.append(year)
            else:
                new_xticks.append(' ')

        line_secs.set_xticklabels(new_xticks, size=8, rotation=60, ha='right')
        line_secs.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)

        idx_repr = filter_list_by_date(REPRESENTATIVES_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
        idx_coun = filter_list_by_date(COUNCILLORS_DATES_MONTH, from_date=self.from_date, to_date=self.to_date)
        repr_list = [dd for dd, bb in zip(REPRESENTATIVES_DATES_MONTH, idx_repr) if bb]
        coun_list = [dd for dd, bb in zip(COUNCILLORS_DATES_MONTH, idx_coun) if bb]

        if flag_pm_date:
            plt.vlines(x=new_pminister_date, ymin=0, ymax=max_value, linestyles='dashed', colors='gray', label='new_pminister')
        plt.vlines(x=repr_list, ymin=0, ymax=max_value, linestyles='solid', colors='gray', label='repr_election')
        # plt.vlines(x=coun_list, ymin=0, ymax=max_value, linestyles='dotted', colors='gray', label='coun_election')

        self.format_plot(ax)
        # if args.mode == 'gt':
        #     plt.ylim(0, 4000)
        if flag_ratio:
            plt.ylim(0, max_value)
            # plt.title(f'Screen time ratio between PM and opposition leaders')
            plt.ylabel('screen time (percentage) %')
            plt.xlabel('')
        else:
            plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
            plt.ylabel('screen time (seconds)')
            plt.xlabel('')
        plt.tight_layout()
        # plt.savefig('figures/ratio_pm_opp.png', dpi=250)
        plt.savefig(f'figures/{self.channel}-ratio_pm_opp.pdf', dpi=250)
        plt.show()

    def plot_ratio_pm_opposition_day(self, **kwargs):
        fig, ax = self._init_plot_()
        #  flag_party=False, flag_ratio=False
        flag_party = kwargs.get('flag_party', False)
        flag_ratio = kwargs.get('flag_ratio', False)
        flag_correct = kwargs.get('flag_correct', False)

        # Here we will do a hack to re-use previous functions --> change year to year_month_day
        df = self.res_df_all_info_from_to.copy()
        df['year'] = self.res_df_all_info_from_to['vid'].str[:10]
        res_corrected = self.compute_screen_seconds(df, flag_correct=flag_correct)
        mask_no_overall = res_corrected['ID'] != 'Overall'
        mask_year_noall = res_corrected[mask_no_overall]['year'] != 'all'
        df_plot = res_corrected[mask_no_overall][mask_year_noall].sort_values(by=['year'])

        # Politics information
        new_pminister_date = ['2001_04_23', '2003_11_10', '2005_09_12', '2006_09_26',
                              '2007_09_26', '2008_09_24', '2009_09_16', '2010_06_08', '2011_09_02', '2012_12_16']
        new_pminister_list = ['Junichiro_Koizumi', 'Junichiro_Koizumi', 'Junichiro_Koizumi', 'Shinzo_Abe',
                              'Yasuo_Fukuda', 'Taro_Aso', 'Yukio_Hatoyama', 'Naoto_Kan', 'Yoshihiko_Noda', 'Shinzo_Abe']
        opp_list = [['Takenori_Kanzaki'], ['Takenori_Kanzaki'], ['Takenori_Kanzaki'], ['Takenori_Kanzaki', 'Akihiro_Ota'],
                    ['Akihiro_Ota'], ['Akihiro_Ota'], ['Mizuho_Fukushima'], [], [], []]

        pm_df_list = []
        opp_df_list = []
        for i, (pm, opp) in enumerate(zip(new_pminister_list, opp_list)):
            mask_pm = (df_plot['year'] >= new_pminister_date[i]) & \
                      (df_plot['year'] <= new_pminister_date[i + 1] if len(new_pminister_date) > i + 1 else True) & \
                      (df_plot['ID'] == pm)
            # mask_pm = df_plot['ID'] == pm
            mask_opp = (df_plot['year'] >= new_pminister_date[i]) & \
                       (df_plot['year'] <= new_pminister_date[i + 1] if len(new_pminister_date) > i + 1 else True) & \
                       (df_plot['ID'].isin(opp))
            pm_df_list.append(mask_pm)
            opp_df_list.append(mask_opp)
        mask_glob = sum(pm_df_list + opp_df_list) > 0
        df_plot = df_plot[mask_glob]

        pol_id = 'ID' if not flag_party else 'party'
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d', 'TP_c'])
        df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_d'])
        # df_plot = pd.melt(df_plot, id_vars=[pol_id, 'year'], value_vars=['TP_t'])

        if flag_party:
            parties_list = []
            for party in df_plot['party'].unique():
                mask_party = df_plot['party'] == party
                for year in df_plot['year'][mask_party].unique():
                    mask_year = df_plot['year'][mask_party] == year
                    parties_list.append([party, year, 'TP', df_plot['value'][mask_party][mask_year].sum()])

            df_plot = pd.DataFrame(data=parties_list, columns=['party', 'year', 'variable', 'value']).sort_values(
                by=['year'])
            color_idx_list = [PARTIES.index(party) for party in df_plot[pol_id].unique()]

        else:
            color_idx_list = [POLITICIANS.index(pol) for pol in df_plot[pol_id].unique()]

        # Care here for parties
        color_palette_mod = get_politician_palette(df_plot[pol_id].unique())
        n_colors = len(df_plot[pol_id].unique())
        line_secs = sns.lineplot(data=df_plot, y='value', x='year', hue=pol_id,
                                 palette=sns.color_palette(palette=color_palette_mod, n_colors=n_colors), ax=ax)
        # New ticks corresponding to each year instead of the whole year_month (unreadable)
        new_xticks = []
        for year_month in df_plot['year'].unique():
            year = year_month.split('_')[0]
            if year not in new_xticks:
                new_xticks.append(year)
            else:
                new_xticks.append(' ')

        line_secs.set_xticklabels(new_xticks, size=8, rotation=60, ha='right')
        line_secs.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=6, prop={'size': 6}, shadow=True)
        # election_list = ['2003_11_09', '2005_09_11', '2009_08_30', '2012_12_16']
        # 2001_04_26, 2003_11_09 and 2005_09_11 videos are not available
        election_list = ['2003_11_10', '2005_09_12', '2009_08_30', '2012_12_16']
        plt.vlines(x=election_list, ymin=0, ymax=df_plot['value'].max(), linestyles='dashed', colors='gray',
                   label='election')
        plt.vlines(x=new_pminister_date, ymin=0, ymax=df_plot['value'].max(), linestyles='solid', colors='gray',
                   label='new_pminister')

        self.format_plot(ax)
        if flag_ratio:
            plt.ylim(0, 100)
            plt.title(f'Screen time ratio between PM and opposition parties')
            plt.ylabel('screen time (percentage) %')
            plt.xlabel('')
        else:
            plt.title(f'{self.detector}-{self.feats}-{self.mod_feat}')
            plt.ylabel('screen time (seconds)')
            plt.xlabel('')

        plt.tight_layout()
        plt.show()


def main_gt():
    res_statistics = ResStatistics(RES_PATH, channel=args.channel, mode='gt', from_date_str=FROM_DATE, to_date_str=TO_DATE)
    res_statistics.print_prec_rec()
    res_statistics.print_screen_time()
    res_statistics.print_correction_error()
    # res_statistics.plot_correction_error()
    # res_statistics.plot_screen_time()
    # res_statistics.plot_tp_fp_fn()
    # res_statistics.plot_tp_fp_fn_pol_year('Yoshihiko_Noda')
    # res_statistics.plot_screen_time_timeline()
    # res_statistics.plot_screen_time_timeline_month(flag_party=False, flag_pm=False)
    # res_statistics.plot_screen_time_timeline_day(flag_party=False, flag_pm=False)
    res_statistics.plot_pm_opposition_month(flag_ratio=True)
    res_statistics.plot_ratio_pm_opposition_day()


def main_demo():
    # res_statistics_gt = ResStatistics(RES_PATH, mode='gt')
    res_statistics_demo = ResStatistics(RES_PATH_DEMO, channel=args.channel, mode='demo', from_date_str=FROM_DATE, to_date_str=TO_DATE)
    # res_statistics_demo.print_screen_time()
    # res_statistics_demo.plot_screen_time_timeline()
    res_statistics_demo.print_top_appearances(POLITICIANS_TO_TEST)
    common_name = f'{str(res_statistics_demo.from_date)}-{str(res_statistics_demo.to_date)}.pdf'
    if len(PARTIES_TO_TEST) == 1:
        common_name = f'{PARTIES_TO_TEST[0]}-{common_name}'
    if TOP_K > 0:
        save_name = f'{args.channel}-top{TOP_K}-{common_name}'
    else:
        save_name = f'{args.channel}-{common_name}'

    # res_statistics_demo.plot_screen_time_timeline_month(politicians=POLITICIANS_TO_TEST, parties=PARTIES_TO_TEST, topk=TOP_K, flag_party=False, flag_pm=True, flag_prev_pm=False, flag_dates=True, save_name=save_name, flag_perc=True, flag_smooth=True)
    # res_statistics_demo.plot_screen_time_timeline_day(politicians=POLITICIANS_TO_TEST, parties=PARTIES_TO_TEST, topk=TOP_K, flag_party=False, flag_pm=True, flag_dates=True, flag_perc=True, flag_smooth=True, save_name='daily_avg')
    # res_statistics_demo.plot_pm_opposition_month(flag_ratio=True, flag_pm_date=True, max_value=102)
    res_statistics_demo.plot_pm_opp()
    # res_statistics_demo.plot_pm_opposition_month(flag_ratio=True, flag_pm_date=True)
    # res_statistics_demo.plot_ratio_pm_opposition_day()


def main_train():
    # Statistics from the dataset, should also read from FPs and stuff
    res_statistics = ResStatistics(RES_PATH, mode='train', channel=args.channel, detector=DETECTOR, feats=FEATS,
                                   mod_feat=MOD_FEAT, from_date_str=FROM_DATE, to_date_str=TO_DATE)
    res_statistics.print_prec_rec()
    res_statistics.print_screen_time()

    # res_statistics.plot_pm_opposition_month(flag_ratio=True)
    # res_statistics.plot_ratio_pm_opposition_day(flag_correct=False)


def main_train_valid():
    # We will use the information we already have and emulate a train/validation split to compute the error between
    # detections and the corrected detections
    res_statistics = ResStatistics(RES_PATH, channel=args.channel, mode='gt', from_date_str=FROM_DATE, to_date_str=TO_DATE)
    res_statistics.split_train_valid()
    # Recompute correction factor and metrics based on train split
    res_statistics.gt_correction_factor = res_statistics.compute_correction_factor_year(res_statistics.train_res_df_all_info_from_to)
    res_statistics.res_df_metrics_overall = res_statistics.compute_metrics(res_statistics.train_res_df_all_info_from_to)
    # Recompute corrected screen seconds based on validation split
    res_statistics.res_corrected = res_statistics.compute_screen_seconds(res_statistics.valid_res_df_all_info_from_to)
    # Compute correction error between
    res_statistics.print_correction_error()
    res_statistics.plot_correction_error()


if __name__ == '__main__':
    # Here read detector and feature modifier from args
    parser = get_parser()
    args = parser.parse_args()
    POLITICIANS_TO_TEST = args.who
    PARTIES_TO_TEST = args.party
    TOP_K = args.top
    FROM_DATE = args.from_date
    TO_DATE = args.to_date
    DETECTOR = args.detector
    MOD_FEAT = args.mod_feat
    FEATS = args.feats

    RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / 'results'
    # RES_PATH = Path(args.res_path) / Path(args.channel) / Path(args.mode) / 'results-year'

    if args.mode.lower() == 'gt':
        main_gt()
    elif args.mode.lower() == 'demo':
        RES_PATH_DEMO = Path(args.res_path) / Path(args.channel) / Path(args.mode) / 'results'
        main_demo()
    elif args.mode.lower() == 'train':
        main_train()
    elif args.mode.lower() == 'val':
        main_train_valid()
        # for i in range(10):
        #     main_train_valid()
