from __future__ import division
import multiprocessing as mp
import itertools
import numpy as np
from scipy import interpolate
from pylab import flipud
import pandas as pd
from pandas.core.categorical import Categorical
import re
from collections import defaultdict
from multiflexxlib import plotting
from multiflexxlib import ub
from multiflexxlib.ub import UBMatrix, etok, ktoe, angle_to_qs
import pyclipper
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import matplotlib.path as mpl_path
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.axisartist import Subplot
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
import pickle
import warnings
import os
import pkg_resources

try:
    import tkinter
    from tkinter import filedialog
except ImportError:
    import Tkinter as tkinter
    import tkFileDialog as filedialog


NUM_CHANNELS = 31
EF_LIST = [2.5, 3.0, 3.5, 4.0, 4.5]
CHANNEL_SEPARATION = 2.5
NORM_FACTOR = [1.0, 1.16, 1.23, 1.30, 1.27]

try:
    DETECTOR_WORKING = np.loadtxt(pkg_resources.resource_filename(__name__, 'res/alive.csv'))
except IOError:
    print('Dead detector map not found - assuming all working.')
    DETECTOR_WORKING = np.ones([NUM_CHANNELS, len(EF_LIST)])

try:
    WEIGHTS = np.loadtxt(pkg_resources.resource_filename(__name__, 'res/weights.csv'), delimiter=',')
except IOError:
    print('Boundary angle channel strategy not defined - assuming equal weights.')
    WEIGHTS = np.ones([NUM_CHANNELS, len(EF_LIST)])

try:
    INTENSITY_COEFFICIENT = np.loadtxt(pkg_resources.resource_filename(__name__, 'res/int_corr.csv'), delimiter=',')
except IOError:
    print('Intensity correction matrix not found - assuming all ones.')
    INTENSITY_COEFFICIENT = np.ones(NUM_CHANNELS)
INTENSITY_COEFFICIENT = INTENSITY_COEFFICIENT / NORM_FACTOR


def _nan_float(string):
    try:
        return float(string)
    except ValueError:
        if '*' in string:
            return np.NaN
        else:
            raise


def _nan_int(string):
    try:
        return int(string)
    except ValueError:
        if '*' in string:
            return np.NaN
        else:
            raise


def _extract_ki_from_header(en, fx, kfix):
    e_fix = ktoe(kfix)
    if fx == 2:
        ei = e_fix + en
        return etok(ei)
    elif fx == 1:
        ei = e_fix - en
        return etok(ei)
    else:
        raise ValueError('Invalid FX value: 2 for fix kf, 1 for fix ki, got %d' % fx)


def _number_to_scan(num):
    if isinstance(num, int):
        return '{:06d}'.format(num)
    elif isinstance(num, str):
        return num
    else:
        raise TypeError('Expecting a number or a numeric string to convert to FLEXX scan file number.')


def _parse_flatcone_line(line):
    data = np.array([_nan_int(x) for x in line.split()])
    array = np.reshape(data, (-1, len(EF_LIST)))[0: -1, :]  # throws out last line which is only artifact
    ang_channels = np.asarray([np.arange(1, NUM_CHANNELS + 1)]).T  # starts at 1 to match stickers
    array_with_ch_no = np.hstack([ang_channels, array])
    dataframe_flatcone = pd.DataFrame(data=array_with_ch_no, columns=['aCh', 'e1', 'e2', 'e3', 'e4', 'e5'])
    dataframe_flatcone.set_index('aCh', inplace=True)
    return dataframe_flatcone


def _parse_param_line(line):
    line_name = line[0:5]
    line_body = line[6:].strip()
    if line_name == 'COMND':
        no_points = int(re.findall('(?<=NP)[\s\t0-9]*', line_body)[0].strip())
        return line_name, {'value': line_body, 'NP': no_points}
    elif '=' not in line_body:
        return line_name, line_body
    else:
        equations = line_body.split(',')
        line_dict = {}
        for eq in equations:
            param_name, value_raw = [x.strip() for x in eq.split('=')]
            try:
                value = _nan_float(value_raw)
            except ValueError:
                value = value_raw
            line_dict[param_name] = value
        return line_name, line_dict


def parse_ill_data(file_object, start_flag='DATA_:\n'):
    """
    Parses ILL TASMAD scan files.
    :param file_object: Handle to opened file or stream. Or alternately path to scan file.
    :param start_flag: Start flag of data section. Omit for default.
    :return: (header_dict, dataframe)
    """
    # first parse headers
    try:
        file_object.seek(0, 0)
    except AttributeError:
        file_object = open(file_object, 'r')
    text_data = file_object.read()
    headers = re.findall('^[A-Z_]{5}:.*', text_data, re.MULTILINE)
    header_dict = defaultdict(dict)
    for line in headers:
        line_name, line_body = _parse_param_line(line)
        if type(line_body) is dict:
            header_dict[line_name].update(line_body)
        else:
            header_dict[line_name].update({'value': line_body})
    # then parse scan parameters and counts
    data_section = text_data[text_data.find(start_flag) + len(start_flag) + 1:]
    column_names = data_section.splitlines()[0].split()
    # line only w 0-9, . -, spc, tab
    parameters_text_lines = re.findall('^[0-9*\-\s\t.]+?$', data_section, re.MULTILINE)
    parameters_value_array = np.asarray([[_nan_float(num) for num in line.split()] for line in parameters_text_lines])
    data_frame = pd.DataFrame(data=parameters_value_array, columns=column_names)
    data_frame['PNT'] = data_frame['PNT'].astype('int16')
    df_clean = data_frame.T.drop_duplicates().T
    # parse flatcone data if present
    flat_all = re.findall('(?<=flat: )[0-9w\s\t\n*]+(?=endflat)', text_data, re.MULTILINE)
    flat_number_lines = len(flat_all)
    if len(df_clean) == 0:
        raise ValueError('file %s does contain any data.' % file_object.name)
    if len(df_clean) - flat_number_lines <= 1:  # sanity check: only 1 missing flatcone line is acceptable
        flat_frames = []
        for nth, line in enumerate(flat_all):
            try:
                flat_frames.append(_parse_flatcone_line(line))
            except ValueError:
                raise ValueError('point %d in file %s is faulty.' % (nth + 1, file_object.name))
        if len(df_clean) - flat_number_lines == 1:
            df_clean.drop(df_clean.index[-1], inplace=True)  # if only one line is missing then just drop last line
        df_clean = df_clean.assign(flat=flat_frames)
    else:
        pass
    return dict(header_dict), df_clean


def ub_from_header(scan_header):
    # type: ((dict, Scan)) -> UBMatrix
    """
    Make a UBMatrix object from TASMAD scan header.
    :param scan_header:
    :return: UBMatrix object
    """
    if isinstance(scan_header, Scan):
        scan_header = scan_header.header
    param = scan_header['PARAM']
    lattice_parameters = [param['AS'], param['BS'], param['CS'], param['AA'], param['BB'], param['CC']]
    hkl1 = [float(param['AX']), float(param['AY']), float(param['AZ'])]
    hkl2 = [float(param['BX']), float(param['BY']), float(param['BZ'])]
    ub_matrix = UBMatrix(lattice_parameters, hkl1, hkl2)
    return ub_matrix


class Scan(object):
    """
    Reads a TASMAD scan file, extracts metadata and do essential conversions. Assumes const-Ei scan!
    Should not be instantiated by directly invoking constructor. Use read_mf_scan() or read_mf_scans() instead.
    """
    def __init__(self, file_name, ub_matrix=None, intensity_matrix=None, a3_offset=0.0, a4_offset=0.0):
        """
        Scan object.
        :param file_name: File name of TASMAD scan file.
        :param ub_matrix: UBMatrix object to be used. Omit to generate from file header.
        :param intensity_matrix: Intensity correction matrix to be used. Omit to use default.
        """
        file_name = _number_to_scan(file_name)
        f = open(file_name)
        self.header, self.data = parse_ill_data(f)
        self.file_name = os.path.abspath(file_name)
        self._a3_offset = a3_offset
        self._a4_offset = a4_offset
        self._apply_offsets(a3_offset, a4_offset)
        if 'flat' not in self.data.columns:
            raise AttributeError('%s does not contain flatcone data.' % file_name)
        elif 'A3' not in self.header['STEPS'].keys():
            raise AttributeError('%s is not A3 scan.' % file_name)
        elif 'EI' in self.header['STEPS'].keys():
            raise AttributeError('%s is not a const-E scan.' % file_name)
        if intensity_matrix:
            self.intensity_matrix = intensity_matrix
        else:
            self.intensity_matrix = INTENSITY_COEFFICIENT

        if not ub_matrix:
            self.ub_matrix = ub_from_header(self.header)
        else:
            self.ub_matrix = ub_matrix

        self.converted_dataframes = []
        self._update_data_array()
        print('finished loading %s, a3_offset = %.2f, a4_offset = %.2f' % (file_name, self.a3_offset, self.a4_offset))

    @property
    def ki(self):
        try:
            ki = self.data.iloc[0]['KI']
        except KeyError:
            try:
                ki = etok(self.data.iloc[0]['EI'])
            except KeyError:
                ki = _extract_ki_from_header(self.header['POSQE']['EN'], self.header['PARAM']['FX'],
                                             self.header['PARAM']['KFIX'])
        return ki

    @property
    def tt(self):
        try:
            tt = self.data.iloc[-1]['TT']  # takes final value as signature value for the scan
        except KeyError:
            tt = None
        return tt

    @property
    def mag(self):
        try:
            mag = self.data.iloc[-1]['MAG']
        except KeyError:
            mag = None
        return mag

    @property
    def ei(self):
        """
        Initial Energy (Ei) of scan.
        :return: Ei in meV
        """
        return ktoe(self.ki)

    @property
    def np_planned(self):
        """
        Total planned points in scan based on command.
        :return: Integer steps.
        """
        return self.header['COMND']['NP']

    @property
    def np_actual(self):
        """
        Actual finished points. Different from planned if scan is unfinished.
        :return: Integer steps.
        """
        return len(self.data)

    @property
    def scan_number(self):
        """
        Scan number.
        :return: String of scan file name, which should be numeric for TASMAD files.
        """
        return os.path.split(self.file_name)[1]

    @property
    def a3_offset(self):
        return self._a3_offset

    @property
    def a4_offset(self):
        return self._a4_offset

    @a3_offset.setter
    def a3_offset(self, value):
        a3o_old = self.a3_offset
        a3o_new = value
        a3_add = a3o_new - a3o_old
        self._apply_offsets(a3_add, 0.0)
        self._update_data_array()
        self._a3_offset = a3o_new

    @a4_offset.setter
    def a4_offset(self, value):
        a4o_old = self.a3_offset
        a4o_new = value
        a4_add = a4o_new - a4o_old
        self._apply_offsets(0.0, a4_add)
        self._update_data_array()
        self._a4_offset = a4o_new

    @property
    def planned_locus_list(self):
        kf_list = [etok(e) for e in EF_LIST]
        a3_start, a3_end_actual, a3_end_planned = self.a3_ranges
        a4_start, a4_end_actual, a4_end_planned = self.a4_ranges
        return [calculate_locus(self.ki, kf, a3_start, a3_end_planned, a4_start, a4_end_planned,
                                self.ub_matrix, expand_a3=True) for kf in kf_list]

    @property
    def actual_locus_list(self):
        kf_list = [etok(e) for e in EF_LIST]
        a3_start, a3_end_actual, a3_end_planned = self.a3_ranges
        a4_start, a4_end_actual, a4_end_planned = self.a4_ranges
        return [calculate_locus(self.ki, kf, a3_start, a3_end_actual, a4_start, a4_end_actual,
                                self.ub_matrix) for kf in kf_list]

    def _apply_offsets(self, a3_offset, a4_offset):
        self.data.A3 = self.data.A3 + a3_offset
        self.data.A4 = self.data.A4 + a4_offset

    def _update_data_array(self):
        num_ch = NUM_CHANNELS
        channel_separation = CHANNEL_SEPARATION
        num_flat_frames = len(self.data)
        # an numpy array caching a3, a4 angles and monitor counts, shared across all energy channels
        a3_a4_mon_array = np.zeros([num_flat_frames * num_ch, 3])

        a4_angle_mask = np.linspace(-channel_separation * (num_ch - 1) / 2,
                                    channel_separation * (num_ch - 1) / 2, num_ch)

        for i in range(num_flat_frames):
            a3_a4_mon_array[i * num_ch: (i + 1) * num_ch, 0] = self.data.loc[i, 'A3']
            a3_a4_mon_array[i * num_ch: (i + 1) * num_ch, 1] = self.data.loc[i, 'A4'] + a4_angle_mask
            a3_a4_mon_array[i * num_ch: (i + 1) * num_ch, 2] = self.data.loc[i, 'M1']

        data_template = pd.DataFrame(index=range(num_flat_frames * num_ch),
                                     columns=['A3', 'A4', 'MON', 'px', 'py', 'pz', 'h', 'k', 'l',
                                              'counts', 'valid', 'coeff', 'ach', 'point'], dtype='float64')
        # data_template = data_template.assign(file=pd.Series(index=range(num_flat_frames * num_ch)), dtype='str')
        # filename source not recorded for now due to performance concerns
        data_template.loc[:, ['A3', 'A4', 'MON']] = a3_a4_mon_array
        self.converted_dataframes = [data_template.copy() for _ in range(len(EF_LIST))]
        for ef_channel_num, ef in enumerate(EF_LIST):
            qs = self.ub_matrix.angle_to_qs(self.ki, etok(ef), a3_a4_mon_array[:, 0], a3_a4_mon_array[:, 1])
            self.converted_dataframes[ef_channel_num].loc[:, ['px', 'py', 'pz']] = self.ub_matrix.convert(qs, 'sp').T
            self.converted_dataframes[ef_channel_num].loc[:, ['h', 'k', 'l']] = self.ub_matrix.convert(qs, 'sr').T
        coefficient = INTENSITY_COEFFICIENT
        detector_working = DETECTOR_WORKING
        for point_num in range(num_flat_frames):
            flatcone_array = np.asarray(self.data.loc[point_num, 'flat'])
            for ef_channel_num in range(len(EF_LIST)):
                dataframe = self.converted_dataframes[ef_channel_num]
                rows = slice(point_num * num_ch, (point_num + 1) * num_ch - 1, None)
                dataframe.loc[rows, 'counts'] = flatcone_array[:, ef_channel_num]
                dataframe.loc[rows, 'valid'] = detector_working[:, ef_channel_num]
                dataframe.loc[rows, 'coeff'] = coefficient[:, ef_channel_num]
                dataframe.loc[rows, 'point'] = self.data.loc[point_num, 'PNT']
                dataframe.loc[rows, 'ach'] = range(1, num_ch + 1)
                # dataframe.loc[rows, 'file'] = self.file_name

    @property
    def a3_ranges(self):
        a3_start = self.data.iloc[0]['A3']
        a3_end_actual = self.data.iloc[-1]['A3']
        try:
            a3_end_planned = self.header['VARIA']['A3'] + \
                             self.header['STEPS']['A3'] * (self.header['COMND']['NP'] - 1) + self._a3_offset
        except KeyError:
            a3_end_planned = a3_end_actual
        return a3_start, a3_end_actual, a3_end_planned

    @property
    def a4_ranges(self):
        a4_start = self.header['VARIA']['A4'] + self._a4_offset  # A4 is not necessarily outputted in data
        if 'A4' not in self.header['STEPS']:
            a4_end_planned = a4_start
            a4_end_actual = a4_start
        else:
            a4_end_planned = self.header['VARIA']['A4'] + \
                             self.header['STEPS']['A4'] * (self.header['COMND']['NP'] - 1) + self._a4_offset
            a4_end_actual = self.data.iloc[-1]['A4']
        return a4_start, a4_end_actual, a4_end_planned

    def to_csv(self, file_name=None, channel=None):
        pass


def make_bin_edges(values, tolerance=0.2):
    # type: ((list, pd.Series), float) -> list
    """
    :param values: An iterable list of all physical quantities, repetitions allowed.
    :param tolerance: maximum difference in value for considering two points to be the same.
    :return: a list of bin edges

    Walks through sorted unique values, if a point is further than tolerance away from the next, a bin edge is
    dropped between the two points, otherwise no bin edge is added. A beginning and ending edge is added at
    tolerance / 2 further from either end.
    """
    values_array = np.array(values).ravel()
    unique_values = np.asarray(list(set(values_array)))
    unique_values.sort()
    bin_edges = [unique_values[0] - tolerance / 2]
    for i in range(len(unique_values) - 1):
        if unique_values[i+1] - unique_values[i] > tolerance:
            bin_edges.append((unique_values[i] + unique_values[i+1]) / 2)
        else:
            pass

    bin_edges.append(unique_values[-1] + tolerance / 2)

    return bin_edges


def _merge_locus(locus_list):
    clipper = pyclipper.Pyclipper()
    for locus in locus_list:
        clipper.AddPath(pyclipper.scale_to_clipper(locus), pyclipper.PT_SUBJECT)

    merged_locus = np.array(pyclipper.scale_from_clipper(clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO)))
    return merged_locus


def _merge_scan_points(data_frames, a3_tolerance=0.2, a4_tolerance=0.2):
    """
    Bins actual detector counts together from multiple runs.
    :param data_frames: Pandas data frames from Scan objects.
    :param angle_tolerance: Max angle difference before two angles are considered discreet.
    :return: An intermediate data structure even I don't really remember.
    """
    joined_frames = pd.concat(data_frames, axis=0, ignore_index=True)
    joined_frames = joined_frames.assign(counts_norm=joined_frames.counts/joined_frames.coeff)
    joined_frames = joined_frames.drop(joined_frames[joined_frames.valid != 1].index)  # delete dead detectors

    a3_cuts = bin_and_cut(joined_frames.A3, tolerance=a3_tolerance)
    a4_cuts = bin_and_cut(joined_frames.A4, tolerance=a4_tolerance)
    group = joined_frames.groupby([a3_cuts, a4_cuts])
    sums = group['counts', 'counts_norm', 'MON'].sum()
    means = group['A3', 'A4', 'px', 'py', 'pz', 'h', 'k', 'l'].mean()
    error_bars = np.sqrt(sums.counts)
    per_monitor = sums.counts_norm / sums.MON
    result = pd.concat([sums, means], axis=1)
    result = result.assign(err=error_bars)
    result = result.assign(permon=per_monitor)
    result = result.dropna()
    return result.reset_index(drop=True)


def bin_and_cut(data, tolerance=0.2):
    # type: (pd.Series, float) -> Categorical
    """
    Applies adaptive binning and return a pandas.Categorical cut object
    :param data: a series or list of numbers. Repetition allowed.
    :param tolerance: Binning tolerance.
    :return: pd.cut results.
    """
    bin_edges = make_bin_edges(data, tolerance)
    cut = pd.cut(data, bin_edges)
    return cut


def series_to_binder(items):
    """
    Helper function for converting list to _DataBinder object. The _DataBinder class is just for overriding str method.
    :param items: Anything that makes sense with list(items).
    :return:
    """
    # type: (pd.Series) -> _DataBinder
    return _DataBinder(list(items))


def bin_scans(list_of_data,  # type: ['Scan']
              nan_fill=0, ignore_ef=False,
              en_tolerance=0.05, tt_tolerance=1.0, mag_tolerance=0.05, a3_tolerance=0.2, a4_tolerance=0.2,
              angle_voronoi=False):
    # type: (...)-> BinnedData
    """
    Bin raw Scan objects into BinnedData object.
    :param list_of_data: a list of Scan objects.
    :param nan_fill: how to deal NaNs in metadata such as temperature. Default is fill 0.
    :param ignore_ef: Not implemented.
    :param en_tolerance: Energy binning tolerance.
    :param tt_tolerance: Temperature binning tolerance.
    :param mag_tolerance: Magnetic field binning tolerance.
    :param a3_tolerance: A3 angle binning tolerance of data points.
    :param a4_tolerance: A4 angle binning tolerance of data points.
    :param angle_voronoi: Performs Voronoi partition in angle plane instead of reciprocal plane.
    :return: BinnedData object.
    """
    all_data = pd.DataFrame(index=range(len(list_of_data) * len(EF_LIST)),
                            columns=['name', 'ei', 'ef', 'en', 'tt', 'mag', 'points', 'locus_a', 'locus_p'],
                            dtype=object)
    file_names = [data.file_name for data in list_of_data]
    for i, scan in enumerate(list_of_data):
        for j in range(len(EF_LIST)):
            ef = EF_LIST[j]
            all_data.loc[i * len(EF_LIST) + j, ['name', 'ei', 'ef', 'en']] = [scan.file_name, scan.ei, ef, scan.ei - ef]
            all_data.loc[i * len(EF_LIST) + j, ['tt', 'mag']] = [scan.tt, scan.mag]
            all_data.loc[i * len(EF_LIST) + j, ['points', 'locus_a', 'locus_p']] = [scan.converted_dataframes[j],
                                                                                    scan.actual_locus_list[j],
                                                                                    scan.planned_locus_list[j]]

    all_data = all_data.fillna(nan_fill)
    cut_ei = bin_and_cut(all_data.ei, en_tolerance)
    cut_en = bin_and_cut(all_data.en, en_tolerance)
    cut_tt = bin_and_cut(all_data.tt, tt_tolerance)
    cut_mag = bin_and_cut(all_data.mag, mag_tolerance)

    if ignore_ef:
        raise NotImplementedError('For the love of god do not try to mix data from different final energies!')
    else:
        grouped = all_data.groupby([cut_ei, cut_en, cut_tt, cut_mag])
    grouped_meta = grouped['ei', 'ef', 'en', 'tt', 'mag'].mean()
    grouped_data = grouped['points'].apply(series_to_binder).apply(lambda x:
                                                                   _MergedDataPoints(x, a3_tolerance, a4_tolerance))

    grouped_locus_a = grouped['locus_a'].apply(series_to_binder).apply(_MergedLocus)
    grouped_locus_p = grouped['locus_p'].apply(series_to_binder).apply(_MergedLocus)
    joined = pd.concat([grouped_meta, grouped_data, grouped_locus_a, grouped_locus_p], axis=1)
    index_reset = joined.dropna().reset_index(drop=True)
    return BinnedData(index_reset, file_names=file_names, ub_matrix=list_of_data[0].ub_matrix,
                      angle_voronoi=angle_voronoi)


def read_mf_scan(filename, ub_matrix=None, intensity_matrix=None, a3_offset=0.0, a4_offset=0.0):
    # type: (str, UBMatrix, np.ndarray) -> Scan
    """
    Reads TASMAD scan files.
    :param filename: TASMAD file name to read.
    :param ub_matrix: UBMatrix to be used. Omit to generate automatically.
    :param intensity_matrix: Int. matrix to use. Omit to use default.
    :param a3_offset: Value to be added to A3 angles in this scan file.
    :param a4_offset: Value to be added to A4 angles in this scan file.
    :return: Scan object
    """
    scan_object = Scan(filename, ub_matrix, intensity_matrix, a3_offset=a3_offset, a4_offset=a4_offset)
    return scan_object


def read_mf_scans(filename_list=None,  # type: ['str']
                  ub_matrix=None, intensity_matrix=None, processes=1, a3_offset=None, a4_offset=None):
    """
    # type: (...) -> ['Scan']
    Reads TASMAD scan files.
    :param filename_list: A list of TASMAD file names to read. User will be prompted for a folder if omitted.
    :param ub_matrix: UBMatrix to be used. Omit to generate automatically.
    :param intensity_matrix: Int. matrix to use. Omit to use default.
    :param processes: Number of processes.
    :param a3_offset: Number, list or None. Will be added to A3 angles if provided. Each element will be added to
    corresponding scan file if a list is provided. List length must match number of files.
    :param a4_offset: Number, list or None. Will be added to A4 angles if provided. Each element will be added to
    corresponding scan file if a list is provided. List length must match number of files.
    :return: A list containing resulting Scan objects.
    """
    if filename_list is None:
        path = ask_directory('Folder containing data')
        filename_list = list_flexx_files(path)
    if len(filename_list) == 0:
        raise FileNotFoundError('No file to read.')

    a3_offset_list = _expand_offset_parameter(a3_offset, filename_list)
    a4_offset_list = _expand_offset_parameter(a4_offset, filename_list)
    arg_list = []
    for name, a3o, a4o in zip(filename_list, a3_offset_list, a4_offset_list):
        arg_list.append((name, ub_matrix, intensity_matrix, a3o, a4o))
    if processes > 1:
        pool = mp.Pool(processes=processes)
        data_list = pool.starmap(read_mf_scan, arg_list)
    else:
        data_list = list(itertools.starmap(read_mf_scan, arg_list))
    return data_list


def _expand_offset_parameter(param, filename_list):
    length = len(filename_list)
    if param is None:
        return [0.0 for _ in range(length)]
    elif isinstance(param, (int, float)):
            return [param for _ in range(length)]
    elif isinstance(param, (list, tuple)):
        if len(filename_list) == len(param):
            return param
        else:
            raise ValueError('Offset list length and number of files mismatch.')
    elif isinstance(param, dict):
        param_filtered = {_number_to_scan(key): param[key] for key in param.keys()}
        offset_list = []
        for filename in filename_list:
            filename = os.path.split(filename)[1]
            try:
                offset_list.append(param_filtered[filename])
            except KeyError:
                offset_list.append(0.0)
        return offset_list
    else:
        raise TypeError('Offset should be None, a number, a list or a dict.')


def read_and_bin(filename_list=None, ub_matrix=None, intensity_matrix=None, processes=1,
                 en_tolerance=0.05, tt_tolerance=1.0, mag_tolerance=0.05, a3_tolerance=0.2, a4_tolerance=0.2,
                 a3_offset=None, a4_offset=None, angle_voronoi=False):
    """
    Reads and bins MultiFLEXX scan files together.
    :param filename_list: A list containing absolute or relative paths of TASMAD scan files to read. User will be
    prompted to choose a directory if omitted.
    :param ub_matrix: UBMatrix object to be used. Omit to generate from data headers.
    :param intensity_matrix: Intensity correction matrix to be used. Omit to use the default one.
    :param processes: Number of processes to use.
    :param en_tolerance: Energy tolerance before two values are considered discrete, default to 0.05meV.
    :param tt_tolerance: Temperature tolerance, default to 1.0K.
    :param mag_tolerance: Magnetic field tolerance, default to 0.05T.
    :param a3_tolerance: A3 angle tolerance, default is 0.2deg.
    :param a4_tolerance: A4 angle tolerance, default is 0.2deg.
    :param a3_offset: Angle value to be added into raw A3 angles, in degrees.
    :param a4_offset: Angle value to be added into raw A4 angles, in degrees.
    :param angle_voronoi: Whether to perform Voronoi tessellation in angles instead of Q-coordinates.
    :return: BinnedData object.
    """
    if filename_list is None:
        items = read_mf_scans(filename_list, ub_matrix, intensity_matrix, processes, a3_offset, a4_offset)
    else:
        if isinstance(filename_list, list):
            items = read_mf_scans(filename_list, ub_matrix, intensity_matrix, processes, a3_offset, a4_offset)
        elif os.path.isdir(filename_list):
            filename_list = list_flexx_files(filename_list)
            items = read_mf_scans(filename_list, ub_matrix, intensity_matrix, processes, a3_offset, a4_offset)
        else:
            raise ValueError('%s: Got a parameter that is neither a list nor a directory')
    df = bin_scans(items, en_tolerance=en_tolerance, tt_tolerance=tt_tolerance, mag_tolerance=mag_tolerance,
                   a3_tolerance=a3_tolerance, a4_tolerance=a4_tolerance, angle_voronoi=angle_voronoi)
    return df


class _DataBinder(list):
    """
    Helper class to override __str__ behaviour.
    """
    def __str__(self):
        return '%d items' % len(self)


class _MergedLocus(list):
    """
    Helper class to override __str__ behaviour.
    """
    def __init__(self, items):
        # type: (_DataBinder) -> None
        binned_locus = _merge_locus(items)
        super(_MergedLocus, self).__init__(binned_locus)

    def __str__(self):
        patches = len(self)
        total_vertices = float(np.sum([len(patch) for patch in self]))
        return '%dp %dv' % (patches, total_vertices)


class _MergedDataPoints(pd.DataFrame):
    # Helper class to override __str__ behaviour.
    def __init__(self, items, a3_tolerance=0.2, a4_tolerance=0.2):
        # type: (_DataBinder, float) -> None
        binned_points = _merge_scan_points(items, a3_tolerance=a3_tolerance, a4_tolerance=a4_tolerance)
        super(_MergedDataPoints, self).__init__(binned_points)

    def __str__(self):
        return '%d pts' % len(self)


class BinnedData(object):
    def __init__(self, source_dataframe, file_names, ub_matrix=None, angle_voronoi=False):
        # type: (pd.DataFrame, [str], UBMatrix) -> None
        """
        Should not be instantiated on its own.
        :param source_dataframe:
        :param file_names:
        :param ub_matrix:
        """
        self._file_names = file_names
        self.data = source_dataframe
        self.ub_matrix = ub_matrix
        self._generate_voronoi(angle_voronoi)
        self.angle_voronoi = angle_voronoi

    def file_names(self):
        """
        Files used in this dataset.
        :return: List of strings.
        """
        return self._file_names

    def __str__(self):
        return str(pd.concat((self.data[['ei', 'en', 'ef', 'tt', 'mag']],
                              self.data[['locus_a', 'locus_p', 'points']].astype('str')), axis=1))

    def _generate_voronoi(self, angle_voronoi):
        # TODO: defuse this iterator landmine: is this iterating through index column or what?
        if not angle_voronoi:
            list_of_lop = []
            for i, item in enumerate(self.data['points']):
                lop = plotting.generate_vpatch(item['px'], item['py'], self.ub_matrix.figure_aspect, max_cell=0.2)
                list_of_lop.append(lop)
            self.data = self.data.assign(voro=list_of_lop)
        else:
            list_of_lop = []
            for i, item in enumerate(self.data['points']):
                lop_angle = plotting.generate_vpatch(item['A3'], item['A4'], self.ub_matrix.figure_aspect, max_cell=2.5)
                lop_s = [self.ub_matrix.angle_to_qs(etok(self.data.ei[i]), etok(self.data.ef[i]),
                                                    poly[:, 0], poly[:, 1]) for poly in lop_angle]
                lop_p = [self.ub_matrix.convert(poly, 'sp') for poly in lop_s]
                lop_p_filtered = [poly.T[:, 0:2] for poly in lop_p]
                list_of_lop.append(lop_p_filtered)
            self.data = self.data.assign(voro=list_of_lop)

    def cut_voronoi(self, start, end, subset=None, precision=2, labels=None, monitor=True, plot=True):
        """
        1D-cut through specified start and end points.
        :param start: starting point in r.l.u., vector.
        :param end: ending point in r.l.u., vector.
        :param subset: a list of indices to cut. Omit to cut all available data.
        :param precision: refer to make_label method.
        :param labels: refer to make_label method.
        :param monitor: if normalize by monitor count.
        :return: ECut object.
        """
        start_p = self.ub_matrix.convert(start, 'rp')[0:2]
        end_p = self.ub_matrix.convert(end, 'rp')[0:2]
        seg = np.vstack([start_p, end_p])
        if subset is None:
            subset = self.data.index
        cut_results = []
        point_indices = []
        list_bin_polygons = []
        for index in subset:
            df = self.data.loc[index, 'points']
            voro = self.data.loc[index, 'voro']
            included = plotting.segment_intersect_polygons(seg, voro)
            bin_polygons = [v for v, include in zip(voro, included) if include]
            list_bin_polygons.append(bin_polygons)
            df_filtered = df.loc[included]
            point_indices.append(df_filtered.index)
            points = df_filtered[['px', 'py']]
            if monitor:
                intensities = df_filtered['permon']
            else:
                intensities = df_filtered['counts_norm']
            yerr = intensities / np.sqrt(df_filtered['counts'])
            percentiles = plotting.projection_on_segment(np.asarray(points), seg, self.ub_matrix.figure_aspect)
            result = pd.DataFrame({'x': percentiles, 'y': intensities, 'yerr': yerr}).sort_values(by='x')
            cut_results.append(result)
        cut_object = ConstECut(cut_results, point_indices, list_bin_polygons, self, subset, start, end)
        if plot:
            cut_object.plot(precision=precision, labels=labels)
        return cut_object

    def cut_bins(self, start, end, subset=None, xtol=None, ytol=None, no_points=None, precision=2, labels=None,
                 plot=True):
        """
        Generate 1D-cuts with rectangular bins.
        :param start: starting point in r.l.u., vector.
        :param end: ending point in r.l.u., vector.
        :param subset: a list of indices to cut. Omit to cut all available data.
        :param xtol: Bin size along cutting axis, in absolute reciprocal length.
        :param ytol: Lateral half bin size in [h, k, l] or absolute reciprocal length.
        :param no_points: Number of bins along cutting axis.
        :param precision: refer to make_label method.
        :param labels: refer to make_label method.
        :param plot: Automatically spawns a plot if true.
        :return: ConstECut object.
        """
        # Find absolute tolerances
        if xtol is not None and ytol is not None:
            raise ValueError('Only either of ytol or np should be supplied.')
        start_s = self.ub_matrix.convert(start, 'rs')
        end_s = self.ub_matrix.convert(end, 'rs')
        length = np.linalg.norm(end_s - start_s)
        if no_points is None and xtol is None:
            xtol = length / 10
        if no_points is not None:
            xtol = length / (no_points - 1)
        else:
            try:
                xtol = float(xtol)
            except TypeError:
                xtol = np.linalg.norm(self.ub_matrix.convert(xtol, 'rs'))

        if ytol is not None:
            try:
                ytol = float(ytol)
            except TypeError:
                ytol = np.linalg.norm(self.ub_matrix.convert(ytol, 'rs'))
        else:
            ytol = xtol
        # End finding tolerances
        if subset is None:
            subset = self.data.index
        cut_results = []
        point_indices = []
        list_bin_polygons = []
        for index in subset:
            frame = self.data.loc[index, 'points']
            points_s = self.ub_matrix.convert(np.asarray(frame.loc[:, ['px', 'py', 'pz']]), 'ps', axis=0)
            pd_cut = _binning_1d_cut(start_s, end_s, points_s, xtol, ytol)
            group = frame.groupby(pd_cut)
            counts_norm = group.counts_norm.sum().dropna()
            counts = group.counts.sum().dropna()
            yerr_scale = 1 / np.sqrt(counts)
            monitor = group['MON'].sum().dropna()
            counts_permon = counts_norm / monitor
            yerr = counts_permon * yerr_scale
            coords_p = group['px', 'py', 'pz'].mean().dropna()
            coords_s = self.ub_matrix.convert(coords_p, 'ps', axis=0)
            projections = plotting.projection_on_segment(coords_s, np.vstack((start_s, end_s)))
            cut_result = pd.DataFrame({'x': projections, 'y': counts_permon, 'yerr': yerr})\
                .dropna().reset_index(drop=True)
            indices = pd_cut[0].dropna().index.intersection(pd_cut[1].dropna().index)
            cut_results.append(cut_result)
            point_indices.append(indices)
            bin_polygons_s = _rectangular_bin_bounds(start_s, end_s, xtol, ytol)
            bin_polygons = [self.ub_matrix.convert(bins, sys='sp', axis=0)[:, 0:2] for bins in bin_polygons_s]
            list_bin_polygons.append(bin_polygons)
        cut_object = ConstECut(cut_results, point_indices, list_bin_polygons, self, subset, start, end)
        if plot:
            cut_object.plot(precision=precision, labels=labels)
        return cut_object

    def dispersion(self, start, end, no_points=21):
        energies = self.data.en
        en_cuts = bin_and_cut(energies, tolerance=0.05)
        multiindex = self.data.groupby(en_cuts)['ef'].nsmallest(1).index
        try:
            indices = [ind[1] for ind in multiindex]
        except TypeError:
            indices = multiindex
        c = self.cut_bins(start=start, end=end, subset=indices, no_points=no_points, plot=False)
        return c.vstack()

    def plot(self, subset=None, cols=None, aspect=None, plot_type=None, controls=True):
        # type: (..., list, int, float, str, bool) -> 'Plot2D'
        """
        Generate const-E colormaps.
        :param subset: list, indices of entries to be plotted. Omit to plot all.
        :param cols: How many columns should resulting plot have.
        :param aspect: y-x aspect ratio of generated plot. Larger value means unit length of y axis is greater. Omit to
        scale to equal length in absolute reciprocal length.
        :param plot_type: String. 'v': Voronoi, 'm': Mesh, 's': Scatter, 'd' Delaunay interpolation. Will be cycles for
        all plots. e.g. 'vm' will alternate between Voronoi patches and mesh.
        :param controls: True to show controls on screen.
        :return: Plot2D object.
        """
        plot_object = Plot2D(data_object=self, subset=subset, cols=cols, aspect=aspect, style=plot_type,
                             controls=controls)
        return plot_object

    def make_label(self, index, multiline=False, precision=2, columns=None):
        # type: (...) -> str
        """
        Makes legend entries for plots.
        :param multiline: If a newline is inserted between each property.
        :param index: Index of record to operate on.
        :param precision: precision of values in labels.
        :param columns: which properties to present in legend. None for all.
        :return: String representing an legend entry.
        """
        if columns is None:
            columns = ['en', 'ef', 'tt', 'mag']
        else:
            for nth, item in enumerate(columns):
                if item not in self.data.columns:
                    columns.pop(nth)

        elements = ['%s=%.*f' % (elem, precision, self.data.loc[index, elem]) for elem in columns]
        if multiline:
            join_char = '\n'
        else:
            join_char = ', '
        return join_char.join(elements)

    def to_csv(self):
        subdir_name = '+'.join(self.scan_files()) + '_out'
        full_dir_name = os.path.join(self.save_folder, subdir_name)
        try:  # not using exist_ok for python 2 compatibility
            os.makedirs(full_dir_name)
        except OSError:
            pass
        summary = str(self)
        f_summary = open(os.path.join(full_dir_name, 'summary.txt'), 'w')
        f_summary.write(summary)
        f_summary.close()
        for index in self.data.index:
            index_dir = os.path.join(full_dir_name, str(index))
            try:
                os.makedirs(index_dir)
            except OSError:
                pass
            self.data.loc[index, 'points'].to_csv(os.path.join(index_dir, 'points.csv'))
            for nth, patch in enumerate(self.data.loc[index, 'locus_a']):
                file_name = 'actual_locus_%d.csv' % nth
                full_name = os.path.join(index_dir, file_name)
                np.savetxt(full_name, patch, delimiter=',', header='px, py', comments='')
            for nth, patch in enumerate(self.data.loc[index, 'locus_p']):
                file_name = 'planned_locus_%d.csv' % nth
                full_name = os.path.join(index_dir, file_name)
                np.savetxt(full_name, patch, delimiter=',', header='px, py', comments='')

    def draw_voronoi_patch(self, ax, index, mesh=False, set_aspect=True):
        """
        Draw Voronoi tessellation patch on given ax.
        :param ax: matplotlib axes object.
        :param index: index of data entry to plot.
        :param mesh: True to only plot Voronoi tessellation mesh and no color.
        :param set_aspect: Set aspect to equal in absolute reciprocal length.
        :return: matplotlib collection.
        """
        record = self.data.loc[index, :]
        patch = draw_voronoi_patch(ax, record, mesh)
        if set_aspect:
            ax.set_aspect(self.ub_matrix.figure_aspect)
        self.set_axes_labels(ax)
        return patch

    def draw_interpolated_patch(self, ax, index, method='nearest', set_aspect=True):
        """
        Draw interpolated patch.
        :param ax: matplotlib axes object.
        :param index: index of data entry to plot.
        :param method: Interpolation method.
        :param set_aspect: Set aspect to equal in absolute reciprocal length.
        :return: matplotlib collection.
        """
        record = self.data.loc[index, :]
        patch = draw_interpolated_patch(ax, record, method=method)
        if set_aspect:
            ax.set_aspect(self.ub_matrix.figure_aspect)
        return patch

    def draw_scatter(self, ax, index, color=True, set_aspect=True):
        """
        Draw round scatter points.
        :param ax: matplotlib axes object.
        :param index: index of data entry to plot.
        :param color: If use colormap to show data.
        :param set_aspect: Set aspect to equal in absolute reciprocal length.
        :return: matplotlib collection.
        """
        record = self.data.loc[index, :]
        s = draw_scatter(ax, record, color=color)
        if set_aspect:
            ax.set_aspect(self.ub_matrix.figure_aspect)
        return s

    def set_axes_labels(self, ax):
        """
        Set axes labels (like [H 0 0]) to ax object.
        :param ax: Which ax to set to.
        :return: None
        """
        xlabel, ylabel = ub.guess_axes_labels(self.ub_matrix.plot_x, self.ub_matrix.plot_y_nominal)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @property
    def save_folder(self):
        """
        Gives the folder of first data file. A good place to export CSV files.
        :return: path
        """
        return os.path.dirname(self._file_names[0])

    def scan_files(self, full=False):
        """
        Which scan files are included in this dataset.
        :param full: Returns full path.
        :return: list of strings.
        """
        if full:
            return self._file_names
        else:
            return [os.path.split(name)[1] for name in self._file_names]

    def summary(self):
        """
        Print a summary of contained data to STDOUT.
        :return: None
        """
        print(self)

    def dump(self, file_name=None):
        """
        Pickle and dump self.
        :param file_name: File name to dump to.
        :return: None
        """
        if file_name is None:
            file = filedialog.asksaveasfile(initialdir=self.save_folder, defaultextension='.dmp', mode='wb',
                                            filetypes=(('multiflexxlib dump', '.dmp'),))
        else:
            file = open(file_name, 'wb')
        pickle.dump(self, file)

    def __copy__(self):
        return BinnedData(self.data, self._file_names, self.ub_matrix)

    def copy(self):
        return self.__copy__()

    def __add__(self, other):
        # type: (BinnedData) -> BinnedData
        file_names = set(self.file_names() + other.file_names())
        if self.ub_matrix != other.ub_matrix:
            raise ValueError('Cannot merge BinnedData objects with different ub-matrix')
        else:
            ub_matrix = self.ub_matrix
        data = pd.concat([self.data, other.data], axis=0, ignore_index=True)
        return BinnedData(data, file_names, ub_matrix)


class ConstECut(object):
    def __init__(self, cuts, point_indices, list_bin_polygons, data_object, data_indices, start, end):
        """
        Const-E cut object, should not be instantiated on its own.
        :param cuts:
        :param point_indices:
        :param list_bin_polygons:
        :param data_object:
        :param data_indices:
        :param start:
        :param end:
        """
        self.cuts = cuts
        self.data_object = data_object
        self.data_indices = data_indices
        self.point_indices = point_indices
        self.list_bin_polygons = list_bin_polygons
        self.figure, self.ax = None, None
        self.artists = None
        self.legend = None
        self.start_r = np.asarray(start)
        self.end_r = np.asarray(end)

    def to_csv(self):
        """
        Export to CSV file. Only supports cuts with only one set of data to avoid confusion.
        :return: None
        """
        if len(self.cuts) > 1:
            # Solely to shift responsibility of managing files to user.
            raise NotImplementedError('Saving to CSV only supported for cuts only containing 1 set of data.')
        file = filedialog.asksaveasfile(initialdir=self.data_object.save_folder, defaultextension='.csv',
                                        filetypes=(('comma-separated values', '.csv'), ))
        if file is None:
            return
        self.cuts[0].to_csv(file)

    def plot(self, precision=2, labels=None):
        """
        Plot cut results.
        :param precision: Precision used in labels.
        :param labels: Which labels to include.
        :return: None. Access figure and axes objects from self.figure, self.ax respectively.
        """
        self.figure, self.ax = plt.subplots()
        self.artists = []
        ax = self.ax
        for i, cut in enumerate(self.cuts):
            label = self.data_object.make_label(self.data_indices[i], precision=precision, columns=labels)
            artist = ax.errorbar(cut.x, cut.y, yerr=cut.yerr, fmt='o', label=label)
            self.artists.append(artist)
        self.legend = ax.legend()
        self.set_axes_labels(ax)
        self.put_parasite_axis(ax)
        self.figure.tight_layout()

    def inspect(self, shade=True):
        """
        Generate a graph showing which data points are included in the cuts.
        :return: None
        """
        f, axes = plt.subplots(nrows=2, ncols=len(self.cuts), sharex='row', sharey='row')
        axes = axes.reshape(2, -1)
        for i, cut in enumerate(self.cuts):
            ax_top = axes[0, i]
            ax_top.set_aspect(self.data_object.ub_matrix.figure_aspect)
            self.data_object.set_axes_labels(ax_top)
            ax_bottom = axes[1, i]
            locus_p = self.data_object.data.loc[self.data_indices[i], 'locus_p']
            points = self.data_object.data.loc[self.data_indices[i], 'points']
            indices = self.point_indices[i]
            draw_locus_outline(ax_top, locus_p)
            bins_collection = plotting.draw_patches(self.list_bin_polygons[i], mesh=True)
            ax_top.add_collection(bins_collection)
            self.put_parasite_axis(ax_bottom)
            # self.data_object.draw_voronoi_patch(ax_top, index=self.data_indices[i])
            if shade:
                ax_top.scatter(x=points.px[indices], y=points.py[indices], c=points.permon[indices], zorder=10, s=12)
            ax_top.scatter(x=points.px, y=points.py, c=[0.8, 0.8, 0.8], zorder=0, s=6)
            plotting.draw_line(ax_top, [self.start_r, self.end_r], self.data_object.ub_matrix)
            self.set_axes_labels(ax_bottom)
            ax_bottom.errorbar(cut.x, cut.y, yerr=cut.yerr, fmt='o')
        f.tight_layout()
        return f, axes

    def set_axes_labels(self, ax):
        """
        Set axes labels
        :param ax: Which axes object to set on.
        :return: None
        """
        ax.set_ylabel('Intensity (a.u.)')
        start_xlabel = '[' + ','.join(['%.2f' % x for x in self.start_r]) + ']'
        end_xlabel = '[' + ','.join(['%.2f' % x for x in self.end_r]) + ']'
        ax.set_xlabel('Relative position\n%s to %s' % (start_xlabel, end_xlabel))

    def vstack(self):
        energies = [self.data_object.data.loc[index, 'en'] for index in self.data_indices]
        energy_bin_edges = _dispersion_bin_edges(energies)
        shapes = []
        values = []
        for i, cut in enumerate(self.cuts):
            q_bins = _dispersion_bin_edges(cut.x)
            e_bin = [energy_bin_edges[i], energy_bin_edges[i+1]]
            for j in range(len(cut)):
                shape = mpl_patches.Rectangle((q_bins[j], e_bin[0]), q_bins[j+1] - q_bins[j], e_bin[1] - e_bin[0])
                shapes.append(shape)
                values.append(cut.y[j])
        shape_col = PatchCollection(shapes)
        shape_col.set_array(np.asarray(values))

        f, ax = plt.subplots(1)
        cut_path = mpl_patches.Rectangle((0, energies[0]), 1, energies[-1] - energies[0], facecolor='None')
        ax.add_patch(cut_path)
        shape_col.set_clip_path(cut_path)
        # patch_poly.set_clip_box([[0, energies[0]], [1, energies[-1]]])
        ax.add_collection(shape_col)
        ax.set_xlim(0, 1)
        ax.set_ylim(energies[0], energies[-1])
        self.set_axes_labels(ax)
        ax.set_ylabel('dE (meV)')
        self.put_parasite_axis(ax)
        return f, ax

    def __len__(self):
        return len(self.data_indices)

    def percentile_to_hkl(self, percentile):
        return self.start_r + (self.end_r - self.start_r) * percentile

    def _changing_indices(self):
        delta = self.end_r - self.start_r
        result = [n for n in range(3) if not np.isclose(delta[n], 0)]
        return tuple(result)

    def put_parasite_axis(self, ax):
        ax_p = ax.twiny()
        fca = self._changing_indices()[0]
        ax_p.set_xlim(self.start_r[fca], self.end_r[fca])
        ax_p.set_xlabel(list('HKL')[fca])


class Plot2D(object):
    """
    2D const-E plots. Should not be instantiated by invoking its constructor.
    """
    def __init__(self, data_object, subset=None, cols=None, aspect=None, style=None, controls=False):
        if subset is None:
            subset = data_object.data.index
        if style is None:
            style = 'v'
        self.data_object = data_object
        ub_matrix = self.data_object.ub_matrix
        rows, cols = _calc_figure_dimension(len(subset), cols)
        self.f, axes = _init_2dplot_figure(rows, cols, ub_matrix)
        self.axes = axes.reshape(-1)
        self._set_hkl_formatter()
        self.patches = None
        self.indices = subset
        self.aspect = aspect
        self.cbar = None
        self.f.data_object = self
        self.__plot__(style)
        if controls:
            self.controls = dict()
            self.add_controls()

    def _set_hkl_formatter(self):
        def format_coord(x, y):
            hkl = self.data_object.ub_matrix.convert([x, y, 0], 'pr')
            length = np.linalg.norm(self.data_object.ub_matrix.convert(hkl, 'rs'))
            return 'h={:.2f}, k={:.2f}, l={:.2f}, qm={:.2f}'.format(hkl[0], hkl[1], hkl[2], length)
        for ax in self.axes:
            ax.format_coord = format_coord

    def __plot__(self, style):
        self.patches = []
        if self.aspect is None:
            aspect = self.data_object.ub_matrix.figure_aspect
        else:
            aspect = self.aspect
        for nth, index in enumerate(self.indices):
            ax = self.axes[nth]
            ax.grid(linestyle='--', zorder=0)
            ax.set_axisbelow(True)
            record = self.data_object.data.loc[index, :]
            method_char = style[nth % len(style)]
            if method_char == 'v':
                self.patches.append(draw_voronoi_patch(ax, record))
            elif method_char == 'd':
                self.patches.append(draw_interpolated_patch(ax, record,
                                                            aspect=self.data_object.ub_matrix.figure_aspect))
            elif method_char == 'm':
                self.patches.append(draw_voronoi_patch(ax, record, mesh=True))
            elif method_char == 's':
                self.patches.append(draw_scatter(ax, record))
            else:
                self.patches.append(draw_voronoi_patch(ax, record))  # default to plotting voronoi patch

            draw_locus_outline(ax, record.locus_p)
            ax.set_aspect(aspect)
            self.data_object.set_axes_labels(ax)
            ax.set_xlim([record.points.px.min(), record.points.px.max()])
            ax.set_ylim([record.points.py.min(), record.points.py.max()])
            legend_str = self.data_object.make_label(index, multiline=True)
            self._write_label(ax, legend_str)
        self.f.tight_layout()
        plt.show(block=False)

    def to_eps(self):
        pass

    def cut(self, start, end, subset=None, precision=2, labels=None, monitor=True):
        """
        1D-cut through specified start and end points.
        :param start: starting point in r.l.u., vector.
        :param end: ending point in r.l.u., vector.
        :param subset: a list of indices to cut. Omit to cut all available data.
        :param precision: refer to make_label method.
        :param labels: refer to make_label method.
        :param monitor: if normalize by monitor count.
        :return: ECut object.
        """
        if subset is None:
            subset = self.indices
        else:
            subset = [self.indices[x] for x in subset]
        cut_obj = self.data_object.cut_voronoi(start, end, subset, precision, labels, monitor)
        return cut_obj

    def update_label(self, index, labels, precision=2):
        """
        Update label on figure.
        :param index: which ax to update
        :param labels: which labels to include, list.
        :param precision: Decimal position of generated labels.
        :return:
        """
        ax = self.axes[index]
        label_text = self.data_object.make_label(self.indices[index], multiline=True, columns=labels,
                                                 precision=precision)
        label = ax.findobj(match=lambda o: o.get_gid() == 'label')
        label[0].set_text(label_text)

    @staticmethod
    def _write_label(ax, text):
        ax.text(1.00, 1.00,
                text,
                transform=ax.transAxes, zorder=200, color='black',
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}, horizontalalignment='right',
                verticalalignment='top', gid='label')

    def set_norm(self, norm):
        """
        Set normalization to plots.
        :param norm: matplotlib normalization object.
        :return: None
        """
        for patch in self.patches:
            patch.set_norm(norm)

    def add_colorbar(self):
        """Add colorbar to plot. For production."""
        f = self.f
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
        cbar = f.colorbar(self.patches[0], cax=cbar_ax)
        cbar.set_label('Normalized intensity (a.u.)')
        self.cbar = cbar

    def set_lognorm(self, vmin=0.01, vmax=1, subset=None):
        # TODO: Add subset parameter
        """
        Sets Log10 colormap normalization to plot.
        :param vmin: min value. needs to be larger than 0.
        :param vmax: max value.
        :return: None
        """
        self.set_norm(LogNorm(vmin=vmin, vmax=vmax))

    def set_linear(self, vmin=0, vmax=1):
        """
        Sets linear normalization to plot.
        :param vmin: min value.
        :param vmax: max value.
        :return:
        """
        self.set_norm(None)
        self.set_clim(vmin, vmax)

    def set_clim(self, vmin, vmax):
        """
        Sets limits to colormaps.
        :param vmin: vmin
        :param vmax: vmax
        :return:
        """
        for p in self.patches:
            p.set_clim((vmin, vmax))

    def set_plim(self, pmin=0, pmax=100):
        """
        Sets limits in percentiles, 100 = 100% pencentile
        :param pmin: min percentile
        :param pmax: max percentile
        :return: None
        """
        for i in range(len(self.indices)):
            index = self.indices[i]
            data = self.data_object.data.loc[index, 'points'].counts_norm
            data_max = np.max(data)
            pmin_value = np.percentile(data, pmin, interpolation='lower')
            pmax_value = np.percentile(data, pmax, interpolation='higher')
            vmin = pmin_value / data_max
            vmax = pmax_value / data_max

            self.patches[i].set_clim((vmin, vmax))

    def add_controls(self):
        """
        Adds control buttons to figure.
        :return: None
        """
        self.f.subplots_adjust(bottom=0.2)
        row_pos = 0
        # LogNorm
        ax = self.f.add_axes([row_pos * 0.1 + 0.1, 0.05, 0.09, 0.05])
        button = Button(ax, 'Log')
        self.controls['log'] = button
        button.on_clicked(lambda event: self.set_lognorm())
        # Linear
        row_pos += 1
        ax = self.f.add_axes([row_pos * 0.1 + 0.1, 0.05, 0.09, 0.05])
        button = Button(ax, 'Linear')
        self.controls['linear'] = button
        button.on_clicked(lambda event: self.set_linear())


def draw_voronoi_patch(ax, record, mesh=False, zorder=10):
    """
    Puts down Voronoi representation on axes object
    :param ax: Matplotlib axes object.
    :param record: A row in BinnedData.data
    :param mesh: Whether only draw mesh.
    :param zorder: zorder to be used for artist.
    :return: PathCollection
    """
    # TODO: Check zorder operation
    values = record.points.permon / record.points.permon.max() + 1e-10  # to avoid drawing zero counts as empty
    v_fill = plotting.draw_patches(record.voro, values, mesh=mesh, zorder=zorder)
    coverage_patch = _draw_coverage_mask(ax, record.locus_a)
    ax.add_collection(v_fill)
    v_fill.set_clip_path(coverage_patch)
    return v_fill


def draw_interpolated_patch(ax, record, aspect=1, method='linear', zorder=10):
    # to avoid drawing zero counts as empty
    values = np.asarray(record.points.permon / record.points.permon.max() + 1e-10)
    px = record.points.px
    py = record.points.py
    ay = record.points.py * aspect
    x_min, x_max = px.min() - 0.01, px.max() + 0.01
    y_min, y_max = ay.min() - 0.01, ay.max() + 0.01
    py_min, py_max = py.min() - 0.01, py.max() + 0.01
    x_grid, y_grid = np.mgrid[x_min:x_max:0.002, y_min:y_max:0.002]
    signal_grid = interpolate.griddata(np.vstack([px, ay]).T, values, (x_grid, y_grid), method=method, fill_value=0)
    mesh = ax.imshow(flipud(signal_grid.T), cmap='inferno',
                     extent=[x_min, x_max, py_min, py_max],
                     zorder=zorder, interpolation='nearest')
    coverage_patch = _draw_coverage_mask(ax, record.locus_a)
    mesh.set_clip_path(coverage_patch)
    return mesh


def draw_scatter(ax, record, color=True, colormap='inferno', size=12, zorder=10):
    if color:
        values = np.asarray(record.points.permon / record.points.permon.max() + 1e-10)
    else:
        values = 0
    px = record.points.px
    py = record.points.py
    s = ax.scatter(x=px, y=py, c=values, cmap=colormap, s=size, zorder=zorder)
    return s


def draw_locus_outline(ax, list_of_locuses):
    for locus in list_of_locuses:
        locus = np.asarray(locus)
        ax.plot(locus[:, 0], locus[:, 1], lw=0.05)


def _calc_figure_dimension(no_plots, cols=None):
    if cols is None:
        if no_plots == 1:
            return 1, 1
        elif no_plots == 2:
            return 1, 2
        elif no_plots == 3:
            return 1, 3
        else:
            sqroot = np.sqrt(no_plots)
            if sqroot == int(sqroot):
                return int(sqroot), int(sqroot)
            else:
                cols = int(sqroot) + 1
                if cols * (cols - 1) < no_plots:
                    rows = cols
                else:
                    rows = cols - 1
                return int(rows), int(cols)
    else:
        if no_plots % cols == 0:
            rows = no_plots / cols
        else:
            rows = no_plots / cols + 1
        return int(rows), int(cols)


def _init_2dplot_figure(rows, cols, ub_matrix):
    # type: (int, int, UBMatrix) -> ...
    if ub_matrix.is_orthogonal:
        f, axes = plt.subplots(rows, cols, sharex='all', sharey='all')
        return f, np.array(axes)
    else:
        f = plt.figure()
        grid_helper = _create_grid_helper(ub_matrix.shear_coeff)
        axes = []
        for i in range(1, rows * cols + 1):
            if i == 1:
                ax = Subplot(f, rows, cols, i, grid_helper=grid_helper)
            else:
                ax = Subplot(f, rows, cols, i, grid_helper=grid_helper, sharex=axes[0], sharey=axes[0])
            ax.xaxis.set_label_coords(0.5, -0.1)
            ax.yaxis.set_label_coords(0.5, -0.1)
            f.add_subplot(ax)
            axes.append(ax)
        return f, np.array(axes)


def _create_grid_helper(shear_coeff):
    def tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return x + y * shear_coeff, y

    def inv_tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return x - y * shear_coeff, y

    return GridHelperCurveLinear((tr, inv_tr))


def _dispersion_bin_edges(sequence):
    sequence = list(sequence)
    bin_edges = [sequence[0] - (sequence[1] - sequence[0]) / 2]
    for i in range(len(sequence) - 1):
        bin_edges.append((sequence[i] + sequence[i + 1]) / 2)
    bin_edges.append(sequence[-1] + (sequence[-1] - sequence[-2]) / 2)
    return bin_edges


def _draw_coverage_mask(ax_handle, locus):
    mpath_path = mpl_path.Path
    combined_verts = np.zeros([0, 2])
    combined_codes = []
    for each in locus:
        codes = [mpath_path.LINETO] * len(each)
        codes[0], codes[-1] = mpath_path.MOVETO, mpath_path.CLOSEPOLY
        combined_codes += codes
        combined_verts = np.vstack([combined_verts, each])
    path = mpath_path(combined_verts, combined_codes)
    patch = mpl_patches.PathPatch(path, facecolor='k', alpha=0, zorder=10)
    ax_handle.add_patch(patch)
    return patch


def ask_directory(title='Choose a folder'):
    """
    Show a dialog asking for a folder.
    :param title:
    :return: Path
    """
    root = tkinter.Tk()
    root.withdraw()
    path = filedialog.askdirectory(initialdir='.', title=title)
    root.destroy()
    return path


def list_flexx_files(path):
    """
    Lists all TASMAD scan files under a directory.
    :param path: Source path.
    :return: A list of file names.
    """
    if path == '':
        raise ValueError('Path \'%s\' is invalid.' % path)
    file_names = [os.path.join(path, s) for s in os.listdir(path) if (s.isdigit() and len(s) == 6)]
    file_names.sort(key=lambda fn: int(os.path.split(fn)[1]))
    return file_names


def _unpack_user_hkl(user_input):
    # type: (str) -> list
    unpacked = [float(s) for s in user_input.split(',')]
    if len(unpacked) != 3:
        raise ValueError('Not a valid h, k, l input.')

    return unpacked


def _rectangular_bin_bounds(start_s, end_s, xtol, ytol):
    start_s, end_s = np.asarray(start_s), np.asarray(end_s)
    delta = end_s - start_s
    norm_delta = np.linalg.norm(delta)
    no_bins = int((norm_delta + xtol / 2) // xtol + 1)
    delta_x_unit = delta / np.linalg.norm(delta)
    delta_y_unit = ub.rotate_around_z(delta_x_unit, np.pi / 2)
    dx = delta_x_unit * xtol / 2
    dy = delta_y_unit * ytol
    first_bin = np.vstack([start_s - dx - dy, start_s - dx + dy, start_s + dx + dy, start_s + dx - dy])
    bins = []
    for i in range(no_bins):
        bins.append(first_bin + dx * 2 * i)
    return bins


def _binning_1d_cut(start, end, points, tol_transverse=None, tol_lateral=None):
    # type: (np.ndarray, np.ndarray, int, ...) -> (Categorical, Categorical)
    """
    Bins given points by a series of provided rectangular bins.
    :param start: starting point.
    :param end: end point.
    :param points: row vector array or pandas DataFrame of point
    :param tol_lateral: lateral tolerance, in absolute units of points.
    :param tol_transverse: transversal tolerance.
    :return: A tuple of two pd.cut for subsequent groupby
    """
    points = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    segment_length = np.linalg.norm(end - start)
    if delta_x == 0 and delta_y == 0:
        raise ValueError('zero length segment provided to 1D-cut')
    else:
        if delta_y < 0:
            angle = np.arccos(delta_x / segment_length)
        else:
            angle = -np.arccos(delta_x / segment_length)

    start_rot = ub.rotate_around_z(start, angle)
    end_rot = ub.rotate_around_z(end, angle)
    scaling_factor = float((end_rot - start_rot)[0])
    points_rot = pd.DataFrame(ub.rotate_around_z(np.asarray(points.loc[:, ['x', 'y', 'z']]), angles=angle, axis=0),
                              index=points.index, columns=['x', 'y', 'z'])
    points_scaled = (points_rot - start_rot) / scaling_factor
    xtol = tol_transverse / scaling_factor
    ytol = tol_lateral / scaling_factor
    ycut = pd.cut(points_scaled.y, [-ytol, ytol])
    xbins = [-xtol/2]
    while True:
        next_bin = xbins[-1] + xtol
        xbins.append(next_bin)
        if next_bin > 1:
            break
    xcut = pd.cut(points_scaled.x, xbins)
    return xcut, ycut


def calculate_locus(ki, kf, a3_start, a3_end, a4_start, a4_end, ub_matrix, expand_a3=False):
    """
    Calculate Q-space coverage of a const-E scan.
    :param ki: ki.
    :param kf: kf.
    :param a3_start: A3 angle start.
    :param a3_end: A3 angle end. Could be bigger or smaller than start.
    :param a4_start: A4 start
    :param a4_end: A4 end
    :param ub_matrix: UBMatrix object.
    :param expand_a3: expand A3 by a minuscule amount to avoid numerical precision problems.
    :return: N*2 array of point coordinates on plot in p system.
    """
    if a4_start > 0:
        a4_span = (NUM_CHANNELS - 1) * CHANNEL_SEPARATION
    else:
        a4_span = (NUM_CHANNELS - 1) * CHANNEL_SEPARATION * (-1)
    if a3_start > a3_end:
        a3_start, a3_end = (a3_end, a3_start)
    if expand_a3:
        a3_end = a3_end + 0.05
        a3_start - a3_start - 0.05
    a3_range = np.linspace(a3_start, a3_end, max(abs(int(a3_end - a3_start)), 2))
    a4_range_low = np.linspace(a4_start - a4_span / 2, a4_end - a4_span / 2, max(abs(int(a3_end - a3_start)), 2))
    a4_range_high = np.linspace(a4_end + a4_span / 2, a4_start + a4_span / 2, max(abs(int(a3_end - a3_start)), 2))
    a4_span_range_low = np.linspace(a4_start + a4_span / 2, a4_start - a4_span / 2, NUM_CHANNELS)
    a4_span_range_high = np.linspace(a4_end - a4_span / 2, a4_end + a4_span / 2, NUM_CHANNELS)

    a3_list = np.hstack((a3_range, a3_range[-1] * np.ones(len(a4_span_range_high)),
                         a3_range[::-1], a3_range[0] * np.ones(len(a4_span_range_low))))
    a4_list = np.hstack((a4_range_low, a4_span_range_high, a4_range_high, a4_span_range_low))
    s_locus = angle_to_qs(ki, kf, a3_list, a4_list)
    p_locus = ub_matrix.convert(s_locus, 'sp')

    return np.ndarray.tolist(p_locus[0:2, :].T)


def load(file_name=None):
    """
    Restores dumped binary pickle to a name.
    :param file_name: Which file to load.
    :return: BinnedData object.
    """
    # type: str -> BinnedData
    if file_name is None:
        file = filedialog.askopenfile(defaultextension='.dmp', mode='rb',
                                      filetypes=(('multiflexxlib dump', '.dmp'),))
    else:
        file = open(file_name, 'rb')

    if file is None:
        raise IOError('Error accessing dump file %s' % file_name)
    else:
        return pickle.load(file)
