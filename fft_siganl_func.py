"""
-------------------------------------------------------------------
fft_folder_func.py

contents)
Define Folder and csv related functions to be used in fft_analyze_main.
-------------------------------------------------------------------^
"""
# python library
import os
import time

# numpy library
import numpy as np

# pandas library
import pandas as pd

# scipy library
from scipy import signal  # For EDF Filtering
from scipy import fftpack

from functools import wraps

from multiprocessing import Process

import fft_folder_func as fft_folder


def stop_watch(func):
	"""
	measure processing time
	
	:param func:
	:return:
	"""
	
	@wraps(func)
	def wrapper(*args, **kargs):
		start = time.time()
		result = func(*args, **kargs)
		elapsed_time = time.time() - start
		print(f"{func.__name__} took {elapsed_time} sec!")
		return result
	
	return wrapper


@stop_watch
def signal_value(edf, ch):
	"""
	Extract necessary data from signal information.

	:param edf:  edf object
	:param ch:  channel
	:return:  x_array(sec), y_array(uv)
	"""
	try:
		y_array = edf.readSignal(ch, 0)
		ch_frq = edf.getSampleFrequencies()
		x_array = np.arange(edf.getFileDuration() * ch_frq[ch])
		x_array = x_array / ch_frq[ch]
		
		signal_array = np.c_[x_array, y_array]
		return signal_array
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass


@stop_watch
def output_of_basic_statistics(signal_array):
	"""
	Output of basic statistics

	:param signal_array:
	:return: statistics_array
	"""
	try:
		statistics_array = np.array([])
		
		signal_array = signal_array[~np.isnan(signal_array)]
		
		# Basic statistical data of ch5_signal
		ave_statistics = np.mean(signal_array)
		min_statistics = np.min(signal_array)
		max_statistics = np.max(signal_array)
		std_statistics = np.std(signal_array)
		median_statistics = np.median(signal_array)
		
		# Add basic statistics to statistics_array
		statistics_array = np.append(statistics_array, ave_statistics)
		statistics_array = np.append(statistics_array, min_statistics)
		statistics_array = np.append(statistics_array, max_statistics)
		statistics_array = np.append(statistics_array, std_statistics)
		statistics_array = np.append(statistics_array, median_statistics)
		
		return statistics_array
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass


def print_signal(title, data_array):
	"""
	print signal information
	:param title: subject
	:param data_array: signal array
	:return:
	"""
	
	print(f"{title}_signal_ave_include_noise = {data_array[0]}")
	print(f"{title}_signal_min_include_noise = {data_array[1]}")
	print(f"{title}_signal_max_include_noise = {data_array[2]}")
	print(f"{title}_signal_std_include_noise = {data_array[3]}")
	print(f"{title}_signal_median_include_noise = {data_array[4]}\n")


def divide_dataframe_epoc(signal_name, numpy_array, header_str):
	"""
	Decompose the dataframe for each EPOC.
	
	:param signal_name:
	:param numpy_array:
	:param header_str:
	:return:
	"""
	
	# Flag for creating dataframe
	first_flag = True
	
	# epoc counter
	epoc_num = 0
	
	for epoc_data in numpy_array:
		if first_flag:
			# create column
			column_t = f"time_epc_{str(epoc_num)}"
			column_signal = f"{signal_name}_epoc_{str(epoc_num)}"
			column_list = [column_t, column_signal]
			
			df_data_frame = pd.DataFrame(epoc_data, index=[header_str], columns=column_list)
			
			first_flag = False
		else:
			# create column
			column_t = f"time_epc_{str(epoc_num)}"
			column_signal = f"{signal_name}_epoc_{str(epoc_num)}"
			
			df_data_frame[column_t] = epoc_data[:, 0]
			df_data_frame[column_signal] = epoc_data[:, 1]
		
		epoc_num = epoc_num + 1
	
	return df_data_frame


@stop_watch
def band_pass(x, sampling_rate, fp, fs, gpass, gstop):
	"""
	band-pass filter

	:param x: data
	:param sampling_rate: sampling rate[Hz]
	:param fp: pass frequency[Hz]
	:param fs: stop frequency[Hz]
	:param gpass: The maximum loss in the passband (dB).
	:param gstop: The minimum attenuation in the stopband (dB).
	:return:
	"""
	try:
		# Nyquist frequency
		fn = sampling_rate / 2
		
		# Normalize the pass-band end frequency with the Nyquist frequency.
		wp = fp / fn
		
		# Normalize the stop-band end frequency with the Nyquist frequency.
		ws = fs / fn
		
		# Calculate the order and Butterworth's normalized frequency.
		n, wn = signal.buttord(wp, ws, gpass, gstop)
		
		# Calculate the numerator and denominator of the filter transfer function.
		b, a = signal.butter(n, wn, 'bandpass')
		
		# Apply a filter to the signal.
		y = signal.filtfilt(b, a, x)
		return y
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass


@stop_watch
def filter_result(data_frame, freq):
	"""
	Data is passed through the filter for processing.

	:param data_frame:  Data frame for analysis
	:param freq:  Data capture frequency
	:return: Data frame after processing
	"""
	try:
		for stage_i in range(0, data_frame.shape[1], 2):
			# Since there is one data in two columns, the step is specified as 2.
			data_frame[data_frame.columns[stage_i + 1]] = \
				band_pass(
					data_frame[data_frame.columns[stage_i + 1]],
					freq,
					np.array([0.3, 35]),
					np.array([0.15, 70]),
					3,
					5.5
				)
		
		return data_frame
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass


@stop_watch
def induction_result(data_frame_fp1, data_frame_fp2, data_frame_a1, data_frame_a2, index_header):
	"""
	The converted result is output by the induction function.
	
	:param data_frame_fp1: dataframe of fp1 signal
	:param data_frame_fp2: dataframe of fp2 signal
	:param data_frame_a1:  dataframe of a1 signal
	:param data_frame_a2:  dataframe of a2 signal
	:param index_header:
	:return:
	"""
	# calculate (a1 + a2) / 2
	data_frame_average = pd.DataFrame()
	
	# calculate fp1_Ave = fp1_filter - Average
	data_frame_fp1_ave = pd.DataFrame()
	
	# calculate fp2_Ave = fp2_filter - Average
	data_frame_fp2_ave = pd.DataFrame()
	
	# calculate fp1_fp2 = fp1_filter - fp2_filter
	data_frame_fp1_fp2 = pd.DataFrame()
	
	try:
		for stage_i0 in range(0, data_frame_a1.shape[1], 2):
			data_frame_average[data_frame_a1.columns[stage_i0]] = data_frame_a1[data_frame_a1.columns[stage_i0]]
			data_frame_average[data_frame_a1.columns[stage_i0 + 1]] = \
				(
						data_frame_a1[data_frame_a1.columns[stage_i0 + 1]] +
						data_frame_a2[data_frame_a2.columns[stage_i0 + 1]]
				) / 2
			
			if not stage_i0:
				data_frame_average.index = [index_header]
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	try:
		for stage_i1 in range(0, data_frame_fp1.shape[1], 2):
			# store time column.
			data_frame_fp1_ave[data_frame_fp1.columns[stage_i1]] = data_frame_fp1[data_frame_fp1.columns[stage_i1]]
			
			# get current column name
			tmp_columns = str(data_frame_fp1.columns[stage_i1 + 1])
			
			# Overwrite new column name
			tmp_columns = tmp_columns.replace('fp1', 'fp1_ave')
			
			data_frame_fp1_ave[tmp_columns] = \
				(
						data_frame_fp1[data_frame_fp1.columns[stage_i1 + 1]] -
						data_frame_average[data_frame_average.columns[stage_i1 + 1]]
				)
			
			if not stage_i1:
				data_frame_fp1_ave.index = [index_header]
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	try:
		for stage_i2 in range(0, data_frame_fp2.shape[1], 2):
			
			data_frame_fp2_ave[data_frame_fp2.columns[stage_i2]] = data_frame_fp2[data_frame_fp2.columns[stage_i2]]
			
			tmp_columns = str(data_frame_fp2.columns[stage_i2 + 1])
			
			tmp_columns = tmp_columns.replace('fp2', 'fp2_ave')
			
			data_frame_fp2_ave[tmp_columns] = \
				(
						data_frame_fp2[data_frame_fp2.columns[stage_i2 + 1]] -
						data_frame_average[data_frame_average.columns[stage_i2 + 1]]
				)
			
			if not stage_i2:
				data_frame_fp2_ave.index = [index_header]
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	try:
		for stage_i3 in range(0, data_frame_fp1.shape[1], 2):
			
			data_frame_fp1_fp2[data_frame_fp1.columns[stage_i3]] = data_frame_fp1[data_frame_fp1.columns[stage_i3]]
			
			tmp_columns = str(data_frame_fp1.columns[stage_i3 + 1])
			
			tmp_columns = tmp_columns.replace('fp1', 'fp1_fp2')
			
			data_frame_fp1_fp2[tmp_columns] = \
				(
						data_frame_fp1[data_frame_fp1.columns[stage_i3 + 1]] -
						data_frame_fp2[data_frame_fp2.columns[stage_i3 + 1]]
				)
			
			if not stage_i3:
				data_frame_fp1_fp2.index = [index_header]
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	return data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2


def data_correction(data_frame, parameter):
	"""
	
	:param data_frame:
	:param parameter:
	:return:
	"""
	try:
		# Numpy array for temporary stock
		np_array_time = np.array([])
		np_array_signal = np.array([])
		
		# Number of columns that do not contain NaN values
		nan_counter = 0
		
		for i_data in range(0, data_frame.shape[1], 2):
			if not data_frame[data_frame.columns[i_data]].isnull().any():
				tmp_column_time = data_frame.columns[i_data]
				np_array_time = \
					np.append(np_array_time, data_frame[tmp_column_time])
				
				nan_counter += 1
			
			if not data_frame[data_frame.columns[i_data + 1]].isnull().any():
				tmp_column_signal = data_frame.columns[i_data + 1]
				np_array_signal = \
					np.append(np_array_signal, data_frame[tmp_column_signal])
		
		df_array_signal = pd.DataFrame(np_array_signal)
		array_index = list(np_array_time)
		df_array_signal.index = array_index
		
		c_array = df_array_signal.quantile(q=[0, 0.25, 0.5, 0.75, 1.0])
		c_array_index = ['0%', '25%', '50%', '75%', '100%']
		c_array.index = c_array_index
		c_array.columns = ['meadian ratio']
		
		print(c_array)
		
		iqr = c_array.loc['75%'] - c_array.loc['25%']
		
		# lower bound
		lower_bd = float(c_array.loc['25%'] - iqr * parameter)
		upper_bd = float(c_array.loc['75%'] + iqr * parameter)
		
		first_correction_flag = True
		
		new_data_frame = pd.DataFrame()
		
		for i_data_frame in range(0, data_frame.shape[1], 2):
			tmp_column_min = data_frame.columns[i_data_frame + 1]
			tmp_column_max = data_frame.columns[i_data_frame + 1]
			
			# Extract the index of outliers.
			reqd_index = list(
				np.where(data_frame[tmp_column_min] < lower_bd) or
				np.where(data_frame[tmp_column_max] > upper_bd)
			)
			
			# Temporary storage of time data
			target_column = data_frame.columns[i_data_frame]
			tmp_time = data_frame[target_column]
			
			# Remove the outlier index in time.(cation: force to change series)
			tmp_time.iloc[reqd_index] = np.nan
			
			new_tmp_time = tmp_time
			
			new_data_frame[tmp_time.name] = new_tmp_time
			
			# Temporary storage of signal data
			target_column = data_frame.columns[i_data_frame + 1]
			tmp_signal = data_frame[target_column]
			
			# Remove the outlier index in signal.(cation: force to change series)
			tmp_signal.iloc[reqd_index] = np.nan
			
			new_tmp_signal = tmp_signal
			
			new_data_frame[tmp_signal.name] = new_tmp_signal
		
		return new_data_frame
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass


@stop_watch
def noise_reduction(data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2, parameter_signal):
	"""
	A function that excludes outliers from the data
	
	:param data_frame_fp1_ave:
	:param data_frame_fp2_ave:
	:param data_frame_fp1_fp2:
	:param parameter_signal:
	:return:
	"""
	
	# get data frame name
	data_frame_fp1_ave_name = data_frame_fp1_ave.name
	data_frame_fp2_ave_name = data_frame_fp2_ave.name
	data_frame_fp1_fp2_name = data_frame_fp1_fp2.name
	
	# ----------------------------< fp1 - ave >-----------------------------------------------------
	print(
		f"------------------------------------------------------------------------------------------\n"
		f" Confirmation of original distribution status({data_frame_fp1_ave_name} stage)\n"
		f"------------------------------------------------------------------------------------------\n"
	)
	# create basic statical data
	statistical_fp1_ave_time = np.array([])
	statistical_fp1_ave_signal = np.array([])
	
	# collect statical data
	for i_collect in range(0, data_frame_fp1_ave.shape[1], 2):
		tmp_column_time = data_frame_fp1_ave.columns[i_collect]
		statistical_fp1_ave_time = \
			np.append(statistical_fp1_ave_time, data_frame_fp1_ave[tmp_column_time])
		
		tmp_column_signal = data_frame_fp1_ave.columns[i_collect + 1]
		statistical_fp1_ave_signal = \
			np.append(statistical_fp1_ave_signal, data_frame_fp1_ave[tmp_column_signal])
	
	try:
		fp1_ave_average, fp1_ave_min, fp1_ave_max, fp1_ave_std, fp1_ave_median = \
			output_of_basic_statistics(statistical_fp1_ave_signal)
		
		print(f"fp1_ave_average_include_noise({data_frame_fp1_ave_name}) = {fp1_ave_average}")
		print(f"fp1_ave_min_include_noise({data_frame_fp1_ave_name}) = {fp1_ave_min}")
		print(f"fp1_ave_max_include_noise({data_frame_fp1_ave_name}) = {fp1_ave_max}")
		print(f"fp1_ave_std_include_noise({data_frame_fp1_ave_name}) = {fp1_ave_std}")
		print(f"fp1_ave_median_include_noise({data_frame_fp1_ave_name}) = {fp1_ave_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	print(
		f"----------------------------------------------------------------------------------------------------\n"
		f"             Noise reduction({data_frame_fp1_ave_name} stage)\n"
		f"----------------------------------------------------------------------------------------------------\n"
	)
	
	# Convert basic statical data to new basic statical
	data_frame_fp1_ave = data_correction(data_frame_fp1_ave, parameter_signal)
	
	# create basic statical data
	new_statistical_fp1_ave_time = np.array([])
	new_statistical_fp1_ave_signal = np.array([])
	
	# collect fp1_ave data
	for i_fp1_collect in range(0, data_frame_fp1_ave.shape[1], 2):
		tmp_column_time = data_frame_fp1_ave.columns[i_fp1_collect]
		new_statistical_fp1_ave_time = \
			np.append(new_statistical_fp1_ave_time, data_frame_fp1_ave[tmp_column_time])
		
		tmp_column_signal = data_frame_fp1_ave.columns[i_fp1_collect + 1]
		new_statistical_fp1_ave_signal = \
			np.append(new_statistical_fp1_ave_signal, data_frame_fp1_ave[tmp_column_signal])
	
	try:
		new_fp1_ave_average, new_fp1_ave_min, new_fp1_ave_max, new_fp1_ave_std, new_fp1_ave_median = \
			output_of_basic_statistics(new_statistical_fp1_ave_signal)
		
		print(f"fp1_ave_average({data_frame_fp1_ave_name}) = {new_fp1_ave_average}")
		print(f"fp1_ave_min({data_frame_fp1_ave_name}) = {new_fp1_ave_min}")
		print(f"fp1_ave_max({data_frame_fp1_ave_name}) = {new_fp1_ave_max}")
		print(f"fp1_ave_std({data_frame_fp1_ave_name}) = {new_fp1_ave_std}")
		print(f"fp1_ave_median({data_frame_fp1_ave_name}) = {new_fp1_ave_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	# ----------------------------< fp2 - ave >-----------------------------------------------------
	print(
		f"----------------------------------------------------------------------------------------------------\n"
		f"       Confirmation of original distribution status({data_frame_fp2_ave_name} stage)\n"
		f"----------------------------------------------------------------------------------------------------\n"
	)
	# create basic statical data
	statistical_fp2_ave_time = np.array([])
	statistical_fp2_ave_signal = np.array([])
	
	# collect statical data
	for i_collect in range(0, data_frame_fp2_ave.shape[1], 2):
		tmp_column_time = data_frame_fp2_ave.columns[i_collect]
		statistical_fp2_ave_time = \
			np.append(statistical_fp2_ave_time, data_frame_fp2_ave[tmp_column_time])
		
		tmp_column_signal = data_frame_fp2_ave.columns[i_collect + 1]
		statistical_fp2_ave_signal = \
			np.append(statistical_fp2_ave_signal, data_frame_fp2_ave[tmp_column_signal])
	
	try:
		fp2_ave_average, fp2_ave_min, fp2_ave_max, fp2_ave_std, fp2_ave_median = \
			output_of_basic_statistics(statistical_fp2_ave_signal)
		
		print(f"fp2_ave_average_include_noise({data_frame_fp2_ave_name}) = {fp2_ave_average}")
		print(f"fp2_ave_min_include_noise({data_frame_fp2_ave_name}) = {fp2_ave_min}")
		print(f"fp2_ave_max_include_noise({data_frame_fp2_ave_name}) = {fp2_ave_max}")
		print(f"fp2_ave_std_include_noise({data_frame_fp2_ave_name}) = {fp2_ave_std}")
		print(f"fp2_ave_median_include_noise({data_frame_fp2_ave_name}) = {fp2_ave_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	print(
		f"----------------------------------------------------------------------------------------------------\n"
		f"             Noise reduction({data_frame_fp2_ave_name} stage)\n"
		f"----------------------------------------------------------------------------------------------------\n"
	)
	# Convert basic statical data to new basic statical
	data_frame_fp2_ave = data_correction(data_frame_fp2_ave, parameter_signal)
	
	# create basic statical data
	new_statistical_fp2_ave_time = np.array([])
	new_statistical_fp2_ave_signal = np.array([])
	
	# collect fp1_ave data
	for i_fp2_collect in range(0, data_frame_fp2_ave.shape[1], 2):
		tmp_column_time = data_frame_fp2_ave.columns[i_fp2_collect]
		new_statistical_fp2_ave_time = \
			np.append(new_statistical_fp2_ave_time, data_frame_fp2_ave[tmp_column_time])
		
		tmp_column_signal = data_frame_fp2_ave.columns[i_fp2_collect + 1]
		new_statistical_fp2_ave_signal = \
			np.append(new_statistical_fp2_ave_signal, data_frame_fp2_ave[tmp_column_signal])
	
	try:
		new_fp2_ave_average, new_fp2_ave_min, new_fp2_ave_max, new_fp2_ave_std, new_fp2_ave_median = \
			output_of_basic_statistics(new_statistical_fp2_ave_signal)
		
		print(f"fp2_ave_average({data_frame_fp2_ave_name}) = {new_fp2_ave_average}")
		print(f"fp2_ave_min({data_frame_fp2_ave_name}) = {new_fp2_ave_min}")
		print(f"fp2_ave_max({data_frame_fp2_ave_name}) = {new_fp2_ave_max}")
		print(f"fp2_ave_std({data_frame_fp2_ave_name}) = {new_fp2_ave_std}")
		print(f"fp2_ave_median({data_frame_fp2_ave_name}) = {new_fp2_ave_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	# ----------------------------< fp1 - FP2 >-----------------------------------------------------
	print(
		f"----------------------------------------------------------------------------------------------------\n"
		f"    Confirmation of original distribution status({data_frame_fp1_fp2_name} stage)\n"
		f"----------------------------------------------------------------------------------------------------\n"
	)
	# create basic statical data
	statistical_fp1_fp2_time = np.array([])
	statistical_fp1_fp2_signal = np.array([])
	
	# collect statical data
	for i_collect in range(0, data_frame_fp1_fp2.shape[1], 2):
		tmp_column_time = data_frame_fp1_fp2.columns[i_collect]
		statistical_fp1_fp2_time = \
			np.append(statistical_fp1_fp2_time, data_frame_fp1_fp2[tmp_column_time])
		
		tmp_column_signal = data_frame_fp1_fp2.columns[i_collect + 1]
		statistical_fp1_fp2_signal = \
			np.append(statistical_fp1_fp2_signal, data_frame_fp1_fp2[tmp_column_signal])
	
	try:
		fp1_fp2_average, fp1_fp2_min, fp1_fp2_max, fp1_fp2_std, fp1_fp2_median = \
			output_of_basic_statistics(statistical_fp1_fp2_signal)
		
		print(f"fp1_fp2_average_include_noise({data_frame_fp1_fp2_name}) = {fp1_fp2_average}")
		print(f"fp1_fp2_min_include_noise({data_frame_fp1_fp2_name}) = {fp1_fp2_min}")
		print(f"fp1_fp2_max_include_noise({data_frame_fp1_fp2_name}) = {fp1_fp2_max}")
		print(f"fp1_fp2_std_include_noise({data_frame_fp1_fp2_name}) = {fp1_fp2_std}")
		print(f"fp1_fp2_median_include_noise({data_frame_fp1_fp2_name}) = {fp1_fp2_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	print(
		f"----------------------------------------------------------------------------------------------------\n"
		f"             Noise reduction({data_frame_fp1_fp2_name} stage)\n"
		f"----------------------------------------------------------------------------------------------------\n"
	)
	# Convert basic statical data to new basic statical
	data_frame_fp1_fp2 = data_correction(data_frame_fp1_fp2, parameter_signal)
	
	# create basic statical data
	new_statistical_fp1_fp2_time = np.array([])
	new_statistical_fp1_fp2_signal = np.array([])
	
	# collect fp1_ave data
	for i_fp12_collect in range(0, data_frame_fp1_fp2.shape[1], 2):
		tmp_column_time = data_frame_fp1_fp2.columns[i_fp12_collect]
		new_statistical_fp1_fp2_time = \
			np.append(new_statistical_fp1_fp2_time, data_frame_fp1_fp2[tmp_column_time])
		
		tmp_column_signal = data_frame_fp1_fp2.columns[i_fp12_collect + 1]
		new_statistical_fp1_fp2_signal = \
			np.append(new_statistical_fp1_fp2_signal, data_frame_fp1_fp2[tmp_column_signal])
	
	try:
		new_fp1_fp2_average, new_fp1_fp2_min, new_fp1_fp2_max, new_fp1_fp2_std, new_fp1_fp2_median = \
			output_of_basic_statistics(new_statistical_fp1_fp2_signal)
		
		print(f"fp1_fp2_average({data_frame_fp1_fp2_name}) = {new_fp1_fp2_average}")
		print(f"fp1_fp2_min({data_frame_fp1_fp2_name}) = {new_fp1_fp2_min}")
		print(f"fp1_fp2_max({data_frame_fp1_fp2_name}) = {new_fp1_fp2_max}")
		print(f"fp1_fp2_std({data_frame_fp1_fp2_name}) = {new_fp1_fp2_std}")
		print(f"fp1_fp2_median({data_frame_fp1_fp2_name}) = {new_fp1_fp2_median}\n")
	
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass
	
	return data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2


class FftProcessor(Process):
	"""

	"""
	
	def __init__(self, arguments):
		super().__init__()
		self.__arguments = arguments
	
	def ov(self, data, sample_rate, fs, overlap):
		"""

		:param data:
		:param sample_rate:
		:param fs:
		:param overlap:
		:return:
		"""
		try:
			# Total data length
			ts = len(data) / sample_rate
			
			# Frame period
			fc = fs / sample_rate
			
			# Frame displacement width for overlap
			x_ol = fs * (1 - (overlap / 100))
			
			# Number of frames to extract (number of data to use for averaging)
			n_ave = int((ts - (fc * (overlap / 100))) / (fc * (1 - (overlap / 100))))
			
			# Define an empty array to contain the extracted data
			array = []
			
			for i in range(n_ave):
				"""
				Extract data in a for loop.
				"""
				# The cutout position is updated in each loop.
				ps = int(x_ol * i)
				
				# Extract the frame size from the cutout position ps and add it to the array.
				array.append(data[ps:ps + fs:1])
			
			# The overlapped extracted data array and the number of data are used as the return value.
			return np.array(array), np.array(n_ave)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	def hanning(self, data_array, fs, rate, n_ave):
		"""
		Window function processing (Hanning window)

		:param data_array:
		:param fs:
		:param n_ave:
		:return:
		"""
		try:
			# Create a Hanning window.
			han = signal.hanning(fs)
			
			# Amplitude Correction Factor
			acf = 1 / (sum(han) / fs)
			
			time_hann = np.arange(0, fs) / rate
			
			# Apply a window function to all overlapped multiple time waveforms.
			for i in range(n_ave):
				for j in range(0, data_array.shape[2], 2):
					tmp_array = data_array[i, :, j + 1]
					
					# Apply the window function.
					data_array[i, :, j] = time_hann
					data_array[i, :, j + 1] = tmp_array * han
			
			return data_array, acf
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	def fft_ave(self, data_array, sample_rate, fs, n_ave, acf):
		"""
		FFT processing

		:param data_array:
		:param sample_rate:
		:param fs:
		:param n_ave:
		:param acf:
		:return:
		"""
		try:
			# ----------------------< matrix process by numpy >---------------------------
			fft_time_array = np.array([])
			fft_signal_array = np.array([])
			
			for i in range(n_ave):
				tmp_array = data_array[i, :, :]
				for j in range(0, data_array.shape[2], 2):
					tmp_time_array = tmp_array[:, j]
					tmp_signal_array = tmp_array[:, j + 1]
					fft_time_array = np.append(fft_time_array, tmp_time_array)
					
					# FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
					fft_signal_array = \
						np.append(fft_signal_array, acf * np.abs(fftpack.fft(tmp_signal_array) / (fs / 2)))
			
			# reshape naarray[fsxn_ave] to ndarray[fs, n_ave]
			num_of_column = int(fft_time_array.size / fs)
			fft_time_array = fft_time_array.reshape([num_of_column, fs]).T
			
			# reshape naarray[fsxn_ave] to ndarray[fs, n_ave]
			num_of_column = int(fft_signal_array.size / fs)
			fft_signal_array = fft_signal_array.reshape([num_of_column, fs]).T
			
			# Create a frequency axis.
			fft_axis = np.linspace(0, sample_rate, fs).T
			
			# extend  data without np.nan
			fft_signal_array_new = fft_signal_array[:, ~np.isnan(fft_signal_array).all(axis=0)]
			
			fft_sum = np.nansum(fft_signal_array_new, axis=0)
			
			fft_signal_array_new = fft_signal_array_new / fft_sum
			
			# Calculate the average of all FFT waveforms.
			fft_mean = np.nanmean(fft_signal_array_new, axis=1)
			
			df_fft_array = pd.DataFrame(fft_signal_array)
			
			df_fft_mean = pd.DataFrame(fft_mean)
			
			df_fft_axis = pd.DataFrame(fft_axis)
			
			return df_fft_array, df_fft_mean, df_fft_axis
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	def advanced_fft_analyze(self, arguments: list):
		"""

		:param arguments:
			arguments [0]: time_interval(int)
			arguments [1]: root_dir(str)
			arguments [2]: signal_name(str)
			arguments [3]: data_frame(DataFrame)
			arguments [4]: total_epoc_number(int)
			arguments [5]: frequency(float)
			arguments [6]: fs(int)
			arguments [7]: over_lap(int)
			arguments [8]: hanning_csv(bool)
			arguments [9]: fft_csv(bool)
			arguments[10]: read_file_name(str)
		:return:
		"""
		data_frame = arguments[3]
		
		read_file_name = arguments[10]
		
		print(
			f"-----------------------------------------------------------------------------------\n"
			f"              FFT starting {arguments[2]}\n"
			f"-----------------------------------------------------------------------------------\n"
		)
		# Data-Frame that temporarily stores data during the measurement interval
		stack_dataframe = pd.DataFrame()
		
		# Data-Frame that temporarily saves the result of averaging FFT for each measurement interval
		fft_amplitude_dataframe = pd.DataFrame()
		
		# This data-frame is consolidated at each measurement interval.
		fft_concat_dataframe = pd.DataFrame()
		
		calculate_times = int((arguments[0] * 2) / 30)
		
		for i, tmp_data in enumerate(data_frame):
			if not (i % 2):
				"""
				-------------------------------------------------------------
				Processing when even numbers(time value)
				-------------------------------------------------------------
				"""
				# update time data
				time_array = data_frame[tmp_data]
				
				# EPOC number
				epoc_number = int(i / 2)
				
				# total time (sec)
				total_time = epoc_number * 30
				
				print(f"{arguments[2]}_epoc_number: {epoc_number} / {arguments[4]}")
				
				if not (i % calculate_times):
					"""
					-------------------------------------------------------------
					Processing when it matches the calculation interval
					-------------------------------------------------------------
					"""
					for tmp in stack_dataframe:
						stack_dataframe = stack_dataframe.drop(tmp, axis=1)
			else:
				"""
				----------------------------------------------------------------
				Processing when odd numbers(signal value)
				----------------------------------------------------------------
				"""
				# signal data
				signal_array = data_frame[tmp_data]
				
				# create DataFrame
				tmp_data_frame = pd.concat([time_array, signal_array], axis=1)
				
				column_time = f"time_epoc_{epoc_number}"
				column_signal = f"{arguments[2]}_epoc_{epoc_number}"
				
				tmp_data_frame.columns = [column_time, column_signal]
				
				# Concatenate related columns.
				stack_dataframe = pd.concat([stack_dataframe, tmp_data_frame], axis=1)
				
				df_columns = stack_dataframe.columns
				
				if (i % calculate_times) == (calculate_times - 1):
					"""
					-----------------------------------------------------------
					calculate averaged FFT
					-----------------------------------------------------------
					"""
					# Calculate overlap
					time_array_signal, n_ave_signal = self.ov(stack_dataframe, arguments[5], arguments[6], arguments[7])
					
					# set stored folder path
					data_hanning_dir = os.path.join(arguments[1], 'Data', 'Hanning', arguments[2])
					
					# Calculate fp1_Ave
					time_array_signal, acf_signal = \
						self.hanning(time_array_signal, arguments[6], arguments[5], n_ave_signal)
					
					# convert dataframe to csv
					for j in range(n_ave_signal):
						# get nd-array each hanning window.
						tmp_item = time_array_signal[j, :, :]
						df_tmp = pd.DataFrame(tmp_item)
						if arguments[8]:
							name_tmp = f'_{arguments[2]}_hanning_time_{j}'
							fft_folder.convert_csv(read_file_name[0], data_hanning_dir, name_tmp, df_tmp)
					
					# Calculate averaged fft
					fft_array_signal, fft_mean_signal, fft_axis_signal = \
						self.fft_ave(time_array_signal, arguments[5], arguments[6], n_ave_signal, acf_signal)
					
					column_freq = f"freq_epoc_{epoc_number}"
					column_signal = f"{arguments[2]}_epoc_{epoc_number}"
					
					# Concatenate related columns.
					fft_concat_signal_epoc = pd.concat([fft_axis_signal, fft_mean_signal], axis=1)
					
					fft_concat_signal_epoc.columns = [column_freq, column_signal]
					
					fft_concat_dataframe = pd.concat([fft_concat_dataframe, fft_concat_signal_epoc], axis=1)
		
		if arguments[9]:
			# set stored folder path
			data_fft_dir = os.path.join(arguments[1], 'Data', 'Fft', arguments[2])
			
			# output csv file of data_frame_fp1_ave
			fft_folder.convert_csv(
				read_file_name[0], data_fft_dir, f'_{arguments[2]}_fft', fft_concat_dataframe
			)
	
	def run(self):
		self.advanced_fft_analyze(self.__arguments)


