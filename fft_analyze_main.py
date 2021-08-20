"""
------------------------------------------------------------
edf_analyze_main.py (Use the device edf file and the csv file stage-classified by the clinical technician.)


PyEDFlib is a Python library to read/write EDF/EDF+/BDF files based on EDFlib.

EDF stands for European Data Format, a data format for EEG data, first published in 1992. In 2003,
n improved version of the file protocol named EDF+ has been published.

The definition of the EDF/EDF+ format can be found under edfplus.info.

The EDF/EDF+ format saves all data with 16 Bit. A version of the format which saves all data with 24 Bit,
called BDF, was introduced by the company BioSemi.

The PyEDFlib Python toolbox is a fork of the python-edf toolbox from Christopher Lee-Messer.
and uses the EDFlib from Teunis van Beelen.
------------------------------------------------------------------------------------------------------------------------

EdfReader
- getNSamples(self)
  - Returns the number of data points.
　　 The value for each channel is output as [100 200 100 100].
- readAnnotations(self)
- getHeader(self)
　- Returns subject and device information. Probably not used very often.
- getSignalHeader(self, chn)
　- Returns the information (sensor name, sampling frequency, etc.) of the channel specified by the argument.
- getSignalHeaders(self)
　- Process the above function on all channels simultaneously.
- getTechnician(self)
- getRecordingAdditional(self)
- getPatientName(self)
- getPatientCode(self)
- getPatientAdditional(self)
- getEquipment(self)
- getAdmincode(self)
- getGender(self)
- getFileDuration(self)
　- Returns the measurement time. The unit is seconds.
- getStartdatetime(self)
- getBirthdate(self, string=True)
- getSampleFrequencies(self)
- getSampleFrequency(self,chn)
- getSignalLabels(self)
　- The sensor name will be returned for all channels.
- getLabel(self,chn)
　- The sensor name of the specified channel will be returned.
- getPrefilter(self,chn)
　- Returns the filter information used for preprocessing for the specified channel.
- getPhysicalMaximum(self,chn=None)
- getPhysicalMinimum(self,chn=None)
- getDigitalMaximum(self, chn=None)
- getDigitalMinimum(self, chn=None)
- getTransducer(self, chn)
　- Returns the type of the measurement device for the specified channel.
- getPhysicalDimension(self, chn)
　- Returns the unit of measurement data for a given channel, e.g. uV, mA, etc.
- readSignal(self, chn, start=0, n=None)
　- Returns the measurement data for the specified channel. Most used.
- file_info(self)
- file_info_long(self)
------------------------------------------------------------
"""

# python library
import os
import sys
import glob
import time
import gc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime
import fft_folder_func as fft_folder
import fft_siganl_func as fft_signal
import fft_graph_func as fft_graph
import fft_machine_func as fft_machine

# pyedfLib
import pyedflib as edf_lib

# numpy
import numpy as np

# pandas
import pandas as pd

# Scikit-learn library
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sn

print(__doc__)

if __name__ == '__main__':
	"""
	main routine
	"""
	# Start to measure processing time
	start = time.time()
	
	# get current folder path.
	project_root_dir = os.getcwd()
	print(f'project_root_dir = {project_root_dir}')
	
	# Setting the display time
	pause_time = 5
	
	# coefficient related DBSCAN
	eps_value = 0.72
	threshold_ratio = 0.12
	
	# iqr range
	parameter_iqr = 1.5
	parameter_iqr_signal = 3.0
	
	create_datasets_flag = False
	
	"""
	----------------------------------------
	Set enable flag of outputting csv file
	----------------------------------------
	"""
	# set whether raw_edf_data is output of not.
	raw_csv_edf_enable = False
	
	# set whether raw_data is output  each EPOC or not.
	raw_csv_epoc_enable = False
	
	# set whether filter_data is output of not.
	filter_csv_enable = False
	
	# set whether induction_data is output of not.
	induction_csv_enable = False
	
	# set whether outlier_data is output of not.
	outlier_csv_enable = False
	
	# set whether hanning_data is output of not.
	hanning_csv_enable = False
	
	# set whether fft_data is output of not.
	fft_csv_enable = True
	print(f"Create datasets?: {fft_csv_enable}\n")
	
	"""
	----------------------------------------
	Set enable flag of outputting csv file
	----------------------------------------
	"""
	# set whether raw_graphic is output of not.
	raw_graph_enable = False
	
	# set whether filter_graphic is output of not.
	filter_graph_enable = False
	
	# set whether inductive_graphic is output of not.
	inductive_graph_enable = False
	
	# set whether outlier_graphic is output of not.
	outlier_graph_enable = False
	
	# set whether fft_graphic is output of not.
	fft_graph_enable = True
	
	# Flag to adjust High parameter or not
	search_mode = False
	print('search_mode: {0}'.format(search_mode))
	
	print(
		f"----------------------------------------------------------------------\n"
		f"                           Confirm def file\n"
		f"----------------------------------------------------------------------\n"
	)
	# confirm whether required files exist or not.
	edf_filter = os.path.join(project_root_dir, 'Data', '*.edf')
	csv_filter = os.path.join(project_root_dir, 'Data', '*.csv')
	
	# create file list
	edf_list = glob.glob(edf_filter)
	csv_list = glob.glob(csv_filter)
	
	# extend file name
	if not len(edf_list):
		# Output message.
		print(f"edf files don't exist!\n")
		# end
		exit()
	else:
		# Delete duplicate files
		edf_hash = fft_folder.delete_duplicate_edf(edf_list)
		edf_list = list(edf_hash.values())
	
	if not len(csv_list):
		# output message.
		print(f"csv files don't exist!\n")
		# end
		exit()
	else:
		csv_hash = fft_folder.delete_duplicate_edf(csv_list)
		csv_list = list(csv_hash.values())
	
	print(
		f"----------------------------------------------------------------------\n"
		f"                           Confirm folder structure\n"
		f"----------------------------------------------------------------------\n"
	)
	
	# Create required Folder
	fft_folder.create_folder(project_root_dir)
	
	print(
		f"----------------------------------------------------------------------\n"
		f"                    Start FFT analysis for each file.\n"
		f"----------------------------------------------------------------------\n"
	)
	
	if create_datasets_flag:
		# Create a DataFrame to be used for classification.
		df_datasets = pd.DataFrame()
		
		# Flag that indicates whether or not the processing was performed only in the first processing of df_datasets.
		df_datasets_first_flag = True
		
		#
		first_process_label_flag = True
		
		# ---------------------< start of main routine >----------------------------
		for file_path in edf_list:
			"""
			Output information each edf file.
			"""
			
			# get file name with extension
			file_name_with_extension = os.path.basename(file_path)
			
			# get_name_without_
			
			print(
				f"----------------------------------------------------------------------\n"
				f"      Start FFT analysis for {file_name_with_extension}.\n"
				f"----------------------------------------------------------------------\n"
			)
			
			# Read edf file
			edf = edf_lib.EdfReader(file_path)
			
			print(f"--------------< information edf file >-----------------\n")
			
			# get a label for signal
			signal_labels = edf.getSignalLabels()
			
			for data_information in signal_labels:
				print(data_information)
			
			print()
			
			print(f"Duration: {edf.getFileDuration()}(sec)")
			print(f"Frequency:\n    {edf.getSampleFrequencies()}(Hz)")
			print(f"N-Samples(=freq. x Duration):\n      {edf.getNSamples()}\n")
			
			# get signal information
			for ch in range(len(signal_labels)):
				print(f"signal information ch[{ch}] = \n {edf.getSignalHeader(ch)}")
			
			print()
			
			# calculate a number of data relating signal.
			display_range_of_signal = edf.getSampleFrequencies()[0] * 30
			
			# calculate a number of data relating resistance.
			display_range_of_resistance = edf.getSampleFrequencies()[6] * 30
			
			# calculate a number of data relating voltage.
			display_range_of_voltage = edf.getSampleFrequencies()[10] * 30
			
			# Create from 0 sec to 30 sec
			time_signal = np.array([])
			
			for i in range(display_range_of_signal):
				tmp_real_time = i / edf.getSampleFrequencies()[0]
				time_signal = np.append(time_signal, tmp_real_time)
			
			# Convert time to string.
			header_signal_str = [str(n) for n in time_signal]
			
			time_resistance = np.array([])
			
			for i in range(display_range_of_resistance):
				tmp_real_time = i / edf.getSampleFrequencies()[6]
				time_resistance = np.append(time_resistance, tmp_real_time)
			
			# Convert time to string.
			header_resistance_str = [str(n) for n in time_resistance]
			
			time_voltage = np.array([])
			
			for i in range(display_range_of_voltage):
				tmp_real_time = i / edf.getSampleFrequencies()[10]
				time_voltage = np.append(time_voltage, tmp_real_time)
			
			# Convert time to string.
			header_voltage_str = [str(n) for n in time_voltage]
			
			print(
				f"-------------------------------------------------------------------\n"
				f"               Output the signal distribution.\n"
				f"-------------------------------------------------------------------\n"
			)
			
			# get signal ch0 information(Fp1)
			ch0_signal = fft_signal.signal_value(edf, 0)
			
			# Basic statistical data of ch0_signal
			ch0_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch0_signal[:, 1])
			
			fft_signal.print_signal("ch0", ch0_signal_basic_statistics)
			
			# get signal ch1 information(Fp2)
			ch1_signal = fft_signal.signal_value(edf, 1)
			
			# Basic statistical data of ch0_signal
			ch1_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch1_signal[:, 1])
			
			fft_signal.print_signal("ch1", ch1_signal_basic_statistics)
			
			# get signal ch2 information(A1)
			ch2_signal = fft_signal.signal_value(edf, 2)
			
			# Basic statistical data of ch2_signal
			ch2_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch2_signal[:, 1])
			
			fft_signal.print_signal("ch2", ch2_signal_basic_statistics)
			
			# get signal ch3 information(A2)
			ch3_signal = fft_signal.signal_value(edf, 3)
			
			# Basic statistical data of ch3_signal
			ch3_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch3_signal[:, 1])
			
			fft_signal.print_signal("ch3", ch3_signal_basic_statistics)
			
			# get signal ch4 information(Light OFF)
			ch4_signal = fft_signal.signal_value(edf, 4)
			
			# get signal ch5 information(R_A1)
			ch5_signal = fft_signal.signal_value(edf, 5)
			
			# Basic statistical data of ch5_signal
			ch5_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch5_signal[:, 1])
			
			fft_signal.print_signal("ch5", ch5_signal_basic_statistics)
			
			# get signal ch6 information(R_Fp1)
			ch6_signal = fft_signal.signal_value(edf, 6)
			
			# Basic statistical data of ch6_signal
			ch6_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch6_signal[:, 1])
			
			fft_signal.print_signal("ch6", ch6_signal_basic_statistics)
			
			# get signal ch7 information(R_Ref)
			ch7_signal = fft_signal.signal_value(edf, 7)
			
			# Basic statistical data of ch7_signal
			ch7_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch7_signal[:, 1])
			
			fft_signal.print_signal("ch7", ch7_signal_basic_statistics)
			
			# get signal ch8 information(R_Fp2)
			ch8_signal = fft_signal.signal_value(edf, 8)
			
			# Basic statistical data of ch8_signal
			ch8_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch8_signal[:, 1])
			
			fft_signal.print_signal("ch8", ch8_signal_basic_statistics)
			
			# get signal ch9 information(R_A2)
			ch9_signal = fft_signal.signal_value(edf, 9)
			
			# Basic statistical data of ch9_signal
			ch9_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch9_signal[:, 1])
			
			fft_signal.print_signal("ch9", ch9_signal_basic_statistics)
			
			# get signal ch10 information(V_A1)
			ch10_signal = fft_signal.signal_value(edf, 10)
			
			# Basic statistical data of ch10_signal
			ch10_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch10_signal[:, 1])
			
			fft_signal.print_signal("ch10", ch10_signal_basic_statistics)
			
			# get signal ch11 information(V_Fp1)
			ch11_signal = fft_signal.signal_value(edf, 11)
			
			# Basic statistical data of ch11_signal
			ch11_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch11_signal[:, 1])
			
			fft_signal.print_signal("ch11", ch11_signal_basic_statistics)
			
			# get signal ch12 information(V_Fp2)
			ch12_signal = fft_signal.signal_value(edf, 12)
			
			# Basic statistical data of ch12_signal
			ch12_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch12_signal[:, 1])
			
			fft_signal.print_signal("ch12", ch12_signal_basic_statistics)
			
			# get signal ch13 information(V_A2)
			ch13_signal = fft_signal.signal_value(edf, 13)
			
			# Basic statistical data of ch13_signal
			ch13_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch13_signal[:, 1])
			
			fft_signal.print_signal("ch13", ch13_signal_basic_statistics)
			
			# get signal ch14 information(V_Battery)
			ch14_signal = fft_signal.signal_value(edf, 14)
			
			# Basic statistical data of ch13_signal
			ch14_signal_basic_statistics = fft_signal.output_of_basic_statistics(ch14_signal[:, 1])
			
			fft_signal.print_signal("ch14", ch14_signal_basic_statistics)
			
			# get signal ch15 information(T_Battery)
			ch15_signal = fft_signal.signal_value(edf, 15)
			
			# get signal ch16 information(T_PCB)
			ch16_signal = fft_signal.signal_value(edf, 16)
			
			# get signal ch17 information(Rec_No_upper)
			ch17_signal = fft_signal.signal_value(edf, 17)
			
			# get signal ch18 information(Rec_No_lower)
			ch18_signal = fft_signal.signal_value(edf, 18)
			
			# get signal ch19 information(Time_upper)
			ch19_signal = fft_signal.signal_value(edf, 19)
			
			# get signal ch20 information(Time_lower)
			ch20_signal = fft_signal.signal_value(edf, 20)
			
			print(
				f"-------------------------------------------------------------------\n"
				f"Read the csv file of stage classification by a clinical technician.\n"
				f"-------------------------------------------------------------------\n"
			)
			# read csv file
			read_file_name = os.path.splitext(os.path.basename(file_path))
			read_csv_file_name = read_file_name[0] + '-ct.csv'
			
			# If the specified csv file does not exist, exit the program.
			read_csv_file_path = os.path.join(project_root_dir, 'Data', read_csv_file_name)
			if not os.path.isfile(read_csv_file_path):
				sys.exit()
			
			# Read csv data to pandas.
			df_clinical_data = pd.read_csv(read_csv_file_path)
			
			# Display the first data to check if the data was read normally.
			print(f"df_clinical_data = \n    {df_clinical_data.head(10)}\n")
			
			# initialize datetime
			date_str = datetime.now()
			dt_start = datetime.now()
			
			# initialize string
			tstr = ""
			
			# Convert DateTime display to Time display.
			for i in range(df_clinical_data.shape[0]):
				if not i:
					date_str = datetime.now()
					tstr = date_str.strftime('%Y/%m/%d')
					start_time = tstr + ' ' + str(df_clinical_data.iloc[i, 1])
					dt_start = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
					
					epoc_time = 0
				else:
					tstr = date_str.strftime('%Y/%m/%d')
					current_time = tstr + ' ' + str(df_clinical_data.iloc[i, 1])
					dt_current = datetime.strptime(current_time, '%Y/%m/%d %H:%M:%S')
					
					time_subtraction = dt_current - dt_start
					epoc_time = time_subtraction.seconds
				
				df_clinical_data.iloc[i, 1] = epoc_time
			
			print('modify dataframe = \n{0}'.format(df_clinical_data.head(10)))
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Once, each EDF signal is stored in the DataFrame.\n"
				f"---------------------------------------------------------------------\n"
			)
			# ------------------< signal >-----------------------
			# ch0(Fp1)
			df_signal_fp1 = pd.DataFrame(ch0_signal, columns=['fp1_time', 'signal_fp1'])
			
			# ch1(Fp2)
			df_signal_fp2 = pd.DataFrame(ch1_signal, columns=['fp2_time', 'signal_fp2'])
			
			# ch2(A1)
			df_signal_a1 = pd.DataFrame(ch2_signal, columns=['a1_time', 'signal_a1'])
			
			# ch3(A2)
			df_signal_a2 = pd.DataFrame(ch3_signal, columns=['a2_time', 'signal_a2'])
			
			# ------------------------< resistance >---------------------
			# ch5(R_A1)
			df_r_a1 = pd.DataFrame(ch5_signal, columns=['r_a1_time', 'resistance_a1'])
			
			# ch6(R_Fp1)
			df_r_fp1 = pd.DataFrame(ch6_signal, columns=['r_fp1_time', 'resistance_fp1'])
			
			# ch7(R_REF)
			df_r_ref = pd.DataFrame(ch7_signal, columns=['r_ref_time', 'resistance_ref'])
			
			# ch8(R_Fp2)
			df_r_fp2 = pd.DataFrame(ch8_signal, columns=['r_fp2_time', 'resistance_fp2'])
			
			# ch9(R_A2)
			df_r_a2 = pd.DataFrame(ch9_signal, columns=['r_a2_time', 'resistance_a2'])
			
			# Combine the columns of resistors to create a single DataFrame
			df_resistance = pd.concat([df_r_fp1, df_r_fp2, df_r_ref, df_r_a1, df_r_a2], axis=1)
			
			# -------------------------< voltage >----------------------------------
			# ch10(V_A1)
			df_v_a1 = pd.DataFrame(ch10_signal, columns=['v_a1_time', 'voltage_a1'])
			
			# ch11(V_Fp1)
			df_v_fp1 = pd.DataFrame(ch11_signal, columns=['v_fp1_time', 'voltage_fp1'])
			
			# ch12(V_Fp2)
			df_v_fp2 = pd.DataFrame(ch12_signal, columns=['v_fp2_time', 'voltage_fp2'])
			
			# ch13(V_A2)
			df_v_a2 = pd.DataFrame(ch13_signal, columns=['v_a2_time', 'voltage_a2'])
			
			# ch14(V_Battery)
			df_v_battery = pd.DataFrame(ch14_signal, columns=['v_battery_time', 'voltage_battery'])
			
			# Combine the columns of voltage to create a single DataFrame.
			df_voltage = pd.concat([df_v_fp1, df_v_fp2, df_v_a1, df_v_a2, df_v_battery], axis=1)
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Save the contents of the EDF file to a csv file.\n"
				f"---------------------------------------------------------------------\n"
			)
			
			file_name_without_extension = file_name_with_extension.split('.')[0]
			
			# basic directory name
			dir_name = os.path.dirname(file_path)
			
			if raw_csv_edf_enable:
				# Store Fp1 signal data to csv file.
				dir_name_fp1 = os.path.join(dir_name, 'Raw_Data', 'Fp1')
				fft_folder.edf_to_csv(dir_name_fp1, file_name_without_extension, 'fp1', df_signal_fp1)
				
				# Store Fp2 signal data to csv file.
				dir_name_fp2 = os.path.join(dir_name, 'Raw_Data', 'Fp2')
				fft_folder.edf_to_csv(dir_name_fp2, file_name_without_extension, 'fp2', df_signal_fp2)
				
				# Store A1 signal data to csv file.
				dir_name_a1 = os.path.join(dir_name, 'Raw_Data', 'A1')
				fft_folder.edf_to_csv(dir_name_a1, file_name_without_extension, 'a1', df_signal_a1)
				
				# Store A2 signal data to csv file.
				dir_name_a2 = os.path.join(dir_name, 'Raw_Data', 'A2')
				fft_folder.edf_to_csv(dir_name_a2, file_name_without_extension, 'a2', df_signal_a2)
				
				# Store resistors data to csv file.
				dir_name_resistance = os.path.join(dir_name, 'Raw_Data', 'Resistance')
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'resistance', df_resistance)
				
				# Store voltage data to csv file.
				dir_name_voltage = os.path.join(dir_name, 'Raw_Data', 'Voltage')
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'voltage', df_voltage)
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Delete the remainder of EPOC.\n"
				f"---------------------------------------------------------------------\n"
			)
			
			# calculate raw each EPOC(30sec)
			epoc_number_signal = int(df_signal_fp1.shape[0] / display_range_of_signal)
			epoc_remainder = int(df_signal_fp1.shape[0] % display_range_of_signal)
			
			print(f"epoc_num_signal = {epoc_number_signal}, epoc_remainder_signal = {epoc_remainder}")
			
			start_index = epoc_number_signal * display_range_of_signal
			end_index = start_index + epoc_remainder
			
			# Delete the remainder of EPOC relating Fp1.
			df_signal_fp1_md0 = df_signal_fp1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating Fp1.
			df_signal_fp2_md0 = df_signal_fp2.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating A1.
			df_signal_a1_md0 = df_signal_a1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating A2.
			df_signal_a2_md0 = df_signal_a2.drop(range(start_index, end_index))
			
			# calculate raw each EPOC(30sec)
			epoc_number_resistance = int(df_r_a1.shape[0] / display_range_of_resistance)
			epoc_remainder = int(df_r_a1.shape[0] % display_range_of_resistance)
			
			print(f"epoc_num_resistance = {epoc_number_resistance}, epoc_remainder_resistance = {epoc_remainder}")
			
			start_index = epoc_number_resistance * display_range_of_resistance
			end_index = start_index + epoc_remainder
			
			# Delete the remainder of EPOC relating r_fp1.
			df_r_fp1_md0 = df_r_fp1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating r_fp2.
			df_r_fp2_md0 = df_r_fp2.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating r_ref.
			df_r_ref_md0 = df_r_ref.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating r_a1.
			df_r_a1_md0 = df_r_a1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating r_a2.
			df_r_a2_md0 = df_r_a2.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating resistance.
			df_resistance_md0 = df_resistance.drop(range(start_index, end_index))
			
			# calculate raw each EPOC(30sec)
			epoc_number_voltage = int(df_v_a1.shape[0] / display_range_of_voltage)
			epoc_remainder = int(df_v_a1.shape[0] % display_range_of_voltage)
			
			print(f"epoc_num_voltage = {epoc_number_voltage}, epoc_remainder_voltage = {epoc_remainder}")
			
			start_index = epoc_number_voltage * display_range_of_voltage
			end_index = start_index + epoc_remainder
			
			# Delete the remainder of EPOC relating v_fp1.
			df_v_fp1_md0 = df_v_fp1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating v_fp2.
			df_v_fp2_md0 = df_v_fp2.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating v_a1.
			df_v_a1_md0 = df_v_a1.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating v_a2.
			df_v_a2_md0 = df_v_a2.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating v_battery.
			df_v_battery_md0 = df_v_battery.drop(range(start_index, end_index))
			
			# Delete the remainder of EPOC relating voltage.
			df_voltage_md0 = df_voltage.drop(range(start_index, end_index))
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Reshape the DataFrame and divide it for each EPOC.\n"
				f"---------------------------------------------------------------------\n"
			)
			# -----------------------------< signal >---------------------------------
			# convert fp1 dataframe to numpy.
			np_signal_fp1 = df_signal_fp1_md0.to_numpy()
			np_signal_fp1 = np_signal_fp1.reshape([epoc_number_signal, display_range_of_signal, np_signal_fp1.shape[1]])
			
			# Decompose the fp1 dataframe for each EPOC.
			df_epoc_fp1 = fft_signal.divide_dataframe_epoc('fp1', np_signal_fp1, header_signal_str)
			
			# convert fp2 dataframe to numpy.
			np_signal_fp2 = df_signal_fp2_md0.to_numpy()
			np_signal_fp2 = np_signal_fp2.reshape(epoc_number_signal, display_range_of_signal, np_signal_fp2.shape[1])
			
			# Decompose the fp2 dataframe for each EPOC.
			df_epoc_fp2 = fft_signal.divide_dataframe_epoc('fp2', np_signal_fp2, header_signal_str)
			
			# convert a1 dataframe to numpy.
			np_signal_a1 = df_signal_a1_md0.to_numpy()
			np_signal_a1 = np_signal_a1.reshape(epoc_number_signal, display_range_of_signal, np_signal_a1.shape[1])
			
			# Decompose the a1 dataframe for each EPOC.
			df_epoc_a1 = fft_signal.divide_dataframe_epoc('a1', np_signal_a1, header_signal_str)
			
			# convert a2 dataframe to numpy.
			np_signal_a2 = df_signal_a2_md0.to_numpy()
			np_signal_a2 = np_signal_a2.reshape(epoc_number_signal, display_range_of_signal, np_signal_a2.shape[1])
			
			# Decompose the a2 dataframe for each EPOC.
			df_epoc_a2 = fft_signal.divide_dataframe_epoc('a2', np_signal_a1, header_signal_str)
			
			# --------------------< resistance >--------------------------------------------
			# convert r_fp1 dataframe to numpy.
			np_r_fp1 = df_r_fp1_md0.to_numpy()
			np_r_fp1 = np_r_fp1.reshape([epoc_number_resistance, display_range_of_resistance, np_r_fp1.shape[1]])
			
			# Decompose the r_fp1 dataframe for each EPOC.
			df_epoc_r_fp1 = fft_signal.divide_dataframe_epoc('r_fp1', np_r_fp1, header_resistance_str)
			
			# convert r_fp2 dataframe to numpy.
			np_r_fp2 = df_r_fp2_md0.to_numpy()
			np_r_fp2 = np_r_fp2.reshape([epoc_number_resistance, display_range_of_resistance, np_r_fp2.shape[1]])
			
			# Decompose the r_fp2 dataframe for each EPOC.
			df_epoc_r_fp2 = fft_signal.divide_dataframe_epoc('r_fp2', np_r_fp2, header_resistance_str)
			
			# convert r_ref dataframe to numpy.
			np_r_ref = df_r_ref_md0.to_numpy()
			np_r_ref = np_r_ref.reshape([epoc_number_resistance, display_range_of_resistance, np_r_ref.shape[1]])
			
			# Decompose the r_ref dataframe for each EPOC.
			df_epoc_r_ref = fft_signal.divide_dataframe_epoc('r_ref', np_r_ref, header_resistance_str)
			
			# convert r_a1 dataframe to numpy.
			np_r_a1 = df_r_a1_md0.to_numpy()
			np_r_a1 = np_r_a1.reshape([epoc_number_resistance, display_range_of_resistance, np_r_a1.shape[1]])
			
			# Decompose the r_a1 dataframe for each EPOC.
			df_epoc_r_a1 = fft_signal.divide_dataframe_epoc('r_a1', np_r_a1, header_resistance_str)
			
			# convert r_a2 dataframe to numpy.
			np_r_a2 = df_r_a2_md0.to_numpy()
			np_r_a2 = np_r_a2.reshape([epoc_number_resistance, display_range_of_resistance, np_r_a2.shape[1]])
			
			# Decompose the r_a2 dataframe for each EPOC.
			df_epoc_r_a2 = fft_signal.divide_dataframe_epoc('r_a2', np_r_a2, header_resistance_str)
			
			# --------------------< voltage >--------------------------------------------
			# convert v_fp1 dataframe to numpy.
			np_v_fp1 = df_v_fp1_md0.to_numpy()
			np_v_fp1 = np_v_fp1.reshape([epoc_number_voltage, display_range_of_voltage, np_v_fp1.shape[1]])
			
			# Decompose the v_fp1 dataframe for each EPOC.
			df_epoc_v_fp1 = fft_signal.divide_dataframe_epoc('v_fp1', np_v_fp1, header_voltage_str)
			
			# convert v_fp2 dataframe to numpy.
			np_v_fp2 = df_v_fp2_md0.to_numpy()
			np_v_fp2 = np_v_fp2.reshape([epoc_number_voltage, display_range_of_voltage, np_v_fp2.shape[1]])
			
			# Decompose the v_fp2 dataframe for each EPOC.
			df_epoc_v_fp2 = fft_signal.divide_dataframe_epoc('v_fp2', np_v_fp2, header_voltage_str)
			
			# convert v_a1 dataframe to numpy.
			np_v_a1 = df_v_a1_md0.to_numpy()
			np_v_a1 = np_v_a1.reshape([epoc_number_voltage, display_range_of_voltage, np_v_a1.shape[1]])
			
			# Decompose the v_a1 dataframe for each EPOC.
			df_epoc_v_a1 = fft_signal.divide_dataframe_epoc('v_a1', np_v_a1, header_voltage_str)
			
			# convert v_a2 dataframe to numpy.
			np_v_a2 = df_v_a2_md0.to_numpy()
			np_v_a2 = np_v_a2.reshape([epoc_number_voltage, display_range_of_voltage, np_v_a2.shape[1]])
			
			# Decompose the v_a2 dataframe for each EPOC.
			df_epoc_v_a2 = fft_signal.divide_dataframe_epoc('v_a2', np_v_a2, header_voltage_str)
			
			# convert v_battery to numpy
			np_v_battery = df_v_battery_md0.to_numpy()
			np_v_battery = np_v_battery.reshape([epoc_number_voltage, display_range_of_voltage, np_v_battery.shape[1]])
			
			# Decompose the v_battery dataframe for each EPOC.
			df_epoc_v_battery = fft_signal.divide_dataframe_epoc('v_battery', np_v_battery, header_voltage_str)
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Save the contents of the EDF file(each EPOC) to a csv file.\n"
				f"---------------------------------------------------------------------\n"
			)
			
			file_name_without_extension = file_name_with_extension.split('.')[0]
			
			# basic directory name
			dir_name = os.path.dirname(file_path)
			
			if raw_csv_epoc_enable:
				# ---------------------< signal >---------------------------------------------------
				
				# Store Fp1 signal data to csv file.
				dir_name_fp1 = os.path.join(dir_name, 'Raw_Data', 'Fp1')
				fft_folder.edf_to_csv(dir_name_fp1, file_name_without_extension, 'epoc_fp1', df_epoc_fp1)
				
				# Store Fp2 signal data to csv file.
				dir_name_fp2 = os.path.join(dir_name, 'Raw_Data', 'Fp2')
				fft_folder.edf_to_csv(dir_name_fp2, file_name_without_extension, 'epoc_fp2', df_epoc_fp2)
				
				# Store A1 signal data to csv file.
				dir_name_a1 = os.path.join(dir_name, 'Raw_Data', 'A1')
				fft_folder.edf_to_csv(dir_name_a1, file_name_without_extension, 'epoc_a1', df_epoc_a1)
				
				# Store A2 signal data to csv file.
				dir_name_a2 = os.path.join(dir_name, 'Raw_Data', 'A2')
				fft_folder.edf_to_csv(dir_name_a2, file_name_without_extension, 'epoc_a2', df_epoc_a2)
				
				# ------------------------< resistance >------------------------------------------------------
				# set using folder.
				dir_name_resistance = os.path.join(dir_name, 'Raw_Data', 'Resistance')
				
				# Store r_fp1 data each epoc.
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'epoc_r_fp1', df_epoc_r_fp1)
				
				# Store r_fp2 data each epoc.
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'epoc_r_fp2', df_epoc_r_fp2)
				
				# Store r_ref data each epoc.
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'epoc_r_ref', df_epoc_r_ref)
				
				# Store r_a1 data each epoc.
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'epoc_r_a1', df_epoc_r_a1)
				
				# Store r_a2 data each epoc.
				fft_folder.edf_to_csv(dir_name_resistance, file_name_without_extension, 'epoc_r_a2', df_epoc_r_a2)
				
				# ------------------------< Voltage >------------------------------------------------------
				# set using folder
				dir_name_voltage = os.path.join(dir_name, 'Raw_Data', 'Voltage')
				
				# Store v_fp1 data each epoc.
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'epoc_v_fp1', df_v_fp1)
				
				# Store v_fp2 data each epoc.
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'epoc_v_fp2', df_v_fp2)
				
				# store v_a1 data each epoc.
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'epoc_v_a1', df_v_a1)
				
				# store v_a2 data each epoc.
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'epoc_v_a2', df_v_a2)
				
				# store v_battery each epoc.
				fft_folder.edf_to_csv(dir_name_voltage, file_name_without_extension, 'epoc_v_battery', df_v_battery)
			
			print(
				f"---------------------------------------------------------------------\n"
				f"      Display  signal data by graphic.\n"
				f"---------------------------------------------------------------------\n"
			)
			if raw_graph_enable:
				save_path = os.path.join(project_root_dir, 'Image', 'Raw_Data', 'Signal')
				fft_graph.raw_signal_plot(
					edf, time_signal,
					file_path,
					df_epoc_fp1,
					df_epoc_fp2,
					df_epoc_a1,
					df_epoc_a2,
					save_path
				)
				
				save_path = os.path.join(project_root_dir, 'Image', 'Raw_Data', 'Resistance')
				fft_graph.raw_resistance_plot(
					edf, time_resistance,
					file_path,
					df_epoc_r_fp1,
					df_epoc_r_fp2,
					df_epoc_r_ref,
					df_epoc_r_a1,
					df_epoc_r_a2,
					save_path
				)
				
				save_path = os.path.join(project_root_dir, 'Image', 'Raw_Data', 'Voltage')
				fft_graph.raw_voltage_plot(
					edf, time_voltage,
					file_path,
					df_epoc_v_fp1,
					df_epoc_v_fp2,
					df_epoc_v_a1,
					df_epoc_v_a2,
					df_epoc_v_battery,
					save_path
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              filtering\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			
			frequency_ch = edf.getSampleFrequencies()
			
			# calculate band-pass filter of fp1 signal
			df_epoc_fp1_filter = fft_signal.filter_result(df_epoc_fp1, edf.getSampleFrequencies()[0])
			
			# calculate band-pass filter of fp2 signal
			df_epoc_fp2_filter = fft_signal.filter_result(df_epoc_fp2, edf.getSampleFrequencies()[1])
			
			# calculate band-pass filter of a1 signal
			df_epoc_a1_filter = fft_signal.filter_result(df_epoc_a1, edf.getSampleFrequencies()[2])
			
			# calculate band-pass filter of a2 signal
			df_epoc_a2_filter = fft_signal.filter_result(df_epoc_a2, edf.getSampleFrequencies()[3])
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Processing to drop dataframe to csv file(filter data)\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if filter_csv_enable:
				# ---------------------< signal >---------------------------------------------------
				# Store Fp1 signal data to csv file.
				dir_name_fp1 = os.path.join(dir_name, 'Filter', 'Fp1')
				fft_folder.edf_to_csv(dir_name_fp1, file_name_without_extension, 'epoc_fp1_filter', df_epoc_fp1_filter)
				
				# Store Fp2 signal data to csv file.
				dir_name_fp2 = os.path.join(dir_name, 'Filter', 'Fp2')
				fft_folder.edf_to_csv(dir_name_fp2, file_name_without_extension, 'epoc_fp2_filter', df_epoc_fp2_filter)
				
				# Store A1 signal data to csv file.
				dir_name_a1 = os.path.join(dir_name, 'Filter', 'A1')
				fft_folder.edf_to_csv(dir_name_a1, file_name_without_extension, 'epoc_a1_filter', df_epoc_a1_filter)
				
				# Store A2 signal data to csv file.
				dir_name_a2 = os.path.join(dir_name, 'Filter', 'A2')
				fft_folder.edf_to_csv(dir_name_a2, file_name_without_extension, 'epoc_a2_filter', df_epoc_a2_filter)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Draw graphic of filter data\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if filter_graph_enable:
				save_path = os.path.join(project_root_dir, 'Image', 'Filter', 'Signal')
				fft_graph.raw_signal_plot(
					edf, time_signal,
					file_path,
					df_epoc_fp1_filter,
					df_epoc_fp2_filter,
					df_epoc_a1_filter,
					df_epoc_a2_filter,
					save_path
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Processing to calculate induction\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			
			data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2 = \
				fft_signal.induction_result(
					df_epoc_fp1_filter,
					df_epoc_fp2_filter,
					df_epoc_a1_filter,
					df_epoc_a2_filter,
					header_signal_str
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Processing to drop dataframe to csv file(induction data)\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if induction_csv_enable:
				# ---------------------< signal >---------------------------------------------------
				# Store Fp1_ave signal data to csv file.
				dir_name_fp1_ave = os.path.join(dir_name, 'Inductive', 'Fp1_Ave')
				fft_folder.edf_to_csv(
					dir_name_fp1_ave, file_name_without_extension, 'epoc_fp1_ave_inductive', data_frame_fp1_ave
				)
				
				# Store Fp2_ave signal data to csv file.
				dir_name_fp2_ave = os.path.join(dir_name, 'Inductive', 'Fp2_Ave')
				fft_folder.edf_to_csv(
					dir_name_fp2_ave, file_name_without_extension, 'epoc_fp2_ave_inductive', data_frame_fp2_ave
				)
				
				# Store Fp1_Fp2 signal data to csv file.
				dir_name_fp1_fp2 = os.path.join(dir_name, 'Inductive', 'Fp1_Fp2')
				fft_folder.edf_to_csv(
					dir_name_fp1_fp2, file_name_without_extension, 'epoc_fp1_fp2_inductive', data_frame_fp1_fp2
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Draw graphic of inductive data\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if inductive_graph_enable:
				save_path = os.path.join(project_root_dir, 'Image', 'Inductive', 'Signal')
				fft_graph.inductive_signal_plot(
					edf, time_signal,
					file_path,
					data_frame_fp1_ave,
					data_frame_fp2_ave,
					data_frame_fp1_fp2,
					save_path
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Confirmation of original distribution status\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			
			data_frame_fp1_ave.name = 'fp1_ave'
			data_frame_fp2_ave.name = 'fp2_ave'
			data_frame_fp1_fp2.name = 'fp1_fp2'
			
			data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2 = \
				fft_signal.noise_reduction(
					data_frame_fp1_ave, data_frame_fp2_ave, data_frame_fp1_fp2, parameter_iqr_signal
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Processing to drop dataframe to csv file(Outlier data)\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if outlier_csv_enable:
				data_outlier_fp1_ave_dir = os.path.join(project_root_dir, 'Data', 'Outlier', 'Fp1_Ave')
				# output csv file of data_frame_fp1_ave
				fft_folder.convert_csv(read_file_name[0], data_outlier_fp1_ave_dir, '_fp1_ave_outliers', data_frame_fp1_ave)
				
				data_outlier_fp2_ave_dir = os.path.join(project_root_dir, 'Data', 'Outlier', 'Fp2_Ave')
				# output csv file of data_frame_fp2_ave
				fft_folder.convert_csv(read_file_name[0], data_outlier_fp2_ave_dir, '_fp2_ave_outliers', data_frame_fp2_ave)
				
				data_outlier_fp1_fp2_dir = os.path.join(project_root_dir, 'Data', 'Outlier', 'Fp1_Fp2')
				# output csv file of data_frame_fp1_fp2
				fft_folder.convert_csv(read_file_name[0], data_outlier_fp1_fp2_dir, '_fp1_fp2_outliers', data_frame_fp1_fp2)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              Draw graphic of outliers data\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if outlier_graph_enable:
				save_path = os.path.join(project_root_dir, 'Image', 'Outlier', 'Signal')
				fft_graph.outliers_signal_plot(
					edf,
					time_signal,
					file_path,
					data_frame_fp1_ave,
					data_frame_fp2_ave,
					data_frame_fp1_fp2,
					save_path
				)
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              FFT Analysis\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			# Fs_signal = 4096  # frame size: 20.48sec * 200Hz = 4096
			Fs_signal = 800  # frame size: 4sec * 200Hz = 800
			Fs_resistance = 20  # frame size: 20sec * 1Hz = 20
			# overlap = 90  # overlap ratio
			overlap = 90  # overlap ratio
			
			# Calculation processing interval(sec)
			calculation_interval = int(30)
			
			# multi-process
			processes_fft = []
			
			# create arguments
			fft_arguments = [
				# fp1_ave
				[
					calculation_interval,
					project_root_dir,
					'Fp1_Ave',
					data_frame_fp1_ave,
					epoc_number_signal,
					frequency_ch[0],
					Fs_signal,
					overlap,
					hanning_csv_enable,
					fft_csv_enable,
					read_file_name
				],
				# fp2_ave
				[
					calculation_interval,
					project_root_dir,
					'Fp2_Ave',
					data_frame_fp2_ave,
					epoc_number_signal,
					frequency_ch[1],
					Fs_signal,
					overlap,
					hanning_csv_enable,
					fft_csv_enable,
					read_file_name
				],
				# fp1_fp2
				[
					calculation_interval,
					project_root_dir,
					'Fp1_Fp2',
					data_frame_fp1_fp2,
					epoc_number_signal,
					frequency_ch[0],
					Fs_signal,
					overlap,
					hanning_csv_enable,
					fft_csv_enable,
					read_file_name
				]
			]
			
			# calculate averaged FFT.
			for argument in fft_arguments:
				process = fft_signal.FftProcessor(argument)
				process.start()
				processes_fft.append(process)
			
			for p in processes_fft:
				p.join()
			
			t_signal = np.arange(0, Fs_signal) / frequency_ch[0]
			
			elapsed_time = time.time() - start
			print("elapsed_time_to_all_fft_processing_time:{0}".format(elapsed_time) + "[sec]")
			
			print(
				f"-----------------------------------------------------------------------------------\n"
				f"              FFT wave graphic\n"
				f"-----------------------------------------------------------------------------------\n"
			)
			if fft_graph_enable:
				# ----------------------< Fp1 - Ave >-------------------------------
				file_name_fp1_ave = f"{read_file_name[0]}_Fp1_Ave_fft_epoc.csv"
				fft_fp1_ave_file_name = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp1_Ave', file_name_fp1_ave)
				
				store_path = os.path.join(project_root_dir, 'Image', 'Fft', 'Fp1_Ave')
				
				fp1_ave_wave_graph_array = pd.read_csv(fft_fp1_ave_file_name)
				
				fft_graph.fft_wave_plot(file_name_fp1_ave, fp1_ave_wave_graph_array, store_path)
				
				# ----------------------< Fp2 - Ave >-------------------------------
				file_name_fp2_ave = f"{read_file_name[0]}_Fp2_Ave_fft_epoc.csv"
				fft_fp2_ave_file_name = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp2_Ave', file_name_fp2_ave)
				
				store_path = os.path.join(project_root_dir, 'Image', 'Fft', 'Fp2_Ave')
				
				fp2_ave_wave_graph_array = pd.read_csv(fft_fp2_ave_file_name)
				
				fft_graph.fft_wave_plot(file_name_fp2_ave, fp2_ave_wave_graph_array, store_path)
				
				# ----------------------< Fp1 - Fp2 >-------------------------------
				file_name_fp1_fp2 = f"{read_file_name[0]}_Fp1_Fp2_fft_epoc.csv"
				fft_fp1_fp2_file_name = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp1_Fp2', file_name_fp1_fp2)
				
				store_path = os.path.join(project_root_dir, 'Image', 'Fft', 'Fp1_Fp2')
				
				fp1_fp2_wave_graph_array = pd.read_csv(fft_fp1_fp2_file_name)
				
				fft_graph.fft_wave_plot(file_name_fp1_fp2, fp1_fp2_wave_graph_array, store_path)
			
			print(
				f"--------------------------------------------------------------------------------------\n"
				f"                 Prepare training data\n"
				f"--------------------------------------------------------------------------------------\n"
			)
			# Extract the csv file with the same name as the edf file.
			base_file_name = os.path.splitext(file_name_with_extension)
			
			target_csv_file = [file_name for file_name in csv_list if base_file_name[0] in file_name]
			
			print(target_csv_file[0])
			
			# Convert csv data of real stage to DataFrame.
			df_label = pd.read_csv(target_csv_file[0])
			
			# Convert csv data of fp1_ave to DataFrame.
			fp1_ave_file_name = f"{base_file_name[0]}_Fp1_Ave_fft_epoc.csv"
			fft_fp1_ave_path = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp1_Ave', fp1_ave_file_name)
			if not os.path.isfile(fft_fp1_ave_path):
				print(f"{fp1_ave_file_name}is not exist.")
			
			df_data_fft_fp1_ave = pd.read_csv(fft_fp1_ave_path)
			
			# Convert csv data of fp2_ave to DataFrame.
			fp2_ave_file_name = f"{base_file_name[0]}_Fp2_Ave_fft_epoc.csv"
			fft_fp2_ave_path = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp2_Ave', fp2_ave_file_name)
			if not os.path.isfile(fft_fp2_ave_path):
				print(f"{fp2_ave_file_name}is not exist.")
			
			df_data_fft_fp2_ave = pd.read_csv(fft_fp2_ave_path)
			
			# Convert csv data of fp1_ave to DataFrame.
			fp1_fp2_file_name = f"{base_file_name[0]}_Fp1_Fp2_fft_epoc.csv"
			fft_fp1_fp2_path = os.path.join(project_root_dir, 'Data', 'Fft', 'Fp1_Fp2', fp1_fp2_file_name)
			if not os.path.isfile(fft_fp1_fp2_path):
				print(f"{fp1_fp2_file_name}is not exist.")
			
			df_data_fft_fp1_fp2 = pd.read_csv(fft_fp1_fp2_path)
			
			# Swap rows and columns.
			df_data_fft_fp1_ave = df_data_fft_fp1_ave.transpose()
			df_data_fft_fp2_ave = df_data_fft_fp2_ave.transpose()
			df_data_fft_fp1_fp2 = df_data_fft_fp1_fp2.transpose()
			
			new_column = df_data_fft_fp1_ave.iloc[0].values.astype(str)
			new_column_fp1_ave = new_column.astype(object) + '_0'
			new_column_fp1_ave = new_column_fp1_ave.astype(str)
			
			new_column_fp2_ave = new_column.astype(object) + '_1'
			new_column_fp2_ave = new_column_fp2_ave.astype(str)
			
			new_column_fp1_fp2 = new_column.astype(object) + '_2'
			new_column_fp1_fp2 = new_column_fp1_fp2.astype(str)
			
			new_index = df_data_fft_fp1_ave.index.values.astype(str)
			new_index = np.chararray.replace(new_index, 'Fp1_Ave', 'fft')
			
			# Match the index name of the dataframe.
			df_data_fft_fp1_ave.set_axis(new_index.tolist(), axis='index', inplace=True)
			df_data_fft_fp1_ave.set_axis(new_column_fp1_ave, axis='columns', inplace=True)
			
			df_data_fft_fp2_ave.set_axis(new_index.tolist(), axis='index', inplace=True)
			df_data_fft_fp2_ave.set_axis(new_column_fp2_ave, axis='columns', inplace=True)
			
			df_data_fft_fp1_fp2.set_axis(new_index.tolist(), axis='index', inplace=True)
			df_data_fft_fp1_fp2.set_axis(new_column_fp1_fp2, axis='columns', inplace=True)
			
			# create dropping raw
			drop_index = list(range(0, df_data_fft_fp1_ave.shape[0], 2))
			remain_index = list(range(1, df_data_fft_fp1_ave.shape[0], 2))
			
			list_index = []
			for tmp_index in drop_index:
				list_index.append(df_data_fft_fp1_ave.index[tmp_index])
			
			list_index_fft = []
			for tmp_index in remain_index:
				list_index_fft.append(df_data_fft_fp1_ave.index[tmp_index])
			
			# Delete the odd lines of each signal, leaving only the fft data.
			df_data_fft_fp1_ave = df_data_fft_fp1_ave.drop(list_index)
			
			df_data_fft_fp2_ave = df_data_fft_fp2_ave.drop(list_index)
			
			df_data_fft_fp1_fp2 = df_data_fft_fp1_fp2.drop(list_index)
			
			# concat each dataframe.
			df_data_fft_total = pd.concat([df_data_fft_fp1_ave, df_data_fft_fp2_ave, df_data_fft_fp1_fp2], axis=1)
			
			print(
				f"--------------------------------------------------------------------------------------\n"
				f"                 Prepare labels\n"
				f"--------------------------------------------------------------------------------------\n"
			)
			# If the number of training data and the number of labels do not match, adjust to the smaller one.
			num_fft = df_data_fft_total.shape[0]
			num_label = df_label.shape[0]
			dif_delta = abs(num_fft - num_label)
			
			label_list = df_label.index.values.astype(str)
			
			if num_fft > num_label:
				remove_index = []
				for tmp in range(dif_delta):
					remove_index.append(list_index[-tmp])
					
				df_data_fft_total = df_data_fft_total.drop(remove_index, axis=0)
			else:
				for tmp in range(dif_delta):
					remove_label = len(label_list) - tmp - 1
					df_label = df_label.drop([remove_label], axis=0)
					df_label.set_axis(list_index_fft, axis='index', inplace=True)
			
			# create Create a dataframe for machine learning.
			df_tmp_datasets = pd.concat([df_data_fft_total, df_label], axis=1)
			
			# Delete rows and columns, including Nan in the Dataframe.
			df_tmp_datasets = df_tmp_datasets.dropna(how='any', axis=0)
			
			if df_datasets_first_flag:
				columns = df_tmp_datasets.columns.values
				df_datasets = df_datasets.append(df_tmp_datasets)
				df_datasets_first_flag = False
			else:
				df_datasets = df_datasets.append(df_tmp_datasets)
			
			print(
				f"----------------------------------------------------------------------------------------------\n"
				f"                        Convert df_datasets to csv file\n"
				f"----------------------------------------------------------------------------------------------\n"
			)
			
			# set file path
			data_sets_path = os.path.join(project_root_dir, 'Data', 'DataSets', 'fft_data_sets.csv')
			
			if first_process_label_flag:
				df_datasets.to_csv(data_sets_path)
				first_process_label_flag = False
			else:
				df_datasets.to_csv(data_sets_path, mode='a', header=False)
			
			gc.collect()
	else:
		print(
			f"----------------------------------------------------------------------------\n"
			f"                      Create training data and test data\n"
			f"----------------------------------------------------------------------------\n"
		)
		df_datasets = None
		
		csv_datasets_path = os.path.join(project_root_dir, 'Data', 'DataSets', 'fft_data_sets.csv')
		
		for tmp in pd.read_csv(csv_datasets_path, index_col=False, chunksize=100000):
			if df_datasets is None:
				df_datasets = tmp
			else:
				df_datasets = df_datasets.append(tmp)
			del tmp
			gc.collect()
		
		tmp_index = df_datasets[df_datasets.columns[0]]
		
		df_datasets = df_datasets.drop(df_datasets.columns[0], axis=1)
		
		df_datasets.set_axis(tmp_index.tolist(), axis='index', inplace=True)
	
	print(
		f"-----------------------------------------------------------------------------------------\n"
		f"                        Exclude NS stage.\n"
		f"-----------------------------------------------------------------------------------------\n"
	)
	
	df_datasets = df_datasets[df_datasets['睡眠ステージ'] != 'NS']
	
	print(
		f"-----------------------------------------------------------------------------------------\n"
		f"                        Create Training data and labels\n"
		f"-----------------------------------------------------------------------------------------\n"
	)
	# create training data
	x_train = df_datasets.drop(['エポック番号( E) ', '開始時刻 ', '睡眠ステージ'], axis=1)
	
	# Create label data
	label_data = df_datasets['睡眠ステージ']
	
	# Replace the string with an integer.
	label_data = label_data.replace(
		['WK', 'REM', 'N1', 'N2', 'N3', 'NS'],
		[0, 1, 2, 3, 4, 5]
	)
	
	# Replace the string with an integer.
	x_train, x_test, y_train, y_test = train_test_split(x_train, label_data, test_size=0.2, random_state=0)
	
	print(
		f"--------------------------------------------------------------------------------------------------\n"
		f"                       Preprocessing\n"
		f"--------------------------------------------------------------------------------------------------\n"
	)
	# Create an instance of MinMax scaling.
	mms = MinMaxScaler()
	
	x_train_std = mms.fit_transform(x_train)
	x_test_std = mms.fit_transform(x_test)
	
	print(
		f"--------------------------------------------------------------------------------------------------\n"
		f"                       training Machine(RandomForest)\n"
		f"--------------------------------------------------------------------------------------------------\n"
	)
	# prepare empty array.
	test_data_result = np.NaN
	predicting_labels = np.NaN
	predicting_probability = np.NaN
	test_target_result = np.NaN
	
	# Train with Random Forest Machine and output the expected label.
	test_data_real, test_predicting_labels, test_predicting_probability, test_real_reals = \
		fft_machine.random_forest_machine(search_mode, x_train_std, y_train, x_test_std, y_test)
	
	# Cross-validation score
	print(
		f"--------------------------------------------------------------------------------------------------\n"
		f"                       confusion matrix \n"
		f"--------------------------------------------------------------------------------------------------\n"
	)
	df_predicting_label = pd.DataFrame(test_predicting_labels)
	
	df_predicting_label = df_predicting_label.astype(str)
	df_predicting_label = df_predicting_label.replace('0', 'Wake')
	df_predicting_label = df_predicting_label.replace('1', 'REM')
	df_predicting_label = df_predicting_label.replace('2', 'NonREM1')
	df_predicting_label = df_predicting_label.replace('3', 'NonREM2')
	df_predicting_label = df_predicting_label.replace('4', 'NonREM3')
	
	test_real_reals = test_real_reals.astype(str)
	test_real_reals = test_real_reals.replace('0', 'Wake')
	test_real_reals = test_real_reals.replace('1', 'REM')
	test_real_reals = test_real_reals.replace('2', 'NonREM1')
	test_real_reals = test_real_reals.replace('3', 'NonREM2')
	test_real_reals = test_real_reals.replace('4', 'NonREM3')
	
	list_real_labels = test_real_reals.values.tolist()
	list_prediction_labels = df_predicting_label.values.tolist()
	data = confusion_matrix(list_real_labels, list_prediction_labels)
	
	df_cm = pd.DataFrame(data, columns=np.unique(list_real_labels), index=np.unique(list_real_labels))
	df_cm.index.name = 'Actual'
	df_cm.columns.name = 'Predicted'
	plt.figure(figsize=(10, 7))
	sn.set(font_scale=1.4)  # for label size
	sn.heatmap(df_cm/np.sum(df_cm), cmap="Blues", annot=True, annot_kws={"size": 16})
	
	plt.show()
	
	print("finish")