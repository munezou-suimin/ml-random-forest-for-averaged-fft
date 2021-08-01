"""
-------------------------------------------------------------------
fft_folder_func.py

contents)
Define Folder and csv related functions to be used in fft_analyze_main.
-------------------------------------------------------------------^
"""
# python library
import os
import hashlib
import traceback

from fft_siganl_func import stop_watch


def delete_duplicate_edf(file_list):
	"""
	Delete duplicate files
	:param file_list:
	:return:
	"""
	hash_list = {}
	
	for file_path in file_list:
		data = open(file_path, 'rb').read()
		
		# Compute the hash of the file.
		h = hashlib.sha256(data).hexdigest()
		
		if h in hash_list:
			if data == open(hash_list[h], 'rb').read():
				print(hash_list[h] + 'と' + file_path + 'は同じ')
				os.remove(file_path)  # remove duplicated file
		else:
			hash_list[h] = file_path
	
	return hash_list


def confirm_folder(path):
	"""
	Create a folder if specified folder was not existed.
	:param path:  using file path
	:return:
	"""
	try:
		if not os.path.isdir(path):
			os.mkdir(path)
	except:
		print(f"Error: {traceback.print_exc()}")
		pass
	finally:
		pass
	
	
@stop_watch
def create_folder(current_path):
	"""
	Create required folders
	
	:param current_path:
	:return:
	"""
	# -----------------< Create Raw data folder >------------------------------
	data_raw_dir = os.path.join(current_path, 'Data', 'Raw_Data')
	confirm_folder(data_raw_dir)
	
	data_raw_ns_dir = os.path.join(data_raw_dir, 'Fp1')
	confirm_folder(data_raw_ns_dir)
	
	data_raw_wk_dir = os.path.join(data_raw_dir, 'Fp2')
	confirm_folder(data_raw_wk_dir)
	
	data_raw_rem_dir = os.path.join(data_raw_dir, 'A1')
	confirm_folder(data_raw_rem_dir)
	
	data_raw_n1_dir = os.path.join(data_raw_dir, 'A2')
	confirm_folder(data_raw_n1_dir)
	
	data_raw_resistance_dir = os.path.join(data_raw_dir, 'Resistance')
	confirm_folder(data_raw_resistance_dir)
	
	data_raw_voltage_dir = os.path.join(data_raw_dir, 'Voltage')
	confirm_folder(data_raw_voltage_dir)
	
	# -------------------------< filter >-----------------------------------
	data_filter_dir = os.path.join(current_path, 'Data', 'Filter')
	confirm_folder(data_filter_dir)
	
	data_filter_ns_dir = os.path.join(data_filter_dir, 'Fp1')
	confirm_folder(data_filter_ns_dir)
	
	data_filter_wk_dir = os.path.join(data_filter_dir, 'Fp2')
	confirm_folder(data_filter_wk_dir)
	
	data_filter_rem_dir = os.path.join(data_filter_dir, 'A1')
	confirm_folder(data_filter_rem_dir)
	
	data_filter_n1_dir = os.path.join(data_filter_dir, 'A2')
	confirm_folder(data_filter_n1_dir)
	
	data_filter_n2_dir = os.path.join(data_filter_dir, 'Resistance')
	confirm_folder(data_filter_n2_dir)
	
	data_filter_n3_dir = os.path.join(data_filter_dir, 'Voltage')
	confirm_folder(data_filter_n3_dir)
	
	# --------------------< Inductive >-------------------------------------
	data_inductive_dir = os.path.join(current_path, 'Data', 'Inductive')
	confirm_folder(data_inductive_dir)
	
	data_inductive_ns_dir = os.path.join(data_inductive_dir, 'Fp1_Ave')
	confirm_folder(data_inductive_ns_dir)
	
	data_inductive_wk_dir = os.path.join(data_inductive_dir, 'FP2_Ave')
	confirm_folder(data_inductive_wk_dir)
	
	data_inductive_rem_dir = os.path.join(data_inductive_dir, 'Fp1_Fp2')
	confirm_folder(data_inductive_rem_dir)
	
	# -------------------------< Outlier >--------------------------------
	data_outlier_dir = os.path.join(current_path, 'Data', 'Outlier')
	confirm_folder(data_outlier_dir)
	
	data_outlier_ns_dir = os.path.join(data_outlier_dir, 'Fp1_Ave')
	confirm_folder(data_outlier_ns_dir)
	
	data_outlier_wk_dir = os.path.join(data_outlier_dir, 'Fp2_Ave')
	confirm_folder(data_outlier_wk_dir)
	
	data_outlier_rem_dir = os.path.join(data_outlier_dir, 'Fp1_Fp2')
	confirm_folder(data_outlier_rem_dir)
	
	data_outlier_n2_dir = os.path.join(data_outlier_dir, 'Resistance')
	confirm_folder(data_outlier_n2_dir)
	
	data_outlier_n3_dir = os.path.join(data_outlier_dir, 'Voltage')
	confirm_folder(data_outlier_n3_dir)
	
	# ------------------------< Hanning window >------------------------
	data_hanning_dir = os.path.join(current_path, 'Data', 'Hanning')
	confirm_folder(data_hanning_dir)
	
	data_hanning_fp1_ave_dir = os.path.join(data_hanning_dir, 'Fp1_Ave')
	confirm_folder(data_hanning_fp1_ave_dir)
	
	data_hanning_fp2_ave_dir = os.path.join(data_hanning_dir, 'Fp2_Ave')
	confirm_folder(data_hanning_fp2_ave_dir)
	
	data_hanning_fp1_fp2_dir = os.path.join(data_hanning_dir, 'Fp1_Fp2')
	confirm_folder(data_hanning_fp1_fp2_dir)
	
	# --------------------------< FFT >--------------------
	data_fft_dir = os.path.join(current_path, 'Data', 'Fft')
	confirm_folder(data_fft_dir)
	
	data_fft_fp1_ave = os.path.join(data_fft_dir, 'Fp1_Ave')
	confirm_folder(data_fft_fp1_ave)
	
	data_fft_fp2_ave = os.path.join(data_fft_dir, 'Fp2_Ave')
	confirm_folder(data_fft_fp2_ave)
	
	data_fft_fp1_fp2 = os.path.join(data_fft_dir, 'Fp1_Fp2')
	confirm_folder(data_fft_fp1_fp2)
	
	# ----------------< Image Folder >--------------------
	image_dir = os.path.join(current_path, 'Image')
	confirm_folder(image_dir)
	
	image_raw_dir = os.path.join(image_dir, 'Raw_Data')
	confirm_folder(image_raw_dir)
	
	image_filter_dir = os.path.join(image_dir, 'Filter')
	confirm_folder(image_filter_dir)
	
	image_inductive_dir = os.path.join(image_dir, 'Inductive')
	confirm_folder(image_inductive_dir)
	
	image_outlier_dir = os.path.join(image_dir, 'Outlier')
	confirm_folder(image_outlier_dir)
	
	image_fft_dir = os.path.join(image_dir, 'Fft')
	confirm_folder(image_fft_dir)
	
	# ------------------< Image Raw >----------------------
	image_raw_signal_dir = os.path.join(image_raw_dir, 'Signal')
	confirm_folder(image_raw_signal_dir)
	
	image_raw_resistance_dir = os.path.join(image_raw_dir, 'Resistance')
	confirm_folder(image_raw_resistance_dir)
	
	image_raw_voltage_dir = os.path.join(image_raw_dir, 'Voltage')
	confirm_folder(image_raw_voltage_dir)
	
	# ------------------< Filter >--------------------------
	
	image_filter_signal_dir = os.path.join(image_filter_dir, 'Signal')
	confirm_folder(image_filter_signal_dir)
	
	# ---------------------< Inductive >------------------------------
	image_inductive_signal_dir = os.path.join(image_inductive_dir, 'Signal')
	confirm_folder(image_inductive_signal_dir)
	
	# -------------------------< outlier >--------------------------
	image_outlier_signal_dir = os.path.join(image_outlier_dir, 'Signal')
	confirm_folder(image_outlier_signal_dir)
	
	image_outlier_resistance_dir = os.path.join(image_outlier_dir, 'Resistance')
	confirm_folder(image_outlier_resistance_dir)
	
	image_outlier_voltage_dir = os.path.join(image_outlier_dir, 'Voltage')
	confirm_folder(image_outlier_voltage_dir)
	
	# -----------------------< FFT >----------------------------
	image_fft_fp1_ave = os.path.join(image_fft_dir, 'Fp1_Ave')
	confirm_folder(image_fft_fp1_ave)
	
	image_fft_fp2_ave = os.path.join(image_fft_dir, 'Fp2_Ave')
	confirm_folder(image_fft_fp2_ave)
	
	image_fft_fp1_fp2 = os.path.join(image_fft_dir, 'Fp1_Fp2')
	confirm_folder(image_fft_fp1_fp2)
	
	image_fft_resistance_dir = os.path.join(image_fft_dir, 'Resistance')
	confirm_folder(image_fft_resistance_dir)
	
	image_fft_voltage_dir = os.path.join(image_fft_dir, 'Voltage')
	confirm_folder(image_fft_voltage_dir)


@stop_watch
def edf_to_csv(dir_name, file_name, signal_name, data_frame):
	"""
	convert def file to csv file.
	
	:param dir_name: directory name
	:param file_name:  title
	:param signal_name: signal_name
	:param data_frame: DataFrame
	:return:
	"""
	
	try:
		# create file name
		file_name = file_name + '_' + signal_name + '.csv'
		
		# create path name
		file_path_name = os.path.join(dir_name, file_name)
		
		# create csv file from dataframe
		data_frame.to_csv(file_path_name)
	except Exception as ex:
		print(ex)
		pass


@stop_watch
def convert_csv(basic_name, dir_name, stage_name, data_frame):
	"""
	Convert the calculation result to csv file.

	:param basic_name:
	:param dir_name:
	:param stage_name:
	:param data_frame:
	:return:
	"""
	try:
		# create file name
		file_epoc_name = basic_name + stage_name + '_epoc.csv'
		
		# create path name
		file_path_epoc_name = os.path.join(dir_name, file_epoc_name)
		
		# create csv file from dataframe
		data_frame.to_csv(file_path_epoc_name, index=False)
	except Exception as ex:
		print(ex)
		pass
	finally:
		pass