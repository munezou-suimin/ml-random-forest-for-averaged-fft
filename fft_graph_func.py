"""
-------------------------------------------------------------------
fft_graph_func.py

contents)
Function to graph the calculation result
-------------------------------------------------------------------^
"""
# python library
import os
import gc
import traceback

# matplotlib library
import matplotlib.pyplot as plt

from fft_siganl_func import stop_watch

import numpy as np

"""
--------------------------------------------------------------------
common setting
--------------------------------------------------------------------
"""

pause_time = 5

# Set the font type and size.
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

# Turn the scale inside.
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


@stop_watch
def raw_signal_plot(edf_file, time_value, file_name, df_fp1, df_fp2, df_a1, df_a2, save_dir):
	"""
	
	:param edf_file:
	:param time_value:
	:param file_name:
	:param df_fp1:
	:param df_fp2:
	:param df_a1:
	:param df_a2:
	:param save_dir:
	:return:
	"""

	print('---< start of signal >-----')
	fig1 = plt.figure(figsize=(20, 10))

	ax1 = fig1.add_subplot(221)
	ax1.yaxis.set_ticks_position('both')
	ax1.xaxis.set_ticks_position('both')
	ax2 = fig1.add_subplot(222)
	ax2.yaxis.set_ticks_position('both')
	ax2.xaxis.set_ticks_position('both')
	ax3 = fig1.add_subplot(223)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax4 = fig1.add_subplot(224)
	ax4.yaxis.set_ticks_position('both')
	ax4.xaxis.set_ticks_position('both')
	
	ax1.set_title(edf_file.getSignalHeader(0)['label'])
	ax2.set_title(edf_file.getSignalHeader(1)['label'])
	ax3.set_title(edf_file.getSignalHeader(2)['label'])
	ax4.set_title(edf_file.getSignalHeader(3)['label'])
	
	# Set the labels for the axes.
	ax1.set_xlabel('time[sec]')
	ax1.set_ylabel('volt[{0}]'.format(edf_file.getSignalHeader(0)['dimension']))
	ax2.set_xlabel('time[sec]')
	ax2.set_ylabel('volt[{0}]'.format(edf_file.getSignalHeader(1)['dimension']))
	ax3.set_xlabel('time[sec]')
	ax3.set_ylabel('volt[{0}]'.format(edf_file.getSignalHeader(2)['dimension']))
	ax4.set_xlabel('time[sec]')
	ax4.set_ylabel('volt[{0}]'.format(edf_file.getSignalHeader(3)['dimension']))
	
	ax1.grid(axis='both')
	ax2.grid(axis='both')
	ax3.grid(axis='both')
	ax4.grid(axis='both')
	
	for i_ax1 in range(0, df_fp1.shape[1], 2):
		try:
			if (i_ax1 + 1) < df_fp1.shape[1]:
				ax1.plot(time_value, df_fp1[df_fp1.columns[i_ax1 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax2 in range(0, df_fp2.shape[1], 2):
		try:
			if (i_ax2 + 1) < df_fp2.shape[1]:
				ax2.plot(time_value, df_fp2[df_fp2.columns[i_ax2 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax3 in range(0, df_a1.shape[1], 2):
		try:
			if (i_ax3 + 1) < df_a1.shape[1]:
				ax3.plot(time_value, df_a1[df_a1.columns[i_ax3 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax4 in range(0, df_a2.shape[1], 2):
		try:
			if (i_ax4 + 1) < df_a2.shape[1]:
				ax4.plot(time_value, df_a2[df_a2.columns[i_ax4 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	# Set up the layout.
	fig1.tight_layout()
	
	store_name = os.path.splitext(os.path.basename(file_name))
	store_name_png = store_name[0] + '_raw_signal.png'
	
	plt.savefig(os.path.join(save_dir, store_name_png))
	
	# plt.show()
	
	plt.show(block=False)
	
	plt.pause(pause_time)
	
	plt.close(fig1)
	
	print('---< end of signal >-----\n\n')
	# ---------------------------------< end of signal >----------------------------------------------------------
	
	gc.collect()


@stop_watch
def raw_resistance_plot(
		edf,
		time_resistance,
		file_name,
		df_epoc_r_fp1,
		df_epoc_r_fp2,
		df_epoc_r_ref,
		df_epoc_r_a1,
		df_epoc_r_a2,
		save_dir
):
	"""
	
	:param edf:
	:param time_resistance:
	:param file_name:
	:param df_epoc_r_fp1:
	:param df_epoc_r_fp2:
	:param df_epoc_r_ref:
	:param df_epoc_r_a1:
	:param df_epoc_r_a2:
	:param save_dir:
	:return:
	"""
	print('---< start of resistance >-----')
	fig2 = plt.figure(figsize=(12, 10))
	
	ax5 = fig2.add_subplot(321)
	ax5.yaxis.set_ticks_position('both')
	ax5.xaxis.set_ticks_position('both')
	ax6 = fig2.add_subplot(322)
	ax6.yaxis.set_ticks_position('both')
	ax6.xaxis.set_ticks_position('both')
	ax7 = fig2.add_subplot(323)
	ax7.yaxis.set_ticks_position('both')
	ax7.xaxis.set_ticks_position('both')
	ax8 = fig2.add_subplot(324)
	ax8.yaxis.set_ticks_position('both')
	ax8.xaxis.set_ticks_position('both')
	ax9 = fig2.add_subplot(325)
	ax9.yaxis.set_ticks_position('both')
	ax9.xaxis.set_ticks_position('both')
	ax10 = fig2.add_subplot(326)
	ax10.yaxis.set_ticks_position('both')
	ax10.xaxis.set_ticks_position('both')
	
	ax5.set_title(edf.getSignalHeader(5)['label'])
	ax6.set_title(edf.getSignalHeader(6)['label'])
	ax7.set_title(edf.getSignalHeader(7)['label'])
	ax9.set_title(edf.getSignalHeader(8)['label'])
	ax10.set_title(edf.getSignalHeader(9)['label'])
	
	# Set the labels for the axes.
	ax5.set_xlabel('time[sec]')
	ax5.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(5)['dimension']))
	ax6.set_xlabel('time[sec]')
	ax6.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(6)['dimension']))
	ax7.set_xlabel('time[sec]')
	ax7.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(7)['dimension']))
	ax9.set_xlabel('time[sec]')
	ax9.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(8)['dimension']))
	ax10.set_xlabel('time[sec]')
	ax10.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(9)['dimension']))
	
	# set grid
	ax5.grid(axis='both')
	ax6.grid(axis='both')
	ax7.grid(axis='both')
	ax8.grid(axis='both')
	ax9.grid(axis='both')
	ax10.grid(axis='both')
	
	for i_ax5 in range(0, df_epoc_r_a1.shape[1], 2):
		try:
			if (i_ax5 + 1) < df_epoc_r_a1.shape[1]:
				ax5.plot(time_resistance, df_epoc_r_a1[df_epoc_r_a1.columns[i_ax5 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
		
	for i_ax6 in range(0, df_epoc_r_fp1.shape[1], 2):
		try:
			if (i_ax6 + 1) < df_epoc_r_fp1.shape[1]:
				ax6.plot(time_resistance, df_epoc_r_fp1[df_epoc_r_fp1.columns[i_ax6 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax7 in range(0, df_epoc_r_ref.shape[1], 2):
		try:
			if (i_ax7 + 1) < df_epoc_r_ref.shape[1]:
				ax7.plot(time_resistance, df_epoc_r_ref[df_epoc_r_ref.columns[i_ax7 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax9 in range(0, df_epoc_r_fp2.shape[1], 2):
		try:
			if (i_ax9 + 1) < df_epoc_r_fp2.shape[1]:
				ax9.plot(time_resistance, df_epoc_r_fp2[df_epoc_r_fp2.columns[i_ax9 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax10 in range(0, df_epoc_r_a2.shape[1], 2):
		try:
			if (i_ax10 + 1) < df_epoc_r_a2.shape[1]:
				ax10.plot(time_resistance, df_epoc_r_a2[df_epoc_r_a2.columns[i_ax10 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	# Set up the layout.
	fig2.tight_layout()
	
	store_name = os.path.splitext(os.path.basename(file_name))
	store_name_png = store_name[0] + '_raw_Resistance.png'
	
	plt.savefig(os.path.join(save_dir, store_name_png))
	
	# plt.show()
	
	plt.show(block=False)
	
	plt.pause(pause_time)
	
	plt.close(fig2)
	
	print('---< end of resistance >-----\n\n')
	# ---------------------------------< end of resistance A1 >---------------------------------------------------
	
	gc.collect()


@stop_watch
def raw_voltage_plot(
		edf, time_voltage,
		file_name,
		df_epoc_v_fp1,
		df_epoc_v_fp2,
		df_epoc_v_a1,
		df_epoc_v_a2,
		df_epoc_v_battery,
		save_dir
):
	
	"""
	
	:param edf:
	:param time_voltage:
	:param file_name:
	:param df_epoc_v_fp1:
	:param df_epoc_v_fp2:
	:param df_epoc_v_a1:
	:param df_epoc_v_a2:
	:param df_epoc_v_battery:
	:param save_dir:
	:return:
	"""
	# ---------------------------------< start of voltage >-------------------------------------------------
	print('---< start of voltage >-----')
	fig3 = plt.figure(figsize=(12, 10))
	
	ax11 = fig3.add_subplot(321)
	ax11.yaxis.set_ticks_position('both')
	ax11.xaxis.set_ticks_position('both')
	ax12 = fig3.add_subplot(322)
	ax12.yaxis.set_ticks_position('both')
	ax12.xaxis.set_ticks_position('both')
	ax13 = fig3.add_subplot(323)
	ax13.yaxis.set_ticks_position('both')
	ax13.xaxis.set_ticks_position('both')
	ax14 = fig3.add_subplot(324)
	ax14.yaxis.set_ticks_position('both')
	ax14.xaxis.set_ticks_position('both')
	ax15 = fig3.add_subplot(325)
	ax15.yaxis.set_ticks_position('both')
	ax15.xaxis.set_ticks_position('both')
	ax16 = fig3.add_subplot(326)
	ax16.yaxis.set_ticks_position('both')
	ax16.xaxis.set_ticks_position('both')
	
	ax11.set_title(edf.getSignalHeader(10)['label'])
	ax12.set_title(edf.getSignalHeader(11)['label'])
	ax13.set_title(edf.getSignalHeader(12)['label'])
	ax14.set_title(edf.getSignalHeader(13)['label'])
	ax15.set_title(edf.getSignalHeader(14)['label'])
	
	# Set the labels for the axes.
	ax11.set_xlabel('time[sec]')
	ax11.set_ylabel('voltage[{0}]'.format(edf.getSignalHeader(10)['dimension']))
	ax12.set_xlabel('time[sec]')
	ax12.set_ylabel('voltage[{0}]'.format(edf.getSignalHeader(11)['dimension']))
	ax13.set_xlabel('time[sec]')
	ax13.set_ylabel('voltage[{0}]'.format(edf.getSignalHeader(12)['dimension']))
	ax14.set_xlabel('time[sec]')
	ax14.set_ylabel('voltage[{0}]'.format(edf.getSignalHeader(13)['dimension']))
	ax15.set_xlabel('time[sec]')
	ax15.set_ylabel('resistance[{0}]'.format(edf.getSignalHeader(14)['dimension']))
	
	# set grid
	ax11.grid(axis='both')
	ax12.grid(axis='both')
	ax13.grid(axis='both')
	ax14.grid(axis='both')
	ax15.grid(axis='both')
	ax16.grid(axis='both')
	
	for i_ax11 in range(0, df_epoc_v_a1.shape[1], 2):
		try:
			if (i_ax11 + 1) < df_epoc_v_a1.shape[1]:
				ax11.plot(time_voltage, df_epoc_v_a1[df_epoc_v_a1.columns[i_ax11 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax12 in range(0, df_epoc_v_fp1.shape[1], 2):
		try:
			if (i_ax12 + 1) < df_epoc_v_fp1.shape[1]:
				ax11.plot(time_voltage, df_epoc_v_fp1[df_epoc_v_fp1.columns[i_ax12 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax13 in range(0, df_epoc_v_fp2.shape[1], 2):
		try:
			if (i_ax13 + 1) < df_epoc_v_fp2.shape[1]:
				ax13.plot(time_voltage, df_epoc_v_fp2[df_epoc_v_fp2.columns[i_ax13 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax14 in range(0, df_epoc_v_a2.shape[1], 2):
		try:
			if (i_ax14 + 1) < df_epoc_v_a2.shape[1]:
				ax14.plot(time_voltage, df_epoc_v_a2[df_epoc_v_a2.columns[i_ax14 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax15 in range(0, df_epoc_v_battery.shape[1], 2):
		try:
			if (i_ax15 + 1) < df_epoc_v_battery.shape[1]:
				ax15.plot(time_voltage, df_epoc_v_battery[df_epoc_v_battery.columns[i_ax15 + 1]])
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	# Set up the layout.
	fig3.tight_layout()
	
	store_name = os.path.splitext(os.path.basename(file_name))
	store_name_png = store_name[0] + '_raw_Voltage.png'
	
	plt.savefig(os.path.join(save_dir, store_name_png))
	
	# plt.show()
	
	plt.show(block=False)
	
	plt.pause(pause_time)
	
	plt.close(fig3)
	
	print('---< end of voltage >-----\n\n')
	# ---------------------------------< end of voltage >---------------------------------------------------
	
	gc.collect()


@stop_watch
def inductive_signal_plot(
		edf, time_signal,
		file_name,
		data_frame_fp1_ave,
		data_frame_fp2_ave,
		data_frame_fp1_fp2,
		save_dir
):

	print('---< start of drawing inductive signal >-----')
	fig4 = plt.figure(figsize=(20, 10))
	
	ax16 = fig4.add_subplot(221)
	ax16.yaxis.set_ticks_position('both')
	ax16.xaxis.set_ticks_position('both')
	ax17 = fig4.add_subplot(222)
	ax17.yaxis.set_ticks_position('both')
	ax17.xaxis.set_ticks_position('both')
	ax18 = fig4.add_subplot(223)
	ax18.yaxis.set_ticks_position('both')
	ax18.xaxis.set_ticks_position('both')
	
	ax16.set_title('Fp1_Ave')
	ax17.set_title('FP2_Ave')
	ax18.set_title('FP1_Fp2')
	
	# Set the labels for the axes.
	ax16.set_xlabel('time[sec]')
	ax16.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(0)['dimension']))
	ax17.set_xlabel('time[sec]')
	ax17.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(1)['dimension']))
	ax18.set_xlabel('time[sec]')
	ax18.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(0)['dimension']))
	
	ax16.grid(axis='both')
	ax17.grid(axis='both')
	ax18.grid(axis='both')
	
	for i_ax16 in range(0, data_frame_fp1_ave.shape[1], 2):
		try:
			if (i_ax16 + 1) < data_frame_fp1_ave.shape[1]:
				ax16.plot(
					time_signal,
					data_frame_fp1_ave[data_frame_fp1_ave.columns[i_ax16 + 1]]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax17 in range(0, data_frame_fp2_ave.shape[1], 2):
		try:
			if (i_ax17 + 1) < data_frame_fp2_ave.shape[1]:
				ax17.plot(
					time_signal,
					data_frame_fp2_ave[data_frame_fp2_ave.columns[i_ax17 + 1]]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax18 in range(0, data_frame_fp1_fp2.shape[1], 2):
		try:
			if (i_ax18 + 1) < data_frame_fp1_fp2.shape[1]:
				ax18.plot(
					time_signal,
					data_frame_fp1_fp2[data_frame_fp1_fp2.columns[i_ax18 + 1]]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	# Set up the layout.
	fig4.tight_layout()
	
	store_name = os.path.splitext(os.path.basename(file_name))
	store_name_png = store_name[0] + '_induction_signal.png'
	
	plt.savefig(os.path.join(save_dir, store_name_png))
	
	# plt.show()
	
	plt.show(block=False)
	
	plt.pause(pause_time)
	
	plt.close(fig4)
	
	print('---< end of signal >-----\n\n')
	# ---------------------------------< end of signal >----------------------------------------------------------
	
	gc.collect()


@stop_watch
def outliers_signal_plot(
		edf,
		time_signal,
		file_name,
		data_frame_fp1_ave,
		data_frame_fp2_ave,
		data_frame_fp1_fp2,
		save_dir
):
	"""
	
	:param edf:
	:param time_signal:
	:param file_name:
	:param data_frame_fp1_ave:
	:param data_frame_fp2_ave:
	:param data_frame_fp1_fp2:
	:param save_dir:
	:return:
	"""
	# ---------------------------------< start of outliers process >---------------------------------------------------
	print('---< start of drawing outliers data at NS stage >-----')
	fig5 = plt.figure(figsize=(20, 10))
	
	ax19 = fig5.add_subplot(221)
	ax19.yaxis.set_ticks_position('both')
	ax19.xaxis.set_ticks_position('both')
	ax20 = fig5.add_subplot(222)
	ax20.yaxis.set_ticks_position('both')
	ax20.xaxis.set_ticks_position('both')
	ax21 = fig5.add_subplot(223)
	ax21.yaxis.set_ticks_position('both')
	ax21.xaxis.set_ticks_position('both')
	
	ax19.set_title('Fp1_Ave_NS')
	ax20.set_title('FP2_Ave_NS')
	ax21.set_title('FP1_Fp2_NS')
	
	# Set the labels for the axes.
	ax19.set_xlabel('time[sec]')
	ax19.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(0)['dimension']))
	ax20.set_xlabel('time[sec]')
	ax20.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(1)['dimension']))
	ax21.set_xlabel('time[sec]')
	ax21.set_ylabel('volt[{0}]'.format(edf.getSignalHeader(0)['dimension']))
	
	# x-axis range
	ax19.set_xlim(0, 30)
	# y-axis range
	ax19.set_ylim(-150, 150)
	
	# x-axis range
	ax20.set_xlim(0, 30)
	# y-axis range
	ax20.set_ylim(-150, 150)
	
	# x-axis range
	ax21.set_xlim(0, 30)
	# y-axis range
	ax21.set_ylim(-150, 150)
	
	ax19.grid(axis='both')
	ax20.grid(axis='both')
	ax21.grid(axis='both')
	
	for i_ax19 in range(0, data_frame_fp1_ave.shape[1], 2):
		try:
			if (i_ax19 + 1) < data_frame_fp1_ave.shape[1]:
				tmp_column_name = data_frame_fp1_ave.columns[i_ax19 + 1]
				ax19.plot(
					time_signal,
					data_frame_fp1_ave[tmp_column_name]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax20 in range(0, data_frame_fp2_ave.shape[1], 2):
		try:
			if (i_ax20 + 1) < data_frame_fp2_ave.shape[1]:
				tmp_column_name = data_frame_fp2_ave.columns[i_ax20 + 1]
				ax20.plot(
					time_signal,
					data_frame_fp2_ave[tmp_column_name]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	for i_ax21 in range(0, data_frame_fp1_fp2.shape[1], 2):
		try:
			if (i_ax21 + 1) < data_frame_fp1_fp2.shape[1]:
				tmp_column_name = data_frame_fp1_fp2.columns[i_ax21 + 1]
				ax21.plot(
					time_signal,
					data_frame_fp1_fp2[tmp_column_name]
				)
		except Exception as ex:
			print(ex)
			pass
		finally:
			pass
	
	# Set up the layout.
	fig5.tight_layout()
	
	store_name = os.path.splitext(os.path.basename(file_name))
	store_name_png = store_name[0] + '_outliers.png'
	
	plt.savefig(os.path.join(save_dir, store_name_png))
	
	# plt.show()
	
	plt.show(block=False)
	
	plt.pause(pause_time)
	
	plt.close(fig5)
	
	print("---< end of outliers process >-----\n\n")
	# -------------------------------< end of outliers process >---------------------------------
	
	gc.collect()


def fft_wave_plot(
		file_path,
		data_frame,
		save_path
):
	"""

	:param file_path:
	:param data_frame:
	:param save_path:
	:return:
	"""
	try:
		file_name_base = file_path.split('.')
		
		# ------------------------------< Start of displaying fft signal in REM stage >---------------------------------
		fig7 = plt.figure(figsize=(20, 10))
		ax27 = fig7.add_subplot(111)
		ax27.yaxis.set_ticks_position('both')
		ax27.xaxis.set_ticks_position('both')
		
		ax27.set_title(file_name_base[0])
		
		ax27.set_xlabel('Frequency [Hz]')
		ax27.set_ylabel('Ampl[mVolt]')
		
		# Set the scale.
		ax27.set_xticks(np.arange(0, 70, 1))
		ax27.set_xlim(0, 70)
		
		ax27.grid(axis='both')
		
		# Prepare the data plot, set the label, line thickness, and legend.
		# fp1_Ave
		for i_ax27 in range(0, data_frame.shape[1], 2):
			try:
				if (i_ax27 + 1) < data_frame.shape[1]:
					tmp_column_name_freq = data_frame.columns[i_ax27]
					tmp_column_name_amplitude = data_frame.columns[i_ax27 + 1]
					ax27.plot(
						data_frame[tmp_column_name_freq],
						data_frame[tmp_column_name_amplitude],
						label=tmp_column_name_amplitude,
						lw=1
					)
			except Exception as ex:
				print(ex)
				pass
			finally:
				pass
		
		# Set up the layout.
		fig7.tight_layout()
		
		store_name = os.path.splitext(os.path.basename(file_path))
		store_name = store_name[0] + '.png'
		
		plt.savefig(os.path.join(save_path, store_name))
		
		# Display the graph.
		# plt.show()
		plt.show(block=False)
		
		plt.pause(pause_time)
		
		plt.close(fig7)
		
		# -----------------------------< End of displaying fft signal in REM stage >-----------------------------------
	except:
		print("Error information(create_table)\n" + traceback.format_exc())
		pass
	
	
	gc.collect()