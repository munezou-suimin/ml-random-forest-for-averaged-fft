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
import multiprocessing

# numpy library
import numpy as np

# pandas library
import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier


n_cpus = multiprocessing.cpu_count()

# Definition of high parameters
search_params = {
	'n_estimators': [5, 10, 20, 30, 50, 100, 300],
	'max_features': [3, 5, 10, 15, 20],
	'random_state': [2525],
	'n_jobs': [1],
	'min_samples_split': [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
	'max_depth': [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}


def random_forest_machine(mode, train_x, train_y, test_x, test_y):
	"""

	:param train_x: training data
	:param train_y: label of training data
	:param test_x: test data
	:param test_y: label of test data
	:return: test_x, gs.predict(test_x), gs.predict_proba(test_x), test_y
	"""
	
	if mode:
		# create ai machine model
		gs = GridSearchCV(
			RandomForestClassifier(),
			search_params,
			refit=True,
			cv=3,
			verbose=True
		)
		
		# Do model learning.
		gs.fit(train_x, train_y)
		
		# Confirmation of correct answer rate
		print('best_score = {0}%\n'.format(gs.best_score_ * 100))
		
		print('acc = {0}%\n'.format(gs.score(test_x, test_y) * 100))
		
		best_param = gs.best_params_
		print('best_param = \n{0}\n'.format(best_param))
	else:
		gs = RandomForestClassifier(
			max_depth=10,
			max_features=15,
			min_samples_split=5,
			n_estimators=20,
			n_jobs=n_cpus,
			verbose=True,
			random_state=2525
		)
		
		gs.fit(train_x, train_y)
		
		print('acc = {0}%\n'.format(gs.score(test_x, test_y) * 100))
	
	return test_x, gs.predict(test_x), gs.predict_proba(test_x), test_y



