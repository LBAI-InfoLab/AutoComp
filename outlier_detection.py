

##
## part of AutoComp
## play with MFI
##



##
## TODO : function to evaluate a single fcs file
##


def create_outlier_detection_training_datafile(data_folder):
	"""
	IN PROGRESS

	Create a training data file from MFI informations
	files present in the data_folder

	Each line correspond to a file.

	TODO : flag if the file is analysable
	"""


	## importation
	import glob

	## initialise training data file
	training_data = open("MFI_training_data.csv", "w")
	mfi_files_list = glob.glob(data_folder+"/*.txt")
	
	## create the header
	cmpt = 0
	horizontal_label = []
	vertical_label = []
	header = "analysable,compensated,"
	mfi_file = open(mfi_files_list[0], "r")
	for line in mfi_file:
		if(cmpt == 0):
			line = line.rstrip()
			line = line.replace("\"", "")
			line_in_array = line.split(",")
			line_in_array = line_in_array[1:]
			for elt in line_in_array:
				horizontal_label.append(elt)
		else:
			line = line.rstrip()
			line = line.replace("\"", "")
			line_in_array = line.split(",")
			vertical_label.append(line_in_array[0])
		cmpt += 1
	
	for elt_2 in vertical_label:
		for elt_1 in horizontal_label:
			header += str(elt_1)+"-"+str(elt_2) +","
	

	mfi_file.close()
	training_data.write(header+"\n")


	## loop over the files in the data_folder
	cmpt = 0
	for mfi_file in mfi_files_list:

		mfi_file_name = mfi_file.split("/")
		mfi_file_name = mfi_file_name[-1].split("_")
		mfi_data = open(mfi_file, "r")


		## TODO
		## determine if the file is an outlier
		analysable = "NA"
		if("bad" in mfi_file_name):
			analysable = 0
		else:
			analysable = 1

		## determine if the file is compensated
		compensated = "NA"
		if("compensated.txt" == mfi_file_name[-1]):
			compensated = 1
		else:
			compensated = 0

		line_to_write = str(analysable)+","+str(compensated)+","


		mfi_cmpt = 0			
		for line in mfi_data:
			if(mfi_cmpt != 0):
					
				line = line.rstrip()
				line = line.replace("\"", "")
				line_in_array = line.split(",")
				line_in_array = line_in_array[1:]
				
				for elt in line_in_array:
					line_to_write += str(elt) +","

			mfi_cmpt += 1

		line_to_write = line_to_write[:-1]		
		if(cmpt == len(mfi_files_list)):
			training_data.write(line_to_write)
		else:
			training_data.write(line_to_write+"\n")

		mfi_data.close()		
		cmpt += 1

	## close training data file
	training_data.close()




def extract_MFI_informations(data_folder):
	"""
	Extract MFI information for all the fcs files
	in the data folder.

	raise a warnings if multiple centers or/and panels
	are found in the data folder

	Extraction is perfromed by a R script.
	"""

	## importation
	import glob
	import os

	## parameters
	panel_list = []
	center_list = []
	MFIEXTRACTION_SCRIPT = "ExtractMFIinformation.R"

	for fcs_file in glob.glob(data_folder+"/*.fcs"):

		fcs_file_name = fcs_file.split("/")
		fcs_file_name = fcs_file_name[-1].split("_")

		panel = fcs_file_name[1]
		center = fcs_file_name[3]

		## control the number of panel
		## and centers in data file
		if(panel not in panel_list):
			panel_list.append(panel)
		if(center not in center_list):
			center_list.append(center)

		## Perfrom Extraction
		os.system("Rscript "+MFIEXTRACTION_SCRIPT+" "+str(fcs_file))

	## raise warnings if needed
	if(len(panel_list) > 1):
		print "[WARNINGS] => Mix of panels used for MFI training file generation"
	if(len(center_list) > 1):
		print "[WARNINGS] => Mix of centers used for MFI training file generation"




def flag_bad_files(data_folder):
	"""
	flag non analysable files with a "bad" tag
	"""

	## importation
	import glob
	import os
	import shutil

	## parameters
	bad_id_list = ["32150099", "32140242", "32140244", "32150093", "32152159", "32140206", "32152297", "32151645", '32150107', '32151097', '32151655', '32151659', '32152270', '32160045', '32160388', '32160509', '32160388', '32160509'] # FPS
	bad_id_list += ["32160554","32161017","32161023","32161025","32161027","32161028","32170216"] # DRFZ
	bad_id_list += ["32151963"] # IRCCS
	bad_id_list += ["32150945","32160263","32161168","32161181","32161202"] # KUL
	bad_id_list += ["32150936"] # UNIGE


	for mfi_info_file in glob.glob(data_folder+"/*.txt"):

		mfi_info_file_name = mfi_info_file.split("/")
		mfi_info_file_name = mfi_info_file_name[-1].split("_")

		panel = mfi_info_file_name[1]
		center = mfi_info_file_name[2]
		patient_id = mfi_info_file_name[3]
		finalPart = mfi_info_file_name[4]
		
		if(patient_id in bad_id_list):
			print "[+] Flag file "+str(mfi_info_file)
			mfi_file_new_name = "Panel_"+str(panel)+"_"+str(center)+"_"+str(patient_id)+"_bad_"+str(finalPart)
			os.system("cp "+str(mfi_info_file)+" "+str(data_folder)+"/"+str(mfi_file_new_name))
			os.system("rm "+str(mfi_info_file))





def split_dataset(data_filename, train_proportion):
	"""
	-> Split data_filename to train and test data file, 
	according to train_proportion (belong to 0 - 1)
	"""

	## importation
	import random

	train_data_filename = data_filename.split(".")
	train_data_filename = train_data_filename[0]+"_train.csv"
	train_data = open(train_data_filename, "w")
	train_data.close()

	test_data_filename = data_filename.split(".")
	test_data_filename = test_data_filename[0]+"_test.csv"
	test_data = open(test_data_filename, "w")
	test_data.close()

	## Get the number of entries, assume the file has no header
	data_file = open(data_filename, "r")
	number_of_lines = 0
	for line in data_file:
		number_of_lines += 1
	data_file.close()

	## compute the number of line to keep in train data
	number_of_lines_to_keep = train_proportion * number_of_lines

	## split the data
	number_of_lines_in_train_dataset = 0
	selected_lines = []
	while(number_of_lines_in_train_dataset < number_of_lines_to_keep):
		line_selected = random.randint(0,number_of_lines)
		if(line_selected not in selected_lines):

			## Find the corresponding line
			data_file = open(data_filename, "r")
			cmpt = 0
			for line in data_file:
				
				line = line.split("\n")
				line = line[0]



				if(cmpt == line_selected and cmpt != 0):
				
					if(number_of_lines_in_train_dataset == number_of_lines_to_keep - 1):
						## write line in train data file
						train_data = open(train_data_filename, "a")
						train_data.write(line)
						train_data.close()
					else:
						## write line in train data file
						train_data = open(train_data_filename, "a")
						train_data.write(line+"\n")
						train_data.close()

					selected_lines.append(line_selected)
					number_of_lines_in_train_dataset += 1
				cmpt += 1

			data_file.close()

	## store the rest of lines in test data file
	data_file = open(data_filename, "r")
	number_of_lines_in_test_dataset = 0
	cmpt = 0
	for line in data_file:
		line = line.split("\n")
		line = line[0]
		if(cmpt not in selected_lines and cmpt != 0):
			if(number_of_lines_in_test_dataset == number_of_lines - number_of_lines_in_train_dataset - 1):

				test_data = open(test_data_filename, "a")
				test_data.write(line)
				test_data.close()
			else:
				test_data = open(test_data_filename, "a")
				test_data.write(line+"\n")
				test_data.close()
		cmpt += 1
	data_file.close()











def run_xgboost():
	"""
	##-----------------------------##
	## playing with xgboosted tree ##
	##-----------------------------##
	"""

	## importation
	import matplotlib
	from matplotlib import pyplot
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	import time
	from numpy import loadtxt
	import xgboost
	from xgboost import XGBClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score


	training_data_file = "MFI_FPS_panel_5_training_reduced.csv"
	train_proportion = 0.5

	split_dataset(training_data_file, train_proportion)

	# load data
	train_dataset = loadtxt('MFI_FPS_panel_5_training_reduced_train.csv', delimiter=",")
	test_dataset = loadtxt('MFI_FPS_panel_5_training_reduced_test.csv', delimiter=",")

	# split data into X and y
	train_X = train_dataset[:,1:-1]
	train_Y = train_dataset[:,0]

	test_X = test_dataset[:,1:-1]
	test_Y = test_dataset[:,0]

	model = XGBClassifier()
	model.fit(train_X, train_Y)

	# make predictions for test data
	y_pred = model.predict(test_X)
	predictions = [round(value) for value in y_pred]

	# evaluate predictions
	accuracy = accuracy_score(test_Y, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

	# feature importance
	print(model.feature_importances_)
	f_importance = model.feature_importances_

	pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
	pyplot.show()





def egalize_learning_dataset(data_folder):
	"""
	IN PROGRESS
	"""

	## importation
	import glob
	import random
	import os

	bad_files_list = []
	compensated_file_list = []
	uncompensated_file_list = []

	for file in glob.glob(data_folder+"/*.txt"):

		file_name = file.split("/")
		file_name = file_name[-1].split("_")
		satus = file_name[-1].replace(".txt", "")
		if("bad" in file_name):
			bad_files_list.append(file)
		elif(satus == "compensated"):
			compensated_file_list.append(file)
		else:
			uncompensated_file_list.append(file)
	

	bad_files_cmpt = len(bad_files_list)
	file_to_save = []

	while(len(file_to_save) <= bad_files_cmpt):
		saved_compensated_file = compensated_file_list[random.randint(0,len(compensated_file_list)-1)]
		saved_uncompensated_file = uncompensated_file_list[random.randint(0,len(uncompensated_file_list)-1)]
		file_to_save.append(saved_uncompensated_file)
		file_to_save.append(saved_compensated_file)

	file_to_save += bad_files_list

	for file in glob.glob(data_folder+"/*.txt"):

		if(file not in file_to_save):
			os.system("rm "+str(file))









def run_lda():
	"""
	Need more than 2 classes to plot something
	"""

	## importation
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.decomposition import PCA
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from numpy import loadtxt
	import numpy
	from sklearn.externals import joblib




	data_file = "MFI_FPS_panel_5_plot.csv"

	## load real data
	data = loadtxt(data_file, delimiter=",", skiprows=1)
	X = data[:,1:-1]
	y = data[:,0]
	target_names = ["comp", "uncomp", "decouple"]
	X = numpy.asarray(X)
	y = numpy.asarray(y)


	## perform PCA
	pca = PCA(n_components=2)
	X_r = pca.fit(X).transform(X)

	## perform lda
	lda = LinearDiscriminantAnalysis(n_components=2)
	X_r2 = lda.fit(X, y).transform(X)

	# Percentage of variance explained for each components
	print('explained variance ratio (first two components): %s'% str(pca.explained_variance_ratio_))

	## lda paramaters
	print lda.coef_

	## save lda model
	joblib.dump(lda, 'lda_outlierDetector_model.pkl')

	## plot PCA
	plt.figure()
	colors = ["navy", "green", "red"]
	lw = 2
	for color, i, target_name in zip(colors, [0, 1, 2], target_names):
	    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
	                label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of MFI dataset Mean')

	## plot LDA, need more than 2 classes
	plt.figure()
	for color, i, target_name in zip(colors, [0, 1, 2], target_names):
	    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
	                label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('LDA of MFI dataset Mean')


	plt.show()








def generate_pca_file():
	"""
	IN PROGRESS
	kind of trash run ...
		
	group 0 : analysable et compense
	group 1 : analysable et pas compense
	group 2 : non analysable

	"""

	dataset = open("MFI_training_data.csv", "r")
	output_file = open("MFI_FPS_panel_5_plot.csv", "w")

	cmpt = 0
	for line in dataset:

		if(cmpt == 0):
			header = "group,"

			line = line.rstrip()
			line_in_array = line.split(",")
			index = 0
			for elt in line_in_array:
				if(index > 1):
					header += str(elt)+","
				index += 1

			header = header[:-1]

			output_file.write(header+"\n")

		else:

			

			line = line.rstrip()
			line_in_array = line.split(",")
			
			index = 0
			group = "NA"

			if(line_in_array[0] == "1" and line_in_array[1] == "1"):
				group = 0			
			elif(line_in_array[0] == "0"):
				group = 2
			else:
				group = 1

			line_to_write = str(group)+","
			for scalar in line_in_array:
				if(index > 1):
					line_to_write += str(scalar)+","
				index += 1

			line_to_write = line_to_write[:-1]

			output_file.write(line_to_write+"\n")

		cmpt += 1
	output_file.close()
	dataset.close()




def generate_lda_file(pca_file):
	"""
	IN PROGRESS
	
	Generate a lda file from the pca file where we select only
	the uncompensated and the decouple files (i.e group 1 and 2)
	to perfrom a 2 group LDA

	"""


	data_file = open(pca_file, "r")
	output_file = open("lda_input_data.csv", "w")
	cmpt = 0
	for line in data_file:
		if(cmpt == 0):
			output_file.write(line)
		else:
			line = line.rstrip()
			line_in_array = line.split(",")
			if(line_in_array[0] != str(0)):
				output_file.write(line+"\n")
		cmpt += 1

	output_file.close()
	data_file.close()



def generate_lda_model():
	"""
	Scavange from test space
	"""

	## importation
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.decomposition import PCA
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from numpy import loadtxt
	import numpy
	from sklearn.externals import joblib
	import os


	## generate training data file
	
	os.system("rm trash/MFI_FPS_all/*")
	extract_MFI_informations("trash/MFI_input_FPS")
	os.system("cp data/MFI/compensated/* trash/MFI_FPS_all/")
	os.system("cp data/MFI/uncompensated/* trash/MFI_FPS_all/")
	flag_bad_files("trash/MFI_FPS_all")
	create_outlier_detection_training_datafile("trash/MFI_FPS_all")
	generate_pca_file()
	generate_lda_file("MFI_FPS_panel_5_plot.csv")

	## load real data
	data_file = "lda_input_data.csv"
	data = loadtxt(data_file, delimiter=",", skiprows=1)
	X = data[:,1:-1]
	y = data[:,0]
	X = numpy.asarray(X)
	y = numpy.asarray(y)

	lda = LinearDiscriminantAnalysis(n_components=2)
	X_r2 = lda.fit(X, y).transform(X)

	## save lda model
	joblib.dump(lda, 'lda_outlierDetector_model.pkl')

	## model validation
	data_file = "MFI_FPS_5_validation.csv"
	data = loadtxt(data_file, delimiter=",", skiprows=1)
	X = data[:,1:-1]
	y = data[:,0]
	target_names = ["comp", "uncomp", "decouple"]
	X = numpy.asarray(X)
	y = numpy.asarray(y)

	predictions = lda.predict(X)
	cmpt = 0
	for prediction in predictions:
		print str(prediction) +" || "+str(y[cmpt])







def detect_decouplage(fcs_file, model):
	"""
	
	Try to predict if ther is a decoupling
	in the fcs file. Use MFI information

	return 1 if file looks okay
	return 2 if file is suspect

	model is lda model file to load, -> trying stuff

	"""

	## importation
	import os
	from sklearn.externals import joblib
	from numpy import loadtxt
	import numpy

	## parameters
	MFIEXTRACTION_SCRIPT = "ExtractMFIinformation.R"

	## clean data MFI folders
	os.system("rm data/MFI/compensated/*")
	os.system("rm data/MFI/uncompensated/*")

	## get information from file name
	file_name_in_array = fcs_file.split("/")
	if("_comp" not in file_name_in_array[-1]):
		status = "uncompensated"
	else:
		status = "compensated"
	file_name_in_array = file_name_in_array[-1].split("_")
	fcs_panel = file_name_in_array[1]
	fcs_id = file_name_in_array[2]
	fcs_center = file_name_in_array[3]
	mfi_information_file_name = "data/MFI/"+str(status)+"/Panel_"+str(fcs_panel)+"_"+str(fcs_center)+"_"+str(fcs_id)+"_"+str(status)+".txt"
	mfi_evaluation_file = "outlier_evaluation_data_"+str(fcs_center)+"_"+str(fcs_panel)+"_"+str(fcs_id)+".csv"

	## Extract MFI information from fcs file
	os.system("Rscript "+MFIEXTRACTION_SCRIPT+" "+str(fcs_file))

	## generate a data file from MFI information file
	flag_bad_files("data/MFI/"+str(status))
	create_outlier_detection_training_datafile("data/MFI/"+str(status))
	generate_pca_file()
	os.system("mv MFI_FPS_panel_5_plot.csv "+str(mfi_evaluation_file))
	generate_lda_file(mfi_evaluation_file)

	## load model
	#clf = joblib.load('lda_outlierDetector_model.pkl')
	clf = joblib.load(model)

	## load data to evaluate
	data = loadtxt(mfi_evaluation_file, delimiter=",", skiprows=1)
	X = data[1:-1]
	y = data[0]
	X = numpy.asarray(X)
	y = numpy.asarray(y)

	## compute prediction
	prediction = clf.predict([X])

	## return prediction
	return str(prediction[0])










def select_random_bad_files(number_of_bad_files):
	"""
	"""

	
	## importation
	import random
	import glob
	import os

	## parameters
	folder_list = ["panel_5_DRFZ", "panel_5_FPS", "panel_5_IDIBELL", "panel_5_IRCCS", "panel_5_KUL", "PANEL_5_SAS", "PANEL_5_UBO", "panel_5_UCL"]
	#folder_list = ["panel_5_FPS", "panel_5_IDIBELL", "panel_5_IRCCS", "PANEL_5_SAS", "PANEL_5_UBO", "panel_5_UCL"]
	forbidden_id = []

	data = open("/home/elwin/decouplage_pannel_5.txt", "r")
	for line in data:
		line = line.rstrip()
		line_in_array = line.split(";")

		index = 0
		for elt in line_in_array:
			if(index != 0 and elt not in forbidden_id):
				forbidden_id.append(elt)
			index +=1
	data.close()

	for x in xrange(0,number_of_bad_files):
		valid_choice = False
		while(not valid_choice):

			center = folder_list[random.randint(0,len(folder_list)-1)]
			fcs_files = glob.glob("/home/elwin/"+str(center)+"/*.fcs")
			candidate_file = fcs_files[random.randint(0,len(fcs_files)-1)]
			candidate_file_name = candidate_file.split("/")
			candidate_file_name = candidate_file_name[-1].split("_")
			if(candidate_file_name[2] in forbidden_id):
				os.system("cp "+str(candidate_file)+" trash/MFI_input_FPS/")
				valid_choice = True



def select_random_good_files(number_of_good_files):
	"""
	"""

	
	## importation
	import random
	import glob
	import os

	## parameters
	folder_list = ["panel_5_DRFZ", "panel_5_FPS", "panel_5_IDIBELL", "panel_5_IRCCS", "panel_5_KUL", "PANEL_5_SAS", "PANEL_5_UBO", "panel_5_UCL"]
	#folder_list = ["panel_5_FPS", "panel_5_IDIBELL", "panel_5_IRCCS", "PANEL_5_SAS", "PANEL_5_UBO", "panel_5_UCL"]
	forbidden_id = []

	data = open("/home/elwin/decouplage_pannel_5.txt", "r")
	for line in data:
		line = line.rstrip()
		line_in_array = line.split(";")
		index = 0
		for elt in line_in_array:
			if(index != 0 and elt not in forbidden_id):
				forbidden_id.append(elt)
			index +=1
	data.close()

	data = open("/home/elwin/pannel_5_fichier_de_merde_analyse_qd_mm_.txt", "r")
	for line in data:
		line = line.rstrip()
		line_in_array = line.split(";")

		index = 0
		for elt in line_in_array:
			if(index != 0 and elt not in forbidden_id):
				forbidden_id.append(elt)
			index +=1
	data.close()

	for x in xrange(0,number_of_good_files):
		valid_choice = False
		while(not valid_choice):

			center = folder_list[random.randint(0,len(folder_list)-1)]
			fcs_files = glob.glob("/home/elwin/"+str(center)+"/*.fcs")
			candidate_file = fcs_files[random.randint(0,len(fcs_files)-1)]
			candidate_file_name = candidate_file.split("/")
			candidate_file_name = candidate_file_name[-1].split("_")
			if(candidate_file_name[2] not in forbidden_id):
				os.system("cp "+str(candidate_file)+" trash/MFI_input_FPS/")
				valid_choice = True




select_random_bad_files(10)



import glob
import random
import os
import numpy as np

## generate validation dataset

## get the list of forbidden id
forbidden_id = []
number_of_files_to_evaluate = 25


log_file = open("learn_lda.log", "w")

good_enough = False
generation = 1
while(not good_enough):

	log_file.write("Generation,"+str(generation)+"\n")

	score_E1 = 0
	score_E2 = 0
	score_E3 = 0
	


	## Build and save model 1
	## new training data for the next model
	
	## Look for a wise elector, if none is avaialable build
	## the first model
	if(os.path.isfile("models/elector_wise.pkl")):
		print "[+] Find Wise elector"
		os.system("cp models/elector_wise.pkl models/elector_1.pkl")
	else:
		os.system("rm trash/MFI_input_FPS/*")
		select_random_bad_files(12)
		select_random_good_files(15)

		## Model Construction
		print "[+] Model 1 Construction"
		generate_lda_model()

		## Save model 1
		os.system("cp lda_outlierDetector_model.pkl models/elector_1.pkl")


	## Build and save model 2
	## new training data for the next model
	os.system("rm trash/MFI_input_FPS/*")
	select_random_bad_files(12)
	select_random_good_files(15)

	## Model Construction
	print "[+] Model 2 Construction"
	generate_lda_model()

	## Save model 1
	os.system("cp lda_outlierDetector_model.pkl models/elector_2.pkl")


	## Build and save model 3
	## new training data for the next model
	os.system("rm trash/MFI_input_FPS/*")
	select_random_bad_files(12)
	select_random_good_files(15)

	## Model Construction
	print "[+] Model 3 Construction"
	generate_lda_model()

	## Save model 1
	os.system("cp lda_outlierDetector_model.pkl models/elector_3.pkl")



	##------------------##
	## Model Evaluation ##
	##------------------##
	## clean validation folders and create validation dataset 
	os.system("rm trash/FCS_multicenter_validation_2/*")
	data = open("/home/elwin/decouplage_pannel_5.txt", "r")
	for line in data:
		line = line.rstrip()
		line_in_array = line.split(";")

		index = 0
		for elt in line_in_array:
			if(index != 0 and elt not in forbidden_id):
				forbidden_id.append(elt)
			index +=1
	data.close()

	data = open("/home/elwin/pannel_5_fichier_de_merde_analyse_qd_mm_.txt", "r")
	for line in data:
		line = line.rstrip()
		line_in_array = line.split(";")

		index = 0
		for elt in line_in_array:
			if(index != 0 and elt not in forbidden_id):
				forbidden_id.append(elt)
			index +=1
	data.close()


	folder_list = ["panel_5_FPS", "panel_5_IDIBELL", "panel_5_IRCCS", "PANEL_5_SAS", "PANEL_5_UBO", "panel_5_UCL"]
	for x in xrange(0,number_of_files_to_evaluate):
		valid_choice = False
		while(not valid_choice):
			center = folder_list[random.randint(0,len(folder_list)-1)]
			fcs_files = glob.glob("/home/elwin/"+str(center)+"/*.fcs")
			candidate_file = fcs_files[random.randint(0,len(fcs_files)-1)]

			candidate_file_name = candidate_file.split("/")
			candidate_file_name = candidate_file_name[-1].split("_")

			if(candidate_file_name[2] not in forbidden_id):
				os.system("cp "+str(candidate_file)+" trash/FCS_multicenter_validation_2/")
				valid_choice = True







	## perform validation -> false positive Score
	print "[+] False positive Evaluation" 
	false_positive_count = 0
	for test_file in glob.glob("trash/FCS_multicenter_validation_2/*.fcs"):
		test_file_name = test_file.split("/")
		test_file_name = test_file_name[-1].split("_")
		center = test_file_name[3]

		prediction_1 = detect_decouplage(test_file, "models/elector_1.pkl")
		prediction_2 = detect_decouplage(test_file, "models/elector_2.pkl")
		prediction_3 = detect_decouplage(test_file, "models/elector_3.pkl")

		if(prediction_1[0] == "1"):
			score_E1 += 1
		if(prediction_2[0] == "1"):
			score_E2 += 1
		if(prediction_3[0] == "1"):
			score_E3 += 1

		prediction_list = [prediction_1,prediction_2,prediction_3]
		prediction_final = str(max(set(prediction_list), key=prediction_list.count))
		
		print "[+] "+str(center) +" : "+ str(prediction_1[0]) +" || "+str(prediction_2[0]) +" || "+str(prediction_3[0]) +" => "+prediction_final
		log_file.write("FP,"+"[+] "+str(center) +" : "+ str(prediction_1[0]) +" || "+str(prediction_2[0]) +" || "+str(prediction_3[0]) +" => "+prediction_final+"\n")

		if(str(prediction_final[0]) == "2" ):
			false_positive_count += 1
	
	false_positive_count = float(false_positive_count) / float(number_of_files_to_evaluate) * 100
	print '[+] False Positive : '+str(false_positive_count)
	log_file.write("FP,"+str(false_positive_count)+"\n")
	

	## perform validation -> false negative score
	print "[+] False Negative Evaluation"
	false_negative_count = 0
	for test_file in glob.glob("trash/FCS_milticenter_validation/*.fcs"):
		test_file_name = test_file.split("/")
		test_file_name = test_file_name[-1].split("_")
		center = test_file_name[3]


		prediction_1 = detect_decouplage(test_file, "models/elector_1.pkl")
		prediction_2 = detect_decouplage(test_file, "models/elector_2.pkl")
		prediction_3 = detect_decouplage(test_file, "models/elector_3.pkl")


		if(prediction_1[0] == "2"):
			score_E1 += 1
		if(prediction_2[0] == "2"):
			score_E2 += 1
		if(prediction_3[0] == "2"):
			score_E3 += 1


		prediction_list = [prediction_1,prediction_2,prediction_3]
		prediction_final = str(max(set(prediction_list), key=prediction_list.count))
		
		print "[+] "+str(center) +" : "+ str(prediction_1[0]) +" || "+str(prediction_2[0]) +" || "+str(prediction_3[0]) +" => "+prediction_final
		log_file.write("FN,"+"[+] "+str(center) +" : "+ str(prediction_1[0]) +" || "+str(prediction_2[0]) +" || "+str(prediction_3[0]) +" => "+prediction_final+"\n")

		
		if(str(prediction_final[0]) == "1" ):
			false_negative_count += 1
	

	false_negative_count = float(false_negative_count) / float(number_of_files_to_evaluate) * 100
	print '[+] False Negative : '+str(false_negative_count)
	log_file.write("FN,"+str(false_negative_count)+"\n")


	if(float(false_negative_count) <= 0.0 and float(false_positive_count) <= 3.0):
		good_enough = True
	


	## Identify the wise elector
	score_list = [score_E1,score_E2,score_E3]
	max_index = np.argmax(score_list)
	wise_elector_number = max_index+1
	os.system("cp models/elector_"+str(wise_elector_number)+".pkl models/elector_wise.pkl")




	generation +=1

log_file.close()




#generate_pca_file()

#run_lda()
#extract_MFI_informations("trash/MFI_input_FPS")
#flag_bad_files("trash/MFI_FPS")

#create_outlier_detection_training_datafile("trash/MFI_FPS")
#egalize_learning_dataset("trash/MFI_FPS")

#run_xgboost()


#from netabio import quality_control
#quality_control.basic_check("MFI_FPS_panel_5_plot.csv")



"""
egalize_learning_dataset("trash/MFI_FPS")
create_outlier_detection_training_datafile("trash/MFI_FPS")
generate_pca_file()
run_lda()
"""




## data generation
"""
import os
extract_MFI_informations("trash/MFI_input_FPS")
os.system("cp data/MFI/compensated/* trash/MFI_FPS_all/")
os.system("cp data/MFI/uncompensated/* trash/MFI_FPS_all/")
flag_bad_files("trash/MFI_FPS_all")
create_outlier_detection_training_datafile("trash/MFI_FPS_all")
generate_pca_file()
"""

## play with lda
#run_lda()



#print X[0:5]
#print clf.predict([X[12]])
#print y[12]

## create validation dataset
"""
fps_bad_files = ["32150107","32151097","32151655","32151659","32152270","32160045","32160388","32160509","32160388","32160509"]
bad_id_list = ["32150099", "32140242", "32140244", "32150093", "32152159", "32140206", "32152297", "32151645"]
validation_list = []
for stuff in fps_bad_files:
	if stuff not in bad_id_list:
		validation_list.append(stuff)
import glob
import os
for fcs_file in glob.glob("/home/elwin/panel_5_FPS/*.fcs"):
	fcs_file_name = fcs_file.split("/")
	fcs_file_name = fcs_file_name[-1].split("_")
	if(fcs_file_name[2] in validation_list):
		os.system("cp "+str(fcs_file)+" trash/FCS_FPS_validation/")
extract_MFI_informations("trash/FCS_FPS_validation")
os.system("cp data/MFI/compensated/* trash/MFI_FPS_validation/")
os.system("cp data/MFI/uncompensated/* trash/MFI_FPS_validation/")
flag_bad_files("trash/MFI_FPS_validation")
create_outlier_detection_training_datafile("trash/MFI_FPS_validation")
generate_pca_file()
os.system("mv MFI_FPS_panel_5_plot.csv MFI_FPS_5_validation.csv")
"""


## test the model, predict validation dataset

"""
## load model
clf = joblib.load('lda_outlierDetector_model.pkl')

## load real data
data_file = "MFI_FPS_5_validation.csv"
data = loadtxt(data_file, delimiter=",", skiprows=1)
X = data[:,1:-1]
y = data[:,0]
target_names = ["comp", "uncomp", "decouple"]
X = numpy.asarray(X)
y = numpy.asarray(y)


predictions = clf.predict(X)

cmpt = 0
for prediction in predictions:
	print str(prediction) +" || "+str(y[cmpt]) 

"""




#generate_lda_model()