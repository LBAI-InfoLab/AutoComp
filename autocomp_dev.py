##
## Try to find a smart solution to the compensation problem
## by smart i mean "pseudo linear" solution, without invoking
## big monsters like CNN
##



def generate_image_from_fcs(fcs_file):
	"""
	Generate image for all channel
	combination in the fcs file
	"""

	## importation
	import FlowCal
	import matplotlib.pyplot as plt

	## open fcs file
	s = FlowCal.io.FCSData(fcs_file)
	s = FlowCal.transform.to_rfi(s)

	## exctract list of channels
	channel_list = s.channels

	## compute and save image
	for channel_1 in channel_list:
		for channel_2 in channel_list:
			if(channel_1 != channel_2):
				try:
					png_file_name = str(fcs_file)+"_"+str(channel_1)+"_"+str(channel_2)+".png"
					FlowCal.plot.density2d(s, channels=[str(channel_1), str(channel_2)], mode='scatter', yscale='logicle', xscale='logicle', ylim=-1000, xlim=(0,500000))
					#FlowCal.plot.density2d(s, channels=[str(channel_1), str(channel_2)], ylim=-1000, xlim=(0,500000))
					plt.savefig(png_file_name)
					plt.close()
				except:
					pass


def randomy_split_dataset(input_data_file, proportion):
	"""
	Randomly split the input data file into 2 data file
	according to proportion
	"""

	## importation
	import random
	

	## parameters
	output_data_1_file = input_data_file.split(".")
	output_data_1_file = output_data_1_file[0]+"_splited_1.csv"
	output_data_2_file = output_data_1_file.replace("1", "2")

	## get the number of lines in data file
	case_cmpt = 0
	input_data = open(input_data_file, "r")
	for line in input_data:
		case_cmpt +=1
	case_cmpt = case_cmpt - 1 # drop the header
	number_of_line_to_keep_in_data_1 = float(proportion)*case_cmpt


	## write header
	input_data = open(input_data_file, "r")
	output_data_1 = open(output_data_1_file, "w")
	output_data_2 = open(output_data_2_file, "w")
	cmpt = 0
	for line in input_data:
		if(cmpt == 0):
			output_data_1.write(line)
			output_data_2.write(line)
		cmpt += 1
	input_data.close()
	output_data_2.close()
	output_data_1.close()


	selected_lines = []
	number_of_line_in_data_1 = 0
	while(number_of_line_in_data_1 < number_of_line_to_keep_in_data_1):
		input_data = open(input_data_file, "r")
		output_data_1 = open(output_data_1_file, "a")
		cmpt = 0		

		for line in input_data:
			if(cmpt != 0):
				dice = random.randint(0,100)
				if(dice > 50 and number_of_line_in_data_1 < number_of_line_to_keep_in_data_1 and cmpt not in selected_lines):
					output_data_1.write(line)
					selected_lines.append(cmpt)
					number_of_line_in_data_1 += 1
				
			cmpt += 1

		input_data.close()
		output_data_1.close()


	## write data 2
	input_data = open(input_data_file, "r")
	output_data_2 = open(output_data_2_file, "a")
	cmpt = 0
	for line in input_data:
		if(cmpt != 0 and cmpt not in selected_lines):
			output_data_2.write(line)
		cmpt += 1
	input_data.close()
	output_data_2.close()



def plot_compensation_matrix(compensation_matrix):
	"""
	Generate an heatmap for the compensation matrix file
	and save it under the results/image folder

	Work on predicted, mannually compensated and uncompensated matrix
	"""

	## importation
	import numpy
	import matplotlib
	import matplotlib.pyplot as plt



	## Detect matrix type
	## - uncompensated
	## - predicted
	## - Manually compensated

	matrix_file_name = compensation_matrix.split("/")
	matrix_file_name = matrix_file_name[-1]
	matrix_file_name_in_array = matrix_file_name.split("_")
	matrix_tag = matrix_file_name_in_array[-1]

	if(matrix_tag not in ["uncompensated.txt", "predicted.txt"]):
		matrix_tag = "compensated"

	output_file_name = "results/images/compensation_matrix_heatmap_"+str(matrix_tag)+".png"
	output_file_name = output_file_name.replace(".txt", "")


	## Parse Matrix file
	## init grid to display structure
	matrix_grid = []
	scalar = "NA"
	for x in xrange(0,8):
		matrix_grid.append([])
		for y in xrange(0,8):
			matrix_grid[x].append(scalar)


	## Deal with uncompensated matrix
	if(matrix_tag == "uncompensated.txt"):
		matrix_data = open(compensation_matrix, "r")
		cmpt = 0
		pos_y = 0
		for line in matrix_data:
			line = line.rstrip()
			if(cmpt > 0):
				line_in_array = line.split(",")
				index = 0
				pos_x = 0
				for scalar in line_in_array:
					if(index > 0):
						matrix_grid[pos_y][pos_x] = float(scalar)
						pos_x += 1
					index += 1
				pos_y += 1
			cmpt += 1
		matrix_data.close()

	## Deal with predicted matrix and 
	## mannualy compensated matrix
	else:
		matrix_data = open(compensation_matrix, "r")
		cmpt = 0
		pos_y = 0
		for line in matrix_data:
			line = line.rstrip()
			if(cmpt > 0):
				line_in_array = line.split("\t")
				index = 0
				pos_x = 0
				for scalar in line_in_array:
					if(index > 1):
						matrix_grid[pos_y][pos_x] = float(scalar)/100.0
						pos_x += 1
					index += 1
				pos_y += 1
			cmpt += 1
		matrix_data.close()


	## Plot matrix
	## get the label
	x_label = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	y_label = x_label

	## get the grid
	grid_to_display = numpy.asarray(matrix_grid)

	## plot the stuff
	fig, ax = plt.subplots()
	im = ax.imshow(grid_to_display)

	# We want to show all ticks...
	ax.set_xticks(numpy.arange(len(x_label)))
	ax.set_yticks(numpy.arange(len(y_label)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(x_label)
	ax.set_yticklabels(y_label)

	# Add colorbar, make sure to specify tick locations to match desired ticklabels
	cbar = fig.colorbar(im, ticks=[numpy.amin(grid_to_display), numpy.amax(grid_to_display)])
	cbar.ax.set_yticklabels([numpy.amin(grid_to_display), numpy.amax(grid_to_display)])  # vertically oriented colorbar

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	ax.set_title("Matrix compensation")
	fig.tight_layout()
	plt.savefig(output_file_name)
	#plt.show()



def transpose_matrix_file(matrix_file):
	"""
	Transpose the matrix in matrix file,
	work for predicted matrix, overwite the
	matrix file
	"""

	# importation
	import numpy
	import shutil

	## parameters
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	output_file_name = matrix_file.split("/")
	output_file_name = output_file_name[-1].split(".")
	output_file_name = output_file_name[0]+"_transposed."+output_file_name[1]

	## init grid
	matrix_grid = []
	scalar = "NA"
	for x in xrange(0,8):
		grid_vector = []
		for x in xrange(0,8):
			grid_vector.append(scalar)
		matrix_grid.append(grid_vector)

	## Parse data
	matrix_data = open(matrix_file, "r")
	cmpt = 0
	pos_y = 0
	header = ""
	prefix = ""
	for line in matrix_data:
		if(cmpt > 0):
			line = line.rstrip()
			line_in_array = line.split("\t")
			index = 0
			pos_x = 0
			prefix = line_in_array[0]
			for scalar in line_in_array:
				if(index > 1):
					matrix_grid[pos_y][pos_x] = float(scalar)
					pos_x += 1

				index += 1
			pos_y += 1
		else:
			header = line
		cmpt += 1
	matrix_data.close()

	## transpose matrix
	matrix_grid = numpy.asarray(matrix_grid)
	matrix_grid = matrix_grid.transpose()

	## write new matrix file
	transposed_matrix = open(output_file_name, "w")
	transposed_matrix.write(header)
	cmpt = 0
	for vector in matrix_grid:
		line_to_write = str(prefix)+"\t"+channel_list[cmpt]+"\t"
		for scalar in vector:
			line_to_write += str(scalar)+"\t"
		line_to_write = line_to_write[:-1]
		if(cmpt == 7):
			transposed_matrix.write(line_to_write)
		else:
			transposed_matrix.write(line_to_write+"\n")

		cmpt += 1
	transposed_matrix.close()

	## replace old matrix file transpose matrix file
	shutil.copy(output_file_name, matrix_file)



def transpose_matrix_file_dev(matrix_file):
	"""
	Transpose the matrix in matrix file,
	work for predicted matrix, overwite the
	matrix file
	"""

	# importation
	import numpy
	import shutil

	## parameters
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	output_file_name = matrix_file.split("/")
	output_file_name = output_file_name[-1].split(".")
	output_file_name = output_file_name[0]+"_transposed."+output_file_name[1]

	## init grid
	matrix_grid = []
	scalar = "NA"
	for x in xrange(0,8):
		grid_vector = []
		for x in xrange(0,8):
			grid_vector.append(scalar)
		matrix_grid.append(grid_vector)

	## Parse data for predicted matrix file
	matrix_data = open(matrix_file, "r")
	cmpt = 0
	pos_y = 0
	header = ""
	prefix = ""
	for line in matrix_data:

		line_in_array = line.split("\t")
		

		## uncompensated matrix
		if(len(line_in_array) == 1):
			if(cmpt > 0):
				line = line.rstrip()
				line_in_array = line.split(",")
				index = 0
				pos_x = 0
				prefix = "0"
				for scalar in line_in_array:
					if(index > 0):
						matrix_grid[pos_y][pos_x] = float(scalar)
						pos_x += 1

					index += 1
				pos_y += 1
			else:
				header = line
				channel_list = header.rstrip()
				channel_list = channel_list.split(",")
				channel_list = channel_list[1:]


		## predcted matrix
		else:
			if(cmpt > 0):
				line = line.rstrip()
				line_in_array = line.split("\t")
				index = 0
				pos_x = 0
				prefix = line_in_array[0]
				for scalar in line_in_array:
					if(index > 1):
						matrix_grid[pos_y][pos_x] = float(scalar)
						pos_x += 1

					index += 1
				pos_y += 1
			else:
				header = line
		cmpt += 1
	matrix_data.close()

	## transpose matrix
	matrix_grid = numpy.asarray(matrix_grid)
	matrix_grid = matrix_grid.transpose()


	## write new matrix file
	transposed_matrix = open(output_file_name, "w")
	transposed_matrix.write(header)
	cmpt = 0
	for vector in matrix_grid:
		line_to_write = str(prefix)+"\t"+channel_list[cmpt]+"\t"
		for scalar in vector:
			line_to_write += str(scalar)+"\t"
		line_to_write = line_to_write[:-1]
		if(cmpt == 7):
			transposed_matrix.write(line_to_write)
		else:
			transposed_matrix.write(line_to_write+"\n")

		cmpt += 1
	transposed_matrix.close()

	## replace old matrix file transpose matrix file
	shutil.copy(output_file_name, matrix_file)



def load_data_for_training(uncompensated_fcs_folder, compensated_fcs_folder, matrix_compensated_folder):
	"""
	IN PROGRESS
	"""

	## importation
	import glob
	import os

	## parameters
	uncompensated_output = "data/fcs/raw/"
	compensated_output = "data/fcs/compensated/"
	matrix_output = "data/matrix/compensated/"
	EXTRACTMATRIX_SCRIPT = "extractMatrix.R"


	## copy all files in the uncompensated_folder
	uncompensated_file_list = glob.glob(uncompensated_fcs_folder+"/*.fcs")
	for uncompensated_file in uncompensated_file_list:
		os.system("cp "+uncompensated_file+" "+uncompensated_output)

	## copy all files in the compensated folder
	compensated_file_list = glob.glob(compensated_fcs_folder+"/*.fcs")
	for compensated_file in compensated_file_list:
		os.system("cp "+compensated_file+" "+compensated_output)

	## Extract matrix for uncompensated files
	for fcs_file in glob.glob(uncompensated_output+"/*.fcs"):
		os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(fcs_file))

	## copy all mannual compensated matrix files
	matrix_file_list = glob.glob(matrix_compensated_folder+"/*.txt")
	for matrix_file in matrix_file_list:
		os.system("cp "+str(matrix_file)+" "+matrix_output)



def select_compensation_matrix():
	"""
	Find the good compensation files
	and store them in the data folder,
	by good I mean with an equivalent
	uncompensated file

	Start small
	"""

	## importation
	import glob
	import shutil

	## get files list
	uncompensated_matrix_files = glob.glob("data/matrix/uncompensated/*.txt")
	compensated_matrix_files = glob.glob("/home/elrohir/COMPENSATION/*.txt")

	output_directory = "data/matrix/compensated/"
	found_cmpt = 0

	for compensated_matrix in compensated_matrix_files:

		clear_to_copy = False

		## parse compensated matrix file name
		compensated_matrix_name = compensated_matrix.split("/")
		compensated_matrix_name_in_array = compensated_matrix_name[-1].split("_")
		compensated_matrix_panel = compensated_matrix_name_in_array[1]
		compensated_matrix_center = compensated_matrix_name_in_array[2]
		compensated_matrix_ID = compensated_matrix_name_in_array[3]
		compensated_matrix_ID = compensated_matrix_ID.split(".")
		compensated_matrix_ID = compensated_matrix_ID[0]

		for uncompensated_matrix in uncompensated_matrix_files:

			## parse uncompensated matrix file name
			uncompensated_matrix_name = uncompensated_matrix.split("/")
			uncompensated_matrix_name_in_array = uncompensated_matrix_name[-1].split("_")
			uncompensated_matrix_panel = uncompensated_matrix_name_in_array[1]
			uncompensated_matrix_center = uncompensated_matrix_name_in_array[2]
			uncompensated_matrix_ID = uncompensated_matrix_name_in_array[3]		

			## test the files, get the "goods one"
			if(uncompensated_matrix_center == compensated_matrix_center and uncompensated_matrix_panel == compensated_matrix_panel and uncompensated_matrix_ID == compensated_matrix_ID):
				clear_to_copy = True

		if(clear_to_copy):
			shutil.copy(compensated_matrix, output_directory+str(compensated_matrix_name[-1]))
			found_cmpt += 1

	print "[*] Retrieve "+str((float(found_cmpt)/float(len(uncompensated_matrix_files))*100)) +"%"




def compute_delta_matrix(matrix_1, matrix_2, output_matrix):
	"""
	compute and write a new matrix from matrix 1 and 2
	which is the "delta matrix", represent the variation
	at each position

	write the result in an output file given ny the
	output_matrix parameters

	matrix_1 correspond to the uncompensated matrix
	matrix_2 correspond to the compensated matrix
	"""


	## deal with the uncompensated matrix
	matrix_1_data = open(matrix_1, "r")
	matrix_1_value = []
	
	cmpt = 0
	for line in matrix_1_data:
		if(cmpt != 0):
			vector = []
			line = line.rstrip()
			line_in_array = line.split(",")
			index = 0
			for scalar in line_in_array:
				if(index != 0):
					vector.append(scalar)
				index += 1
			matrix_1_value.append(vector)
		cmpt += 1
	matrix_1_data.close()

	## deal with the compensated matrix
	matrix_2_data = open(matrix_2, "r")
	matrix_2_value = []
	cmpt = 0
	for line in matrix_2_data:
		line = line.rstrip()
		if(cmpt != 0):
			vector = []
			line_in_array = line.split("\t")
			
			index = 0
			for scalar in line_in_array:
				if(index >= 2):
					vector.append(float(scalar)/100.0)
				index += 1
			matrix_2_value.append(vector)
		cmpt += 1
	matrix_2_data.close()


	## compute the delta matrix
	delta_matrix = []
	pos_y = 0
	for vector in matrix_1_value:
		delta_vector = []
		pos_x = 0
		for scalar in vector:
			compensated_scalar = matrix_2_value[pos_y][pos_x] 
			delta_scalar = float(compensated_scalar) - float(scalar)
			delta_vector.append(delta_scalar)
			pos_x += 1

		delta_matrix.append(delta_vector)
		pos_y += 1


	## write the delta matrix in a csv file
	delta_matrix_file = open(output_matrix, "w")
	cmpt = 0
	for vector in delta_matrix:
		line_to_write = ""
		for scalar in vector:
			line_to_write += str(scalar)+","
		line_to_write = line_to_write[:-1]
		if(cmpt < len(delta_matrix)):
			delta_matrix_file.write(line_to_write+"\n")
		else:
			delta_matrix_file.write(line_to_write)
		cmpt +=1
	delta_matrix_file.close()





def generate_all_delta_matrix():
	"""
	Generate all delta matrix from matrix
	found in the data folder
	"""

	## importation
	import glob

	## get files list
	uncompensated_matrix_files = glob.glob("data/matrix/uncompensated/*.txt")
	compensated_matrix_files = glob.glob("data/matrix/compensated/*.txt")

	output_directory = "data/matrix/delta/"
	found_cmpt = 0

	for compensated_matrix in compensated_matrix_files:

		## parse compensated matrix file name
		compensated_matrix_name = compensated_matrix.split("/")
		compensated_matrix_name_in_array = compensated_matrix_name[-1].split("_")
		compensated_matrix_panel = compensated_matrix_name_in_array[1]
		compensated_matrix_center = compensated_matrix_name_in_array[2]
		compensated_matrix_ID = compensated_matrix_name_in_array[3]
		compensated_matrix_ID = compensated_matrix_ID.split(".")
		compensated_matrix_ID = compensated_matrix_ID[0]

		for uncompensated_matrix in uncompensated_matrix_files:

			## parse uncompensated matrix file name
			uncompensated_matrix_name = uncompensated_matrix.split("/")
			uncompensated_matrix_name_in_array = uncompensated_matrix_name[-1].split("_")
			uncompensated_matrix_panel = uncompensated_matrix_name_in_array[1]
			uncompensated_matrix_center = uncompensated_matrix_name_in_array[2]
			uncompensated_matrix_ID = uncompensated_matrix_name_in_array[3]		

			## test the files, get the "goods one"
			if(uncompensated_matrix_center == compensated_matrix_center and uncompensated_matrix_panel == compensated_matrix_panel and uncompensated_matrix_ID == compensated_matrix_ID):

				delta_matrix = output_directory+str(compensated_matrix_name[-1])
				delta_matrix = delta_matrix.split(".")
				delta_matrix = str(delta_matrix[0])+"_delta."+str(delta_matrix[1])

				compute_delta_matrix(uncompensated_matrix, compensated_matrix, delta_matrix)







def get_untouched_position():
	"""
	
	Loop over the delta matrix file and find the
	position that are left untouched by the compensation
	(i.e delta = 0)
	
	display the grid in the console

	return list of position

	"""

	## importation
	import glob


	## init display matrix
	display_matrix = []
	untouched_char = "#"
	touched_char = "-"
	for x in xrange(0,8):
		display_matrix.append([])
		for y in xrange(0,8):
			display_matrix[x].append(untouched_char)


	## loop over all delta matrix
	delta_matrix_files = glob.glob("data/matrix/delta/*.txt")
	for delta_matrix in delta_matrix_files:

		current_delta_matrix = []
		data_file = open(delta_matrix, "r")
		
		pos_y = 0
		for line in data_file:
			line = line.rstrip()
			line_in_array = line.split(",")
			pos_x = 0
			for scalar in line_in_array:
				if(float(scalar) != 0.0):
					display_matrix[pos_y][pos_x] = touched_char
				pos_x += 1
			pos_y += 1
		data_file.close()
		

		## Display grid
		untouched_position = []
		pos_y = 0
		for vector in display_matrix:
			line_to_display = ""
			pos_x = 0
			for scalar in vector:
				line_to_display += " " + str(scalar)+" "
				if(scalar == untouched_char):
					untouched_position.append((pos_y,pos_x))
				pos_x += 1

			print line_to_display
			pos_y += 1

		print "="*23


		## return untouched position
		return untouched_position




def draw_variance_matrix(data_folder):
	"""
	Get the distribution of delta values for each slot of 
	the compensation matrix from all delta matrix, then compute
	variance of each distribution and plot an heatmap to highlight
	the spot with high variance
	"""

	## importation
	import glob
	import numpy
	import matplotlib
	import matplotlib.pyplot as plt


	## init distribution matrix
	position_to_distribution = {}
	for x in xrange(0,8):
		for y in xrange(0,8):			
			key = str(y)+"_"+str(x)
			position_to_distribution[key] = []


	## init variance matrix
	variance_matrix = []
	variance = "NA"
	for x in xrange(0,8):
		variance_matrix.append([])
		for y in xrange(0,8):
			variance_matrix[x].append(variance)


	## loop over all delta matrix
	delta_matrix_files = glob.glob(data_folder+"/*.txt")
	for delta_matrix in delta_matrix_files:

		data_file = open(delta_matrix, "r")
		pos_y = 0
		for line in data_file:

			line = line.rstrip()
			line_in_array = line.split(",")
			pos_x = 0
			
			for scalar in line_in_array:

				key = str(pos_y)+"_"+str(pos_x)
				position_to_distribution[key].append(float(scalar))
				pos_x += 1

			pos_y += 1
		data_file.close()

	## compute variance matrix
	for y in xrange(0,8):
		for x in xrange(0,8):
			key = str(x)+"_"+str(y)
			distribution = position_to_distribution[key]
			variance = numpy.var(distribution)
			variance_matrix[y][x] = variance


	## plot heatmap

	## get the label
	x_label = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	y_label = x_label

	## get the grid
	grid_to_display = numpy.asarray(variance_matrix)

	## plot the stuff
	fig, ax = plt.subplots()
	im = ax.imshow(grid_to_display)

	# We want to show all ticks...
	ax.set_xticks(numpy.arange(len(x_label)))
	ax.set_yticks(numpy.arange(len(y_label)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(x_label)
	ax.set_yticklabels(y_label)

	# Add colorbar, make sure to specify tick locations to match desired ticklabels
	cbar = fig.colorbar(im, ticks=[numpy.amin(grid_to_display), numpy.amax(grid_to_display)])
	cbar.ax.set_yticklabels([numpy.amin(grid_to_display), numpy.amax(grid_to_display)])  # vertically oriented colorbar


	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	ax.set_title("Matrix compensation variance")
	fig.tight_layout()
	plt.show()




def perform_linear_regression(channel_a, channel_b):
	"""
	
	Perform linear regression on specific slot for
	the compensation matrix

		=> not the best solution, uncompensated matrix are identical

	"""

	## impotation
	import glob
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn import datasets, linear_model
	from sklearn.metrics import mean_squared_error, r2_score


	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]

	## check the target slot
	target_x = "NA"
	target_y = "NA"

	index = 0
	for channel in channel_list:
		if(channel_a == channel):
			target_y = index
		if(channel_b == channel):
			target_x = index
		index += 1

	## get uncompensated coordinates
	uncompensated_vector = []
	uncompensated_files = glob.glob("data/matrix/uncompensated/*.txt")
	for uncompensated_matrix in uncompensated_files:
		cmpt = 0
		pos_y = 0
		data_file = open(uncompensated_matrix, "r")
		for line in data_file:
			if(cmpt != 0):
				line = line.rstrip()
				line_in_array = line.split(",")
				pos_x = 0
				index = 0
				for elt in line_in_array:
					if(index > 0):
						if(pos_x == target_x and pos_y == target_y):
							uncompensated_vector.append(elt)
						pos_x += 1
					index += 1
				pos_y += 1	

			cmpt += 1
		data_file.close()

	## get compensated coordinates
	compensated_vector = []
	compensated_files = glob.glob("data/matrix/compensated/*.txt")
	for compensated_matrix in compensated_files:
		cmpt = 0
		pos_y = 0
		data_file = open(compensated_matrix, "r")
		for line in data_file:
			if(cmpt != 0):
				line = line.rstrip()
				line_in_array = line.split("\t")
				pos_x = 0
				index = 0
				for elt in line_in_array:
					if(index > 1):
						if(pos_x == target_x and pos_y == target_y):
							compensated_vector.append(float(elt)/100.0)
						pos_x += 1
					index += 1
				pos_y += 1	

			cmpt += 1
		data_file.close()

	## Perform linear regression

	## split data into training an testing
	X_train = uncompensated_vector[:-50]
	X_test = uncompensated_vector[-50:]
		
	X_train_formated = []
	for scalar in X_train:
		X_train_formated.append([float(scalar)])
	X_test_formated = []
	for scalar in X_test:
		X_test_formated.append([float(scalar)])

	Y_train = compensated_vector[:-50]
	Y_test = compensated_vector[-50:]

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(X_train_formated, Y_train)

	# Make predictions using the testing set
	Y_pred = regr.predict(X_test_formated)

	# The coefficients
	print "Coefficients: " +str(regr.coef_)
	# The mean squared error
	print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(Y_test, Y_pred))


	"""
	# Plot outputs
	plt.scatter(X_test_formated, Y_test,  color='black')
	plt.plot(X_test_formated, Y_pred, color='blue', linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()
	"""





def get_custum_coeff(channel_a, channel_b):
	"""
	get custum score from the delta matrix	
	"""

	## impotation
	import glob
	import numpy


	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]

	## check the target slot
	target_x = "NA"
	target_y = "NA"

	index = 0
	for channel in channel_list:
		if(channel_a == channel):
			target_x = index
		if(channel_b == channel):
			target_y = index
		index += 1

	## get uncompensated coordinates
	delta_vector = []
	delta_files = glob.glob("data/matrix/delta/*.txt")
	for delta_matrix in delta_files:
		
		pos_y = 0
		data_file = open(delta_matrix, "r")
		for line in data_file:
			line = line.rstrip()
			line_in_array = line.split(",")
			pos_x = 0
			for elt in line_in_array:
				if(pos_x == target_x and pos_y == target_y):
					delta_vector.append(float(elt))
				pos_x += 1
			pos_y += 1	

		data_file.close()

	## Youhou ...
	score = numpy.mean(delta_vector)

	return score




def create_correction_values():
	"""
	Create custom correction values for all
	slot in compensation matrix

	return a dictionnary
	"""

	channel_to_correction = {}
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]

	for channel_a in channel_list:
		for channel_b in channel_list:

			key = str(channel_a)+"_"+str(channel_b)
			channel_to_correction[key] = get_custum_coeff(channel_a, channel_b)

	return channel_to_correction



def create_compensated_matrix(uncompensated_matrix_file, correction_values):
	"""
	
	Apply correction stored in correction_values to the uncompensated 
	matrix and write the results in a new predicted matrix file

	=> Might be a problem there

	"""

	## Init parameters
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	output_folder = "data/matrix/predicted"
	predicted_matrix_name = uncompensated_matrix_file.split("/")
	predicted_matrix_name = predicted_matrix_name[-1].replace("uncompensated", "predicted")
	predicted_matrix = []
	for x in xrange(0,8):
		predicted_vector = []
		for y in xrange(0,8):
			predicted_vector.append("NA")
		predicted_matrix.append(predicted_vector)

	## Parse original matrix
	input_data = open(uncompensated_matrix_file, "r")
	cmpt = 0
	pos_y = 0
	for line in input_data:
		if(cmpt > 0):
			line = line.rstrip()
			line_in_array = line.split(',')
			index = 0
			pos_x = 0 
			for scalar in line_in_array:
				if(index > 0):
					key = str(channel_list[pos_x]+"_"+channel_list[pos_y])
					predicted_value = float(scalar) + float(correction_values[key])
					predicted_matrix[pos_x][pos_y] = predicted_value
					pos_x += 1
				index += 1
			pos_y += 1
		cmpt += 1
	input_data.close()

	## Write predicted matrix
	predicted_matrix_data = open(output_folder+"/"+predicted_matrix_name, "w")
	header_line = "Autofl.\t\t"
	for channel in channel_list:
		header_line += channel+"\t"
	header_line = header_line[:-1]
	predicted_matrix_data.write(header_line+"\n")
	cmpt = 0	
	for vector in predicted_matrix:
		vector_in_line = "0\t"+channel_list[cmpt]+"\t"
		for scalar in vector:
			vector_in_line += str(float(scalar)*100.0)+"\t"
		vector_in_line = vector_in_line[:-1]
		if(cmpt < len(predicted_matrix)):
			predicted_matrix_data.write(vector_in_line+"\n")
		else:
			predicted_matrix_data.write(vector_in_line)
		cmpt += 1
	predicted_matrix_data.close()





def create_all_prediction():
	"""
	Create a prediction matrix for
	each uncompensated matrix found in the
	uncompsated fodler
	"""

	## importation
	import glob

	## get correction values
	correction_structure = create_correction_values()

	## Generate all the predictes matrix
	for uncompensated_matrix_file in glob.glob("data/matrix/uncompensated/*.txt"):
		create_compensated_matrix(uncompensated_matrix_file, correction_structure)






def create_delta_prediction_matrix(compensated_matrix, predicted_matrix, output_matrix):
	"""
	IN PROGRESS
	
	Scavange from compute orginal delta matrix

	"""

	## deal with the compensated matrix
	matrix_1_data = open(compensated_matrix, "r")
	matrix_1_value = []
	
	cmpt = 0
	for line in matrix_1_data:
		if(cmpt != 0):
			vector = []
			line = line.rstrip()
			line_in_array = line.split("\t")
			index = 0
			for scalar in line_in_array:
				if(index >= 2):
					vector.append(scalar)
				index += 1
			matrix_1_value.append(vector)
		cmpt += 1
	matrix_1_data.close()


	## deal with the predicted matrix
	matrix_2_data = open(predicted_matrix, "r")
	matrix_2_value = []
	cmpt = 0
	for line in matrix_2_data:
		line = line.rstrip()
		if(cmpt != 0):
			vector = []
			line_in_array = line.split("\t")
			
			index = 0
			for scalar in line_in_array:
				if(index >= 2):
					vector.append(float(scalar))
				index += 1
			matrix_2_value.append(vector)
		cmpt += 1
	matrix_2_data.close()


	## compute the delta matrix
	delta_matrix = []
	pos_y = 0
	for vector in matrix_1_value:
		delta_vector = []
		pos_x = 0
		for scalar in vector:
			compensated_scalar = matrix_2_value[pos_y][pos_x] 
			delta_scalar = float(compensated_scalar) - float(scalar)
			delta_vector.append(delta_scalar)
			pos_x += 1

		delta_matrix.append(delta_vector)
		pos_y += 1


	## write the delta matrix in a csv file
	delta_matrix_file = open(output_matrix, "w")
	cmpt = 0
	for vector in delta_matrix:
		line_to_write = ""
		for scalar in vector:
			line_to_write += str(scalar)+","
		line_to_write = line_to_write[:-1]
		if(cmpt < len(delta_matrix)):
			delta_matrix_file.write(line_to_write+"\n")
		else:
			delta_matrix_file.write(line_to_write)
		cmpt +=1
	delta_matrix_file.close()




def generate_all_delta_predicted_matrix():
	"""
	IN PROGRESS
	"""

	## importation
	import glob

	## get files list
	compensated_matrix_files = glob.glob("data/matrix/compensated/*.txt")
	predicted_matrix_files = glob.glob("data/matrix/predicted/*.txt")

	output_directory = "data/matrix/delta_predicted/"
	found_cmpt = 0

	for predicted_matrix in predicted_matrix_files:

		## parse compensated matrix file name
		predicted_matrix_name = predicted_matrix.split("/")
		predicted_matrix_name_in_array = predicted_matrix_name[-1].split("_")
		predicted_matrix_panel = predicted_matrix_name_in_array[1]
		predicted_matrix_center = predicted_matrix_name_in_array[2]
		predicted_matrix_ID = predicted_matrix_name_in_array[3]
		predicted_matrix_ID = predicted_matrix_ID.split(".")
		predicted_matrix_ID = predicted_matrix_ID[0]

		for compensated_matrix in compensated_matrix_files:

			## parse uncompensated matrix file name
			compensated_matrix_name = compensated_matrix.split("/")
			compensated_matrix_name_in_array = compensated_matrix_name[-1].split("_")
			compensated_matrix_panel = compensated_matrix_name_in_array[1]
			compensated_matrix_center = compensated_matrix_name_in_array[2]
			compensated_matrix_ID = compensated_matrix_name_in_array[3].split(".")
			compensated_matrix_ID = compensated_matrix_ID[0]		


			

			## test the files, get the "goods one"
			if(compensated_matrix_center == predicted_matrix_center and compensated_matrix_panel == predicted_matrix_panel and compensated_matrix_ID == predicted_matrix_ID):

				delta_matrix = output_directory+str(compensated_matrix_name[-1])
				delta_matrix = delta_matrix.split(".")
				delta_matrix = str(delta_matrix[0])+"_delta."+str(delta_matrix[1])

				create_delta_prediction_matrix(compensated_matrix, predicted_matrix, delta_matrix)






def create_data_file_for_linear_regression(data_folder, output_file):
	"""
	IN PROGRESS
	"""

	## importation
	import glob


	## open output file
	output_data = open(output_file, "w")

	## get all data files
	data_files = glob.glob(data_folder+"/*.txt")

	## write header
	variable_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	header = ""
	for var in variable_list:
		for var2 in variable_list:
			header += var+"-"+var2+","
	header = header[:-1]
	output_data.write(header+"\n")

	## get data, assume matrix files are compensated matrix file
	cmpt = 0
	for data_file in data_files:
		line_to_write = ""
		cmpt = 0
		data = open(data_file, "r")
		for line in data:
			line = line.rstrip()
			line_in_array = line.split("\t")
			if(cmpt > 0):
				index =0
				for elt in line_in_array:
					if(index > 1):
						line_to_write += str(elt)+","
					index += 1
			cmpt += 1
		data.close()
		line_to_write = line_to_write[:-1]
		if(cmpt < len(data_files)):
			output_data.write(line_to_write+"\n")
		else:
			output_data.write(line_to_write)
		cmpt += 1

	## close output file
	output_data.close()


	##-------------##
	## IN PROGRESS ##
	## write extended data files
	extended_data_file_name = output_file.split(".")
	extended_data_file_name = extended_data_file_name[0]+"_extended.csv"

	## get list of patients
	patient_list = []
	for data_file in data_files:
		data_file_in_array = data_file.split("/")
		data_file_in_array = data_file_in_array[-1].split("_")
		patient_tag = "Panel_"+str(data_file_in_array[1])+"_"+str(data_file_in_array[2])+"_"+str(data_file_in_array[3])
		patient_tag = patient_tag.replace(".txt", "")
		if(patient_tag not in patient_list):
			patient_list.append(patient_tag)

	extended_data_file = open(extended_data_file_name, "w")


	## write header
	variable_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	header = ""
	for var in variable_list:
		for var2 in variable_list:
			header += var+"-"+var2+","	
	for var in variable_list:
		for var2 in variable_list:
			header += var+"-"+var2+"-original,"
	header = header[:-1]

	extended_data_file.write(header+"\n")

	patient_cmpt = 0
	for patient in patient_list:

		line_to_write = ""
		compensated_matrix_file = "data/matrix/compensated/"+patient+".txt"
		uncompensated_matrix_file = "data/matrix/uncompensated/"+patient+"_uncompensated.txt"

		compensated_data = open(compensated_matrix_file, "r")
		cmpt = 0
		for line in compensated_data:
			line = line.rstrip()
			line_in_array = line.split("\t")
			if(cmpt > 0):
				index =0
				for elt in line_in_array:
					if(index > 1):
						line_to_write += str(elt)+","
					index += 1
			cmpt += 1
		compensated_data.close()

		uncompensated_data = open(uncompensated_matrix_file, "r")
		cmpt = 0
		for line in uncompensated_data:
			line = line.rstrip()
			line_in_array = line.split(",")
			if(cmpt > 0):
				index =0
				for elt in line_in_array:
					if(index > 0):
						line_to_write += str(float(elt)*100)+","
					index += 1
			cmpt += 1
		uncompensated_data.close()

		## write the line
		line_to_write = line_to_write[:-1]
		if(patient_cmpt < len(patient_list)):
			extended_data_file.write(line_to_write+"\n")
		else:
			extended_data_file.write(line_to_write)

		patient_cmpt += 1

	extended_data_file.close()







## CORRECTIONS BY MODULAR STAGE ##



def init_channel_to_correction():
	"""
	Create the correction matrix structure,
	a dictionnary where each channels have 3 values:

		- S1 : the Stage 1 correction
		- S2 : the Stage 2 correction
		- Corrected : boolean status for the channel
		- Trusted : boolean status, use for stage 2 correction
	
	Write the structure in a pkl file : matrix_correction.pkl 
	"""

	## importation
	import pickle

	## Create the structure
	channel_to_correction = {}
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	for channel_a in channel_list:
		for channel_b in channel_list:

			key = str(channel_a)+"-"+str(channel_b)
			channel_to_correction[key] = {"S1":0,"S2":1, "Corrected":False, "Trusted":False}


	## save the correction structure
	with open('matrix_corrections.pkl', 'wb') as f:
		pickle.dump(channel_to_correction, f, pickle.HIGHEST_PROTOCOL)



def satge_1_correction(trusted_positions):
	"""
	Load correction structure and apply Stage 1 correction
	by using the get_custom_coeff function

	Stage 1 is the simpliest possible correction, mean of differences
	between the uncompensated and the manually compensated matrix in the
	training set, used only on "trusted positions", i.e position where the distribution
	of correction has a low variance.

	Save the updated correction structure in a pkl file : matrix_correction.pkl
	"""

	## Importation
	import pickle

	## Load correction structure
	with open('matrix_corrections.pkl', 'rb') as f:
		correction_structure = pickle.load(f)

	## Compute correction for trusted channels
	for key_channel in correction_structure.keys():
		if(key_channel in trusted_positions):
			channel_in_array = key_channel.split("-")
			channel_1 = channel_in_array[0]
			channel_2 = channel_in_array[1]
			correction_structure[key_channel]["S1"] = get_custum_coeff(channel_1, channel_2)
			correction_structure[key_channel]["Corrected"] = True
			correction_structure[key_channel]["Trusted"] = True

	## save the correction structure
	with open('matrix_corrections.pkl', 'wb') as f:
		pickle.dump(correction_structure, f, pickle.HIGHEST_PROTOCOL)

	


def stage_2_correction(trusted_positions, data_file):
	"""
	Perform stage 2 correction by using multiple linear regression to predict
	the value of each channel from the "trusted channel" identify after stage 1
	correction

	data for linear regression came from data_file

	update correction structure and save it in the matrix_corrections.pkl file

	"""

	## importation
	import pickle
	import pandas as pd
	import statsmodels.api as sm

	## parameters
	rsquarred_treshold = 0.75
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	target_list = []

	## Load correction structure
	with open('matrix_corrections.pkl', 'rb') as f:
		correction_structure = pickle.load(f)

	## Compute stage 2 correction
	df = pd.read_csv(data_file)
	target = pd.read_csv(data_file)
	for channel_1 in channel_list:
		for channel_2 in channel_list:
			if(channel_2 != channel_1):
				target_name = str(channel_1)+"-"+str(channel_2)
				if(target_name not in target_list):
					target_list.append(target_name)

	for target_name in target_list:
	
		## Select only trusted values
		X = df[trusted_positions]

		## set target
		y = target[target_name]

		## make it real
		model = sm.OLS(y, X).fit()
		predictions = model.predict(X)

		## if rsquared above trehsold keep the target variable and
		if(str(model.rsquared) != "nan"):
			if(float(model.rsquared) >= rsquarred_treshold):

				## parse coefficients
				coeff = model.params
				coeff = str(coeff)
				coeff = coeff.replace(" ", ";")
				coeff = coeff.split("\n")
				coeff = coeff[:-1]
				coeff_list = []
				for elt in coeff:
					elt_in_array = elt.split(";")
					parsed_coeff = str(elt_in_array[0]) + ";" + str(elt_in_array[-1])
					coeff_list.append(parsed_coeff)

				## update correction structure
				correction_structure[target_name]["S2"] = coeff_list
				correction_structure[target_name]["Corrected"] = True
				correction_structure[target_name]["Trusted"] = True

				## reset stage 1 connection ? 
				correction_structure[target_name]["S1"] = 0

	## save the correction structure
	with open('matrix_corrections.pkl', 'wb') as f:
		pickle.dump(correction_structure, f, pickle.HIGHEST_PROTOCOL)





def correct_compensation_matrix(uncompensated_matrix_file):
	"""
	IN PROGRESS

	Some very stange bug around trust_order and high_trusted_channels variables
	Fix the problem by declare original_trusted_channels


	Stage 2 seems to be a problem


	TODO : 
		-> function to get list of igh trusted channels
	
	"""

	## importation
	import pickle

	##--------------------##
	## Compute correction ##
	##--------------------##

	## parameters
	high_trusted_channels = ["PE.A-PC7.A","PE.A-PB.A","PC7.A-PB.A"]
	original_trusted_channels = ["PE.A-PC7.A","PE.A-PB.A","PC7.A-PB.A"]
	low_trusted_channels = []
	trust_order = []
	data_file = "data_reg_test_extended.csv"
	number_of_trusted_channel = len(high_trusted_channels)
	total_number_of_channels = 64
	authorized_to_perform_stage_2 = True
	channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	output_folder = "data/matrix/predicted"
	predicted_matrix_name = uncompensated_matrix_file.split("/")
	predicted_matrix_name = predicted_matrix_name[-1].replace("uncompensated", "predicted")

	## init structure
	init_channel_to_correction()
	#trust_order.append(high_trusted_channels)

	## Perform Stage 1
	satge_1_correction(high_trusted_channels)

	## Loop on Stage 2 correction while it's possible
	while(authorized_to_perform_stage_2):
		stage_2_correction(high_trusted_channels, data_file)	

		## Load correction structure
		with open('matrix_corrections.pkl', 'rb') as f:
			correction_structure = pickle.load(f)

		new_trusted_channels = []
		for channel in correction_structure.keys():
			if(correction_structure[channel]["Trusted"] and channel not in high_trusted_channels):
				new_trusted_channels.append(channel)

		## update highly trusted channel
		high_trusted_channels += new_trusted_channels		


		if(len(new_trusted_channels) == 0):
			authorized_to_perform_stage_2 = False
		else:
			trust_order.append(new_trusted_channels)



	## Perform a last stage 1 correction
	## if necessary
	with open('matrix_corrections.pkl', 'rb') as f:
		correction_structure = pickle.load(f)
	target_list = []
	for channel in correction_structure.keys():
		if(correction_structure[channel]["Corrected"] == False):
			target_list.append(channel)

	if(len(target_list) > 0):
		satge_1_correction(target_list)
		low_trusted_channels = target_list

	##---------------------##
	## Generate New Matrix ##
	##---------------------##

	## load corrections
	with open('matrix_corrections.pkl', 'rb') as f:
		correction_structure = pickle.load(f)	
	
	## Prepare predicted matrix
	predicted_matrix = []
	for x in xrange(0,8):
		predicted_vector = []
		for y in xrange(0,8):
			predicted_vector.append("NA")
		predicted_matrix.append(predicted_vector)

	## Prepare original matrix
	original_matrix = []
	for x in xrange(0,8):
		original_vector = []
		for y in xrange(0,8):
			original_vector.append("NA")
		original_matrix.append(original_vector)

	## Compute channel to position on the grid
	channel_to_position = {}
	pos_y = 0
	for channel_y in channel_list:
		original_vector = []
		pos_x = 0
		for channel_x in channel_list:
			key = str(channel_y)+"-"+str(channel_x)
			channel_to_position[key] = {"y":pos_y, "x":pos_x}

			pos_x += 1
		pos_y +=1

	## Parse original matrix
	input_data = open(uncompensated_matrix_file, "r")
	cmpt = 0
	pos_y = 0
	for line in input_data:
		if(cmpt > 0):
			line = line.rstrip()
			line_in_array = line.split(',')
			index = 0
			pos_x = 0 
			for scalar in line_in_array:
				if(index > 0):
					original_matrix[pos_y][pos_x] = float(scalar)
					pos_x += 1
				index += 1
			pos_y += 1
		cmpt += 1
	input_data.close()


	## Apply first stage 1 correction from the
	## first trusted values of the process
	for key_channel in correction_structure.keys():
		if(key_channel in original_trusted_channels):
			stage_1_coeff = correction_structure[key_channel]["S1"]			
			original_value = original_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]]
			predicted_value = float(original_value) + float(stage_1_coeff)
			predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] = predicted_value
	

	## Apply stage 2 corrections

	## Apply correction obtained from original_trusted_value
	for key_channel in correction_structure.keys():

		stage_2_coeff = correction_structure[key_channel]["S2"]
		
		## get all coeff
		if(stage_2_coeff != 1):
			channel_to_coeff = {}
			for coeff in stage_2_coeff:
				coeff_in_array = coeff.split(";")
				channel_to_coeff[coeff_in_array[0]] = coeff_in_array[1]

			for coeff_channel in channel_to_coeff.keys():
				if(coeff_channel in original_trusted_channels):
					if(predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] == "NA"):
						predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] = 0
					correction = float(predicted_matrix[channel_to_position[coeff_channel]["y"]][channel_to_position[coeff_channel]["x"]]) * float(channel_to_coeff[coeff_channel])
					predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] += correction


	## Apply stage 2 correction on the new trusted variables	
	for newly_trusted_channels in trust_order:
		for key_channel in correction_structure.keys():

			stage_2_coeff = correction_structure[key_channel]["S2"]
			
			## get all coeff
			if(stage_2_coeff != 1):
				channel_to_coeff = {}
				for coeff in stage_2_coeff:
					coeff_in_array = coeff.split(";")
					channel_to_coeff[coeff_in_array[0]] = coeff_in_array[1]
			
				for coeff_channel in channel_to_coeff.keys():
					if(coeff_channel in new_trusted_channels):
						if(predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] == "NA"):
							predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] = 0
						correction = float(predicted_matrix[channel_to_position[coeff_channel]["y"]][channel_to_position[coeff_channel]["x"]]) * float(channel_to_coeff[coeff_channel])
						predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] += correction


	## Apply stage 1 correction on anything left
	for key_channel in correction_structure.keys():
		if(key_channel in low_trusted_channels):
			stage_1_coeff = correction_structure[key_channel]["S1"]			
			original_value = original_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]]
			predicted_value = float(original_value) + float(stage_1_coeff)
			predicted_matrix[channel_to_position[key_channel]["y"]][channel_to_position[key_channel]["x"]] = predicted_value


	## Write predicted matrix
	predicted_matrix_data = open(output_folder+"/"+predicted_matrix_name, "w")
	header_line = "Autofl.\t\t"
	for channel in channel_list:
		header_line += channel+"\t"
	header_line = header_line[:-1]
	predicted_matrix_data.write(header_line+"\n")
	cmpt = 0	
	for vector in predicted_matrix:
		vector_in_line = "0\t"+channel_list[cmpt]+"\t"
		for scalar in vector:
			vector_in_line += str(float(scalar)*100.0)+"\t"
		vector_in_line = vector_in_line[:-1]
		if(cmpt < len(predicted_matrix)):
			predicted_matrix_data.write(vector_in_line+"\n")
		else:
			predicted_matrix_data.write(vector_in_line)
		cmpt += 1
	predicted_matrix_data.close()




def write_report(fcs_file):
	"""
	write an html report file to compare the uncompensated
	and the auto compensated images

	Todo : compare with manually compensated images

	"""

	## importation
	import glob
	import FlowCal


	## parameters
	entry_list = []
	uncompensated_images = glob.glob("output/uncompensated_images/*.png")
	predicted_images = glob.glob("output/predicted_images/*.png")
	mannualy_compensated_images = glob.glob("output/compensated_images/*.png")
	report_file_name = "report.html"
	

	## get channel list
	#channel_list = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	s = FlowCal.io.FCSData(fcs_file)
	s = FlowCal.transform.to_rfi(s)
	channel_list = s.channels

	images_type_list = ["all"]
	compare_with_manually_compensated_file = True


	## IN PROGRESSc
	compensation_matrix_compensated = "NA"
	compensation_matrix_uncompensated = "NA"
	compensation_matrix_predicted = "NA"




	## get list of image type
	for channel_1 in channel_list:
		for channel_2 in channel_list:
			if(channel_1 != channel_2):
				image_tag = channel_1+"_"+channel_2
				reverse_tag = channel_2+"_"+channel_1
				if(image_tag not in images_type_list and reverse_tag not in images_type_list):
					images_type_list.append(image_tag)




	## get list of entry
	for image_file in uncompensated_images:
		image_file_in_array = image_file.split("/")
		image_file_in_array = image_file_in_array[-1].split(".")
		image_file_in_array = image_file_in_array[0].split("_")
		file_panel = image_file_in_array[1]
		file_id = image_file_in_array[2]
		file_center = image_file_in_array[3]
		patient_tag = "Panel_"+str(file_panel)+"_"+str(file_id)+"_"+str(file_center)
		if(patient_tag not in entry_list):
			entry_list.append(patient_tag)


		#compensation_matrix_compensated = "data/matrix/compensated/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+".txt"
		#compensation_matrix_uncompensated = "data/matrix/uncompensated/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+"_uncompensated.txt"
		#compensation_matrix_predicted = "data/matrix/predicted/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+"_predicted.txt"


	## Write report
	report_file = open(report_file_name, "w")

	report_file.write("<html>\n")
	report_file.write("<head>\n\t<title>Automatic Compensation</title>\n</head>\n")
	report_file.write("<body>\n")
	report_file.write("\t<h1>Automatic Compensation</h1>\n")

	## Generate grid matrix heatmap
	#plot_compensation_matrix(compensation_matrix_uncompensated)
	#plot_compensation_matrix(compensation_matrix_predicted)
	image_grid_uncompensated = "results/images/compensation_matrix_heatmap_uncompensated.png"
	image_grid_predicted = "results/images/compensation_matrix_heatmap_predicted.png"
	if(compare_with_manually_compensated_file):
		#plot_compensation_matrix(compensation_matrix_compensated)
		image_grid_compensated = "results/images/compensation_matrix_heatmap_compensated.png"

	## write matrix comparison in report
	report_file.write("\t<h2>Matrix Comparison</h2>\n")
	report_file.write("\t<img src=\""+str(image_grid_uncompensated)+"\">\n")
	report_file.write("\t<img src=\""+str(image_grid_predicted)+"\">\n")

	if(compare_with_manually_compensated_file):
		report_file.write("\t<img src=\""+str(image_grid_compensated)+"\">\n")




	for tag in entry_list:
		report_file.write("\t<h2>"+str(tag)+"</h2>\n")

		for image_tag in images_type_list:
			report_file.write("\t<h3>"+str(image_tag)+"</h3>\n")

			uncompensated_image = ""
			for image in uncompensated_images:
				if(image_tag in image):
					uncompensated_image = image

			predicted_image = ""
			for image in predicted_images:
				if(image_tag in image):
					predicted_image = image


			report_file.write("\t<img src=\""+str(uncompensated_image)+"\">\n")
			report_file.write("\t<img src=\""+str(predicted_image)+"\">\n")

			if(compare_with_manually_compensated_file):

				mannualy_image = ""
				for image in mannualy_compensated_images:
					if(image_tag in image):
						mannualy_image = image

				report_file.write("\t<img src=\""+str(mannualy_image)+"\">\n")

	report_file.write("</body>\n")
	report_file.write("</html>")
	report_file.close()



def extract_compensation_from_uncompensated_fcs_files(fcs_folder):
	"""
	IN PROGRESS
	"""

	## importation
	import glob
	import os

	
	## parameters
	EXTRACTMATRIX_SCRIPT = "extractMatrix.R"

	for uncomp_fcs in glob.glob(fcs_folder+"/*.fcs"):
		os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(uncomp_fcs))





def compute_matrix_distance(matrix_1_file, matrix_2_file):
	"""
	Compute distance betwwen a predicted and a computed matrix
	"""

	## parameters
	global_distance = 0

	matrix_1 = open(matrix_1_file, "r")
	cmpt_1 = 0
	for line_1 in matrix_1:

		cmpt_2 = 0
		matrix_2 = open(matrix_2_file, "r")
		for line_2 in matrix_2:
			
			if(cmpt_1 == cmpt_2):

				line_1 = line_1.rstrip()
				line_2 = line_2.rstrip()

				line_1_in_array = line_1.split("\t")
				line_2_in_array = line_2.split("\t")
				if(cmpt_1 > 0):
					index = 0
					for elt_1 in line_1_in_array:
						if(index > 1):
							elt_2 = line_2_in_array[index]
							distance = float(elt_2) - float(elt_1)
							global_distance += abs(distance)
						index += 1
			cmpt_2 += 1

		matrix_2.close()
		cmpt_1 += 1

	matrix_1.close()

	return global_distance




def compensation_tool(fcs_file, training_folder_fcs_uncomp, training_folder_fcs_comp, training_folder_matrix_comp):
	"""
	
	Compensation tool, V1

	A lot of stuff in progress

	-> fcs_file must be an absolute path

	"""

	## importation
	import os
	import shutil

	## parameters
	EXTRACTMATRIX_SCRIPT = "extractMatrix.R"
	GENERATEIMAGE_SCRIPT = "CreateImageFromFCS.R"
	APPLYCOMPENSATION_SCRIPT = "ApplyCompensation.R"
	uncompensated_matrix_file_name = fcs_file.split("/")
	uncompensated_matrix_file_name = uncompensated_matrix_file_name[-1].split("_")
	




	output_folder = "data/matrix/uncompensated"
	Panel = uncompensated_matrix_file_name[1]
	ID = uncompensated_matrix_file_name[2]
	center = uncompensated_matrix_file_name[3]
	uncompensated_matrix_file_name = output_folder+"/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+"_uncompensated.txt"
	predicted_matrix_file_name = uncompensated_matrix_file_name.replace("uncompensated", "predicted")
	manually_compensated_matrix_file = "data/matrix/compensated/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+".txt"	

	image_uncomp_folder = "output/uncompensated_images"
	image_predict_folder = "output/predicted_images"
	image_manually_folder = "output/compensated_images"
	compare_with_manually_compensated_file = True
	

	use_partial_dataset = True


	fcs_file_to_predict_in_array = fcs_file.split("/")
	fcs_file_to_predict = "input/uncompensated/"+fcs_file_to_predict_in_array[-1]
	


	## Load data files
	print "[Step 1][+] Load data files" 
	load_data_for_training(training_folder_fcs_uncomp, training_folder_fcs_comp, training_folder_matrix_comp)
	os.system("cp "+str(fcs_file)+ " input/uncompensated/")
	fcs_file = fcs_file_to_predict
	predicted_fcs = fcs_file+"_comp.fcs"
	compensated_fcs = fcs_file+"_comp.fcs"

	## extract compensation from fcs in the training folder
	## generate uncomp matrix
	print "[Step 2][+] Extract compensation matrix from uncompensated files for training"
	extract_compensation_from_uncompensated_fcs_files(training_folder_fcs_uncomp)

	## generate training dataset
	print "[Step 3][+] Create training file" 
	create_data_file_for_linear_regression("data/matrix/compensated", "data_reg_test.csv")
	if(use_partial_dataset):
		randomy_split_dataset("data_reg_test.csv", 0.3)
		shutil.copy("data_reg_test_splited_1.csv", "data_reg_test.csv")


	## extract compensation from fcs_file
	print "[Step 4][+] Extract compensation matrix from target fcs file"
	os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(fcs_file))

	## compute correction matrix
	print "[Step 5][+] Create correction values" 
	generate_all_delta_matrix()
	correction_values = create_correction_values()


	## Generate image from fcs file
	print "[Step 6][+] Generate images for uncompensated files"
	#generate_image_from_fcs(str(fcs_file))
	#os.system("mv input/uncompensated/*.png "+str(image_uncomp_folder+"/"))

	## compute compensated matrix
	print "[Step 7][+] Compute new compensation matrix"
	print uncompensated_matrix_file_name
	create_compensated_matrix(uncompensated_matrix_file_name, correction_values)
	#correct_compensation_matrix(uncompensated_matrix_file_name)
	
	## test stupid idea
	os.system("cp "+uncompensated_matrix_file_name+" "+predicted_matrix_file_name)
	transpose_matrix_file_dev(predicted_matrix_file_name)
	
	## apply compensated matrix
	print "[Step 8][+] Apply new compensation"
	os.system("Rscript "+APPLYCOMPENSATION_SCRIPT+" "+str(fcs_file)+" "+str(predicted_matrix_file_name))
	os.system("mv "+str(predicted_fcs)+" input/predicted/"+fcs_file_to_predict_in_array[-1]+"_comp.fcs")
	predicted_fcs = "input/predicted/"+fcs_file_to_predict_in_array[-1]+"_comp.fcs"

	## generate image from compensated fcs file
	print "[Step 9][+] Generate images for predicted files"
	#generate_image_from_fcs(str(predicted_fcs))
	#os.system("mv input/predicted/*.png "+str(image_predict_folder+"/"))

	print "[Step 9.1][+] Generate images for predicted and uncompnsated matrixes"
	plot_compensation_matrix(uncompensated_matrix_file_name)
	plot_compensation_matrix(predicted_matrix_file_name)

	## generate image for mannualy compensated file if option is set to
	## True
	if(compare_with_manually_compensated_file):
		pass

		## apply compensated matrix
		print "[Step 10.1][+] Apply mannual compensation"
		os.system("Rscript "+APPLYCOMPENSATION_SCRIPT+" "+str(fcs_file)+" "+str(manually_compensated_matrix_file))
		os.system("mv "+str(compensated_fcs)+" input/compensated/"+fcs_file_to_predict_in_array[-1]+"_comp.fcs")
		compensated_fcs = "input/compensated/"+fcs_file_to_predict_in_array[-1]+"_comp.fcs"

		print "[Step 10.2][+] Generate images for manually compensated files"
		#generate_image_from_fcs(str(compensated_fcs))
		#os.system("mv input/compensated/*.png "+str(image_manually_folder+"/"))

		print "[Step 10.3][+] Generate images for manually compensated matrix"
		plot_compensation_matrix(manually_compensated_matrix_file)

	## create graphical output
	print "[Step 11][+] Write report"
	write_report(fcs_file)

	print "[Step 12][*] EOF"












################
## TEST SPACE ##########################################################################################
################

try_compensation_tool = False
generate_data_file = False
try_modular_compensation = False


if(try_compensation_tool):

	##***********************##
	## try compensation tool ##
	##***********************##

	input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs"
	input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_42_TEST_blablabla.fcs"
	input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/input/Panel_1_32161133_UBO_NAVIOS_12OCT2017_12OCT2017.LMD_intra.fcs"
	
	

	#training_folder = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/CHP_1"
	#input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/CHP_1/fcs_compensated/Panel_1_32150409_CHP_NAVIOS_22APR2015_22APR2015.LMD_intra.fcs_comp.fcs"
	
	training_folder = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/MHH_1"
	#input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/MHH_1/fcs_uncompensated/Panel_1_3215_MHH_CANTOII_05NOV2015_05NOV2015.fcs"

	input_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/MHH_1/fcs_uncompensated/Panel_1_32160674_MHH_CANTOII_08NOV2016_08NOV2016.fcs"


	fcs_uncomp_folder = training_folder+"/fcs_uncompensated"
	fcs_comp_folder = training_folder+"/fcs_compensated"
	matrix_comp_folder = training_folder+"/matrix_compensated"


	#compensation_tool(input_fcs_file, "data_save/fcs/raw", "data_save/fcs/compensated", "data_save/matrix/compensated")
	compensation_tool(input_fcs_file, fcs_uncomp_folder, fcs_comp_folder, matrix_comp_folder)


if(generate_data_file):
	create_data_file_for_linear_regression("data/matrix/compensated", "data_reg_test.csv")


	input_data_file = "data_reg_test.csv"
	proportion = 0.8
	randomy_split_dataset(input_data_file, proportion)


if(try_modular_compensation):

	##******************************##
	## Play with modular correction ##
	##******************************##

	## importation
	import os

	EXTRACTMATRIX_SCRIPT = "extractMatrix.R"

	#high_trusted_channels = ["PE.A-PC7.A","PE.A-PB.A","PC7.A-PB.A"]
	#data_file = "data_reg_test_extended.csv"


	## Load data files
	uncompensated_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/MHH_1/fcs_uncompensated/Panel_1_32160674_MHH_CANTOII_08NOV2016_08NOV2016.fcs"
	compensated_fcs_file = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_test/MHH_1/fcs_compensated/Panel_1_32160674_MHH_CANTOII_08NOV2016_08NOV2016.fcs_intra.fcs_comp.fcs"

	uncompensated_matrix_file_name = uncompensated_fcs_file.split("/")
	uncompensated_matrix_file_name = uncompensated_matrix_file_name[-1].split("_")
	compensated_matrix_file_name = compensated_fcs_file.split("/")
	compensated_matrix_file_name = compensated_matrix_file_name[-1].split("_")
	output_folder = "data/matrix/uncompensated"
	Panel = uncompensated_matrix_file_name[1]
	ID = uncompensated_matrix_file_name[2]
	center = uncompensated_matrix_file_name[3]
	uncompensated_matrix_file_name = output_folder+"/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+"_uncompensated.txt"
	predicted_matrix_file_name = uncompensated_matrix_file_name.replace("uncompensated", "predicted")
	compensated_matrix_file_name = "data/matrix/compensated/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+".txt"	


	## extract compensation from uncompensated fcs_file
	print "[+] Extract compensation matrix from uncompensated fcs file"
	os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(uncompensated_fcs_file))
	

	## extract compensation from uncompensated fcs_file
	print "[+] Extract compensation matrix from compensated fcs file"
	os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(compensated_fcs_file))
	


	#correction_values = init_channel_to_correction()
	#satge_1_correction(high_trusted_channels)
	#stage_2_correction(high_trusted_channels, data_file)

	correct_compensation_matrix(uncompensated_matrix_file_name)
	plot_compensation_matrix(predicted_matrix_file_name)
	plot_compensation_matrix(compensated_matrix_file_name)







def test_training_dataset():
	"""
	IN PROGRESS
	"""


	## importation
	import glob
	import random
	import os


	for training_folder in glob.glob("data_test/*"):


		training_folder_name = training_folder.split("/")
		training_folder_name = training_folder_name[-1]
		fcs_uncomp_folder = training_folder+"/fcs_uncompensated"
		fcs_comp_folder = training_folder+"/fcs_compensated"
		matrix_comp_folder = training_folder+"/matrix_compensated"
		uncompensated_fcs_files = glob.glob(fcs_uncomp_folder+"/*.fcs")
		fcs_file_to_predict = uncompensated_fcs_files[random.randint(0,len(uncompensated_fcs_files)-1)]

		print "[RUNNING] => "+str(training_folder_name)

		## run the beast
		try:
			compensation_tool(fcs_file_to_predict, fcs_uncomp_folder, fcs_comp_folder, matrix_comp_folder)

			## Generate figures

			fcs_file_name = fcs_file_to_predict.split("/")
			fcs_file_name = fcs_file_name[-1].split("_")
			file_panel = fcs_file_name[1]
			file_center = fcs_file_name[3]
			file_id = fcs_file_name[2]

			compensation_matrix_compensated = "data/matrix/compensated/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+".txt"
			compensation_matrix_uncompensated = "data/matrix/uncompensated/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+"_uncompensated.txt"
			compensation_matrix_predicted = "data/matrix/predicted/Panel_"+str(file_panel)+"_"+str(file_center)+"_"+str(file_id)+"_predicted.txt"
			
			plot_compensation_matrix(compensation_matrix_uncompensated)
			plot_compensation_matrix(compensation_matrix_predicted)
			plot_compensation_matrix(compensation_matrix_compensated)

			## save results
			os.system("cp results/images/compensation_matrix_heatmap_compensated.png loop_results_2/matrix/"+str(training_folder_name)+"_compensation_matrix_heatmap_compensated.png")
			os.system("cp results/images/compensation_matrix_heatmap_uncompensated.png loop_results_2/matrix/"+str(training_folder_name)+"_compensation_matrix_heatmap_uncompensated.png")
			os.system("cp results/images/compensation_matrix_heatmap_predicted.png loop_results_2/matrix/"+str(training_folder_name)+"_compensation_matrix_heatmap_predicted.png")

			#os.system("cp output/compensated_images/*.png loop_results_2/compensated_images/")
			#os.system("cp output/uncompensated_images/*.png loop_results_2/uncompensated_images/")
			#os.system("cp output/predicted_images/*.png loop_results_2/predicted_images/")

		except:
			pass


		## clean everything
		os.system("./clean.sh")


















def evaluate_compensation_precision():
	"""
	IN PROGRESS
	"""


	## importation
	import glob
	import os
	import shutil

	## parameters
	EXTRACTMATRIX_SCRIPT = "extractMatrix.R"
	output_folder = "data/matrix/uncompensated"
	use_partial_dataset = True

	log_file = open("results/distances_between_matrix_transposition_only.log", "w")


	## loop over training folders
	#for training_folder in glob.glob("data_test/*"):

	for x in xrange(0,1): ## UBO SPECIAL

		training_folder = "/home/glorfindel/Spellcraft/SIDEQUEST/compensation/data_save"

		training_folder_name = training_folder.split("/")
		training_folder_name = training_folder_name[-1]
		
		#training_folder_fcs_uncomp = training_folder+"/fcs_uncompensated"
		#training_folder_fcs_comp = training_folder+"/fcs_compensated"
		#training_folder_matrix_comp = training_folder+"/matrix_compensated"
		

		## UBO SPECIAL
		training_folder_fcs_uncomp = training_folder+"/fcs/raw"
		training_folder_fcs_comp = training_folder+"/fcs/compensated"
		training_folder_matrix_comp = training_folder+"/matrix/compensated"

		uncompensated_fcs_files = glob.glob(training_folder_fcs_uncomp+"/*.fcs")

		for fcs_file in uncompensated_fcs_files:

			## clean everything
			os.system("./clean.sh")

			uncompensated_matrix_file_name = fcs_file.split("/")
			uncompensated_matrix_file_name = uncompensated_matrix_file_name[-1].split("_")
			Panel = uncompensated_matrix_file_name[1]
			ID = uncompensated_matrix_file_name[2]
			center = uncompensated_matrix_file_name[3]
			uncompensated_matrix_file_name = output_folder+"/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+"_uncompensated.txt"
			predicted_matrix_file_name = uncompensated_matrix_file_name.replace("uncompensated", "predicted")
			manually_compensated_matrix_file = "data/matrix/compensated/Panel_"+str(Panel)+"_"+str(center)+"_"+str(ID)+".txt"

			fcs_file_to_predict_in_array = fcs_file.split("/")
			fcs_file_to_predict = "input/uncompensated/"+fcs_file_to_predict_in_array[-1]	


			## Load data files
			print "[Step 1][+] Load data files" 
			load_data_for_training(training_folder_fcs_uncomp, training_folder_fcs_comp, training_folder_matrix_comp)
			os.system("cp "+str(fcs_file)+ " input/uncompensated/")
			fcs_file = fcs_file_to_predict
			predicted_fcs = fcs_file+"_comp.fcs"
			compensated_fcs = fcs_file+"_comp.fcs"

			## extract compensation from fcs in the training folder
			## generate uncomp matrix
			print "[Step 2][+] Extract compensation matrix from uncompensated files for training"
			extract_compensation_from_uncompensated_fcs_files(training_folder_fcs_uncomp)

			## generate training dataset
			print "[Step 3][+] Create training file" 
			create_data_file_for_linear_regression("data/matrix/compensated", "data_reg_test.csv")
			if(use_partial_dataset):
				randomy_split_dataset("data_reg_test.csv", 0.3)
				shutil.copy("data_reg_test_splited_1.csv", "data_reg_test.csv")

			## extract compensation from fcs_file
			print "[Step 4][+] Extract compensation matrix from target fcs file"
			os.system("Rscript "+EXTRACTMATRIX_SCRIPT+" "+str(fcs_file))

			## compute correction matrix
			print "[Step 5][+] Create correction values" 
			generate_all_delta_matrix()
			correction_values = create_correction_values()
		
			## compute compensated matrix
			print "[Step 6][+] Compute new compensation matrix"
			os.system("cp "+uncompensated_matrix_file_name+" "+predicted_matrix_file_name)
			transpose_matrix_file_dev(predicted_matrix_file_name)
			#create_compensated_matrix(uncompensated_matrix_file_name, correction_values)
			#correct_compensation_matrix(uncompensated_matrix_file_name)
			#transpose_matrix_file(predicted_matrix_file_name)

			## evaluate matrix distance
			distance = compute_matrix_distance(predicted_matrix_file_name, manually_compensated_matrix_file)

			print "[*]\t"+str(center)+"\t"+str(Panel)+"\t"+str(ID)+"\t"+str(distance)
			log_file.write(str(center)+","+str(Panel)+","+str(ID)+","+str(distance)+"\n")


	log_file.close()














#test_training_dataset()
evaluate_compensation_precision()
#transpose_matrix_file_dev("test_matrix.csv")
