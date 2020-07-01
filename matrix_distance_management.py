







def generate_matrix_dist_structure(data_file):
	"""
	IN PROGRESS
	"""

	print "tardis"

	## parameters
	center_to_panel_to_patient_to_dist = {}

	## parse data file
	input_data = open(data_file, "r")
	for line in input_data:

		line = line.rstrip()
		line_in_array = line.split(",")

		if(line_in_array[0] not in center_to_panel_to_patient_to_dist.keys()):
			center_to_panel_to_patient_to_dist[line_in_array[0]] = {} 


	input_data.close()



	print center_to_panel_to_patient_to_dist


#generate_matrix_dist_structure("results/distances_between_matrix.log")




def generate_panel_dist_figure(panel_number):
	"""
	IN PROGRESS
	"""


	## importation
	import numpy
	import matplotlib.pyplot as plt

	## parameters
	#data_file = "results/distances_between_matrix_complete_save.log"
	data_file = "matrix_distances.csv"
	center_to_dist = {}
	center_to_mean_dist = {}


	input_data = open(data_file, "r")
	for line in input_data:
		line = line.rstrip()
		line_in_array = line.split(",")

		if(str(line_in_array[1]) == str(panel_number)):
			if(line_in_array[0] not in center_to_dist.keys()):
				center_to_dist[line_in_array[0]] = []
				center_to_dist[line_in_array[0]].append(float(line_in_array[-1]))
			else:
				center_to_dist[line_in_array[0]].append(float(line_in_array[-1]))
	input_data.close()

	
	## Plot Hist
	for center in center_to_dist.keys():
		center_to_mean_dist[center] = numpy.mean(center_to_dist[center])

	
	width = 0.33
	plt.bar(center_to_mean_dist.keys(), center_to_mean_dist.values(), width, color='b')
	plt.savefig("results/panel_"+str(panel_number)+"_distance.png")





generate_panel_dist_figure(1)

"""
for x in xrange(0,10):
	generate_panel_dist_figure(x)
"""