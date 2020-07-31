'''
This code is licensed and documented by Teuvonet Technologies. 
Any use of this code, proprietary or personal, needs approval from the company.

'''

'''
This code is licensed and documented by Teuvonet Technologies. 
Any use of this code, proprietary or personal, needs approval from the company.

'''

import pyopencl as cl
import pyopencl.algorithm as algo
import pyopencl.array as pycl_array
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import os
import time
import imp
from operator import itemgetter

from sklearn.metrics import mean_squared_error
from math import sqrt

class Stack_Ensemble:

	def running_sum(self, a):
		tot = 0
		for i in a:
			tot += i
			yield tot
		yield tot+20

	def __init__(self):
		map_data = np.array(0, dtype = np.float32)
		#map_data_after = np.array(0, dtype=np.float32)
		#distances = np.array(0, dtype = np.float32)
		#weights_array = np.array(0, dtype = np.float32)
		#radius_map = np.array(0, dtype = np.float32)
		#n_partitions = np.array(0, dtype = np.float32)
		#no_neurons = np.array(0, dtype = np.float32)
		#no_nets = np.array(0, dtype = np.float32)
		#len_of_each_part = np.array(0, dtype = np.float32)
		#tot_components = np.array(0, dtype = np.float32)
		#input_start_point = np.array(0, dtype = np.float32)
		#active_centers = np.array(0, dtype = np.float32)
		#number_of_hyper = np.array(0, dtype = np.float32)
		#f_array_test_data = np.array(0, dtype = np.float32)
		#number_of_datapoints = np.array(0, dtype = np.float32)
		#neurons_per_net = np.array(0, dtype = np.float32)
		#a = np.array(0, dtype = np.float32)
		
	

	def encode_target(self, df, target_column):
	    """Add column to df with integers for the target.
	    Args
	    ----
	    df -- pandas DataFrame.
	    target_column -- column to map to int, producing
		             new Target column.
	    Returns
	    -------
	    df_mod -- modified DataFrame.
	    targets -- list of target names.
	    """
	    df_mod = df.copy()
	    targets = df_mod[target_column].unique()
	    map_to_int = {name: n for n, name in enumerate(targets)}
	    df_mod["Encode "+str(target_column)] = df_mod[target_column].replace(map_to_int)

	    return (df_mod, targets)

	def the_entire_stack_process(self, input_data_df, target, should_i_boost, test_data_df, context_django, train_split):

		#print("Entered pyopencl_boost")
		number_of_iterations = 1
		learning_rate_list = [0.5, 0.7, 0.9, 0.4, 0.3, 0.5, 0.01, 0.6, 0.0011, 0.011, 0.001, 0.02, 0.03]
		net_lattice = [3,4,5]
		no_neurons_per_net = [ i * i for i in net_lattice]
		no_neurons = sum(no_neurons_per_net) # 3*3 + 4*4 + 5*5 = 50 neurons
		no_nets = len(no_neurons_per_net)
		no_hyper = 1 # number of HP tials

		cl_filename = "./pyopencl_kernel.cl"
		with open(cl_filename, 'r') as fd:
			clstr = fd.read()

		platforms = cl.get_platforms()
		devices = platforms[0].get_devices()
		#print(platforms)
		#print(devices)

		context = cl.Context([devices[0]])

		program = cl.Program(context, clstr).build()

		myType = np.float32

		queue = cl.CommandQueue(context)

		mem_flags = cl.mem_flags
	
		#print("Input shape: "+str(input_data_df.shape))

		#Extract input and the target from the input as Dataframes

		#input_data_df = input_data_df.iloc[np.random.permutation(len(input_data_df))]
	
		input_data = input_data_df.loc[:, input_data_df.columns != target]
		target_arr = input_data_df.loc[:, input_data_df.columns == target]

		original_input_cols = [x for x in input_data.columns]

		input_data_df_rmse = input_data_df

		X=input_data.loc[:, input_data.columns!=target]

		encoded_features = {} #Django_Change

		'''

		for i in X.columns:
			if X.dtypes[i] == object:#isinstance(X.at[0,i],str):
				#print("Encoding:"+str(X.columns[i]))
				original_feature_name = i #Django_Change
				X, col=encode_target(X, i)
				encoded_features["Encode "+str(i)] = original_feature_name #Django_Change
				X=X.loc[:, X.columns!=i]
		'''
	
		encoded_features = [x for x in X.columns if X.dtypes[x] == object]
	
		#print("-------------------Encoded features"+str(encoded_features))
	
		X = pd.get_dummies(input_data)		

		input_data = X
	
		#print("Input shape:"+str(input_data.shape))

		input_column_names = [x for x in input_data.columns]

		################Normalization#################
		from sklearn.preprocessing import MinMaxScaler
		train_norm_scale = MinMaxScaler()
		train_norm_data = train_norm_scale.fit_transform(X.values)
		X = pd.DataFrame(train_norm_data, columns = input_column_names)

		std_denormalize = target_arr.std(ddof = 0)[0]
		max_denormalize = max(target_arr.values)
		min_denormalize = min(target_arr.values)

		target_norm_scale = MinMaxScaler()
		target_norm_data = target_norm_scale.fit_transform(target_arr.values)
		target_arr = pd.DataFrame(target_norm_data, columns = [target])
		################Normalization#################

		input_data = X

		tot_components = input_data.shape[1]#sum(no_attributes_per_part) # Total attributes
		no_tuples = (input_data_df.shape[0])
	
		temp_input_column_names = []
		new_input_column_names = []
		y=0;
		while y < number_of_iterations:
		    y += 1
		    temp_input_column_names = [x for x in input_data.columns]
		    np.random.shuffle(temp_input_column_names ) 
		    new_input_column_names += temp_input_column_names

		#new_input_column_names1 = ['F','D','G','B','A','E','H']
		#new_input_column_names = new_input_column_names+new_input_column_names1
		#print("New column Input datasets22" + str(new_input_column_names))

		tot_components = len(new_input_column_names)

		input_data_df = input_data
		"""
		new_input_column_names = [x for x in input_data.columns]
		np.random.shuffle(new_input_column_names)
		"""

		#new_input_column_names = ['A','D','B']

		#print(new_input_column_names)
	
		if tot_components >0:#< 100 :
			partition_size = tot_components 
	
		no_attributes_per_part = []
		#print("Check:"+str(tot_components))
		tot_components_iteration = int(tot_components/number_of_iterations)
		#print("Check1:"+str(tot_components_iteration))
		for y in range(0,number_of_iterations):
			for x in range(0, int((tot_components_iteration)/partition_size)):
				no_attributes_per_part.append(partition_size)
			if((tot_components_iteration)%partition_size)!=0:
				no_attributes_per_part.append((tot_components_iteration)%partition_size)
	
		"""
		if (tot_components) % partition_size != 0:
			if partition_size > (tot_components)/2:
				no_attributes_per_part.append(tot_components % partition_size)
			else:	
				k = (tot_components) % partition_size
				while k > 0:
					i = 0
					while k > 0 and i < len(no_attributes_per_part):
						no_attributes_per_part[i] = no_attributes_per_part[i] + 1
						k = k - 1
						i = i + 1
						#print(k)
		"""
		#print("Test1 : "+ str(no_attributes_per_part))

		no_partitions = len(no_attributes_per_part) # from data

		#print("Num partitions :"+str(no_partitions))
		#print("Number of part:"+str(no_partitions))

		#print([x for x in input_data.columns])
		#print(which_partition)
		#print("Input:"+str(input_data))
		input_data = input_data[new_input_column_names].values.ravel()
		input_data = np.array(input_data, dtype = np.float32)
		#print("Input Data: "+ str(input_data))
		#i = 0
		#for inp in input_data:
			#print(str(i)+" "+str(inp))
			#i = i + 1

		#input_data = np.array(np.random.uniform(0, 1, size=(tot_components, no_tuples)).T , dtype = np.float32)
		#np.array(np.arange(0,tot_components), dtype = np.float32)


		#Test Values------------------------
		#print("Input:"+str(input_data))

		#print(input_data_df.shape[1])

		#no_attributes_per_part = [3,2]
		#no_attributes_per_part = [200, 200 ,200, 200, 200, 250, 250, 58]
		#no_attributes_per_part = [1000,1000,1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
		#no_attributes_per_part = [12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500] # from data
		#which_partition = [0,0,0,1,1,1]

		# Test Values -----------------------------

		#print(input_data)

		input_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = input_data)

		no_work_units = tot_components * no_neurons # one row for HP tuning

		map_data = np.array(np.zeros(shape=(no_hyper, no_work_units),), dtype = np.float32)
		#np.array(np.random.uniform(0, 3, size=(no_hyper, no_work_units)) , dtype = np.float32).ravel() 
		#np.array(np.random.uniform(0, 0.1, size=(no_hyper, no_work_units)) , dtype = np.float32).ravel() 
		#np.array(np.random.uniform(0, 1, size=(no_hyper, no_work_units)) , dtype = np.float32).ravel() 
		#np.array(np.arange(1, no_hyper*no_work_units+1), dtype = np.float32)

		'''
		print("Map before:")

		i = 0
		for m in map_data:
			print(str(i)+" "+str(m))
			i = i + 1
			if i % 7 == 0:
				print("\n")
		'''
		#print("Map:"+str(map_data[1][4])+" "+str(map_data[1][5])+"\n\n\n")

		#print("Waiting for results !!!!!!!!!!!!!!!")
		no_work_units_x = no_neurons * no_partitions
		no_work_units_y = no_hyper
	
		map_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = map_data)

		#print("X:"+str(no_work_units_x)+" Y:"+str(no_work_units_y))


		distances = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		#print("Distances:"+str(distances)+"\n\n\n")
		distances_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances.nbytes)


		len_of_each_part = np.array(no_attributes_per_part, dtype = np.int32)
		len_buffer_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = len_of_each_part)


		no_neurons = np.array(no_neurons, dtype = np.int32)
		no_neurons_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_neurons)


		tot_components = np.array(tot_components, dtype = np.int32)
		no_components_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_components)


		n_partitions = np.array(no_partitions, dtype = np.int32)
		no_partitions_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = n_partitions)

		neurons_per_net = np.array(no_neurons_per_net, dtype = np.int32)
		#print(neurons_per_net)
		neurons_per_net_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = neurons_per_net)

		no_nets = np.array(no_nets, dtype = np.int32)
		no_nets_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_nets)


		min_array = np.array(np.full((no_nets * n_partitions * no_hyper,), sys.maxsize), dtype = np.float32)
		#print(min_array)
		min_array_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = min_array.nbytes)

		min_pos = np.array(np.full((no_nets * n_partitions * no_hyper,), sys.maxsize), dtype = np.int32)
		#print(min_pos)
		min_pos_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = min_pos.nbytes)

		net_lattice = np.array(net_lattice, dtype = np.int32)
		net_lattice_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = net_lattice)
		#Extra for update weights



		tot_components = np.array(tot_components)
		no_components_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_components)

		map_side_size = np.array(net_lattice, dtype=np.int32)
		map_side_size_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_side_size)

		number_of_net = np.array(no_nets, dtype = np.int32)
		number_of_net_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_net)

		no_neurons1 = np.array(list(self.running_sum([x*x for x in net_lattice])), dtype = np.int32)
		no_neurons1_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_neurons1)
	
		tot_neurons_per_hp = np.array(no_neurons * no_partitions, dtype = np.int32)
		tot_neurons_per_hp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_neurons_per_hp)

		tot_neurons_per_partition = np.array(no_neurons, dtype = np.int32)
		tot_neurons_per_partition_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_neurons_per_partition)	
	
		number_of_net_per_partition = np.array(no_nets, dtype = np.int32)
		number_of_net_per_partition_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_net_per_partition)


		number_of_hyper=np.array(no_hyper, dtype = np.int32)
		number_of_hyper_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_hyper)


		temp_guass = np.array(np.zeros((no_neurons * no_partitions, no_hyper)).T, dtype = np.float32)#np.random.uniform(0, 100, size=(5, 5)), dtype=np.float32)
		temp_guass_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, distances.nbytes)


		input_start_point = np.array(0, dtype=np.int32)
		input_start_point_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = input_start_point)

		number_of_datapoints = np.array(train_split * no_tuples, dtype=np.int32)
		number_of_datapoints_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_datapoints)

		j = 0
		learning_rate_list_per_part = []
		for l in learning_rate_list:
			for i in range(0, no_nets * n_partitions):
				learning_rate_list_per_part.append(l)
			if j == no_hyper-1:
				break
			j = j + 1

		#print(learning_rate_list_per_part)

		y =  np.array(learning_rate_list_per_part).astype(np.float32)
		learning_rate =  pycl_array.to_device(queue, y )
		# Create two random pyopencl arrays
		#a=pycl_array.to_device(queue, np.random.rand(27).astype(np.float32))
		#c = pycl_array.empty_like(a)  # Create an empty pyopencl destination array
		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)
		c = cl.Buffer(context, mem_flags.WRITE_ONLY, size = a.nbytes)
		#np.random.uniform(0, 1, size = (no_hyper * no_partitions * no_neurons,)).T, dtype = np.float32)
		w =  np.array(no_neurons).astype(np.int32)


		#print("SIZE:"+str(a.shape[0]))

		a_init = a
		start_time = time.time()

		'''
		print("Gauss list before:")
		i = 0
		for c1 in a:
			print(str(i)+" "+str(c1))
			i = i +1

		'''

		culumative_differnce_first = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
		culumative_differnce_first_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = culumative_differnce_first)


		culumative_differnce_average = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
		culumative_differnce_average_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = culumative_differnce_average)

		culumative_differnce_average_first = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
		culumative_differnce_average_first_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = culumative_differnce_average_first)

		culumative_differnce_per_neuron = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
		culumative_differnce_per_neuron_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = culumative_differnce_per_neuron)

		neigh_rate = np.array(np.ones(no_nets  * n_partitions * no_hyper), dtype = np.float32)
		neigh_rate_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = neigh_rate)

		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)
		c = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = a.nbytes, hostbuf = a)

		#print(a)
	
		#Train the first set of Kohonen nets in Phase 1

		program.phase1_neuron(queue, (no_work_units_x, no_work_units_y), None, map_buf_1, input_buf_1, distances_buf_1, len_buffer_1, no_neurons_buf, no_components_buf, no_partitions_buf, neurons_per_net_buf, no_nets_buf, net_lattice_buf, min_array_buf, min_pos_buf,no_components_buf,map_side_size_buf,number_of_net_buf,no_neurons1_buf,tot_neurons_per_hp_buf,tot_neurons_per_partition_buf, number_of_net_per_partition_buf,number_of_hyper_buf,temp_guass_buf_1,c,input_start_point_buf,number_of_datapoints_buf, culumative_differnce_first_buf,culumative_differnce_per_neuron_buf,culumative_differnce_average_buf,culumative_differnce_average_first_buf,learning_rate.data, neigh_rate_buf)

		#cl.enqueue_copy(queue, min_array, min_array_buf)
		#cl.enqueue_copy(queue, distances, distances_buf_1)
		#cl.enqueue_copy(queue, min_pos, min_pos_buf)
		map_data_after = map_data[:]
		#print("Before:"+str(map_data))
		cl.enqueue_copy(queue, map_data, map_buf_1)
		#print("After:"+str(map_data))



		"""
		print("\nBefore Map:")

		for m in map_data.ravel():
			print(str(m)+" ")

		print("\n\n\n\n\n")

	
		print("\nAfter Map:")

		i=0
		for m in map_data.ravel():
			print(str(i)+" "+str(m)+" ")
			i = i + 1
			if i %3 == 0:
				print("\n")
		print("\n")
		"""
	
		overall_distances = np.array(np.zeros((tot_components * no_hyper,)).T, dtype = np.float32)
		overall_distances_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = overall_distances.nbytes)

		work_x = tot_components * 1
		work_y = no_hyper


		active_centers = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.int32)
		active_centers_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers)
	
		#Calculate activations of trained neurons

		program.active_centers_phase1(queue,(no_work_units_x, no_work_units_y), None, map_buf_1, no_components_buf, overall_distances_buf, no_neurons_buf, no_partitions_buf, number_of_hyper_buf, no_nets_buf, neurons_per_net_buf, len_buffer_1,number_of_datapoints_buf,input_buf_1,distances_buf_1,active_centers_buf,no_neurons1_buf,map_side_size_buf,min_pos_buf,min_array_buf,neurons_per_net_buf)

		cl.enqueue_copy(queue, active_centers, active_centers_buf)


		#print("--------ACTIVE CENTERS PHASE1------------"+str(active_centers))


		#print(str(work_x)+" "+str(work_y))
	
		#Calculate average distance per feature to be able to rank them later.

		program.distance_computation_phase1(queue, (work_x , work_y), None, map_buf_1, no_components_buf, overall_distances_buf, no_neurons_buf, no_partitions_buf, number_of_hyper_buf, no_nets_buf, neurons_per_net_buf, len_buffer_1, active_centers_buf)

		cl.enqueue_copy(queue, overall_distances, overall_distances_buf)

		#print("Unsorted Distances:")
		i = 0
		j = 0
		for x in overall_distances:
			#print("Feature:"+str(i)+" Distance: "+str(x)+" Hyper: "+str(j))
			i = i + 1
			if i == tot_components:
				j = j + 1
				i = 0
	
		cust_type = np.dtype([('feature', np.int32),('dist', np.float32),('hyper',np.int32)])


		#print(c_decl)
		#print(new_type)

		#print(tools.dtype_to_ctype(cust_type))
		tot_components_per_iter = int(tot_components/number_of_iterations)
		#print("Testing"+ str(tot_components_per_iter))
		my_objects = np.empty(((no_hyper*number_of_iterations, tot_components_per_iter)), dtype = cust_type)
		i = 0
		j = 0
		k = 0

		#print(my_objects.shape)
		#print("\n\n\nUnsorted:")
		for x1 in overall_distances:
			#print("Feature:"+str(new_input_column_names[i])+" "+str(i)+"\tDistance: "+str(x1)+"\tHyper: "+str(j))
			my_objects[j][i] = (i, x1, j)
			i = i + 1
			k = k + 1
			if i == tot_components_per_iter:
				j = j + 1
				i = 0
				#print("\n")
		#print("Unsorted:"+str(my_objects))
		#my_arr = np.empty(5, dtype = cust_type)
		my_object = np.array(my_objects, dtype = cust_type)

		def getKey(elem):
			return elem[1]

		#print("Sorted using inbuilt:"+str(sorted(my_objects, key = getKey)))


		#from pyopencl.algorithm import RadixSort
		#sort = RadixSort(context, arguments = "__global dist_feature* ary", key_expr="ary[i].dist", sort_arg_names=["ary"], key_dtype = np.int32, scan_kernel = MyScan)

		#sorted_obj = obj_buf#pycl_array.zeros(queue, (no_hyper,tot_components), dtype = cust_type)

		sorted_obj = []

	
		i=0
		for x in range(0, no_hyper*number_of_iterations):
			#(sorted_obj_hyp,), evt = sort(pycl_array.to_device(queue, my_object[x]), key_bits = 64, queue = queue)
			sorted_obj_hyp = sorted(my_object[x], key = getKey)
			sorted_obj.append(sorted_obj_hyp)	
			#print("Test 2 \n:"+str(i) + str(sorted_obj_hyp))
			i=i+1
		#print("Sorted using pyopencl:"+str(sorted_obj))

		#sorted_obj_host = sorted_obj.map_to_host(queue)
		#print("Sorted:"+str(sorted_obj_host))

		#Generation of 'sum11' array that can be used too cal agg results
		sum11=[]
		for r in range(0,tot_components_per_iter):
			sum11.append(0)
		#print(sum11)
		new_input_column_names_iter = []
		for x in range(0,no_hyper):
			new_input_column_names_iter = new_input_column_names_iter + (new_input_column_names)
		#Generation of 'agg_coulmn_name' array that can be used too map agg results
		agg_coulmn_name = []
		j=0
		for x in input_column_names:
			agg_coulmn_name.append(input_column_names[j])
			j=j+1
		#print(agg_coulmn_name)
		#print("\n\n\nSorted:")
		i = 0
		part_num = 0
		j=0
		for x in range(0, no_hyper*number_of_iterations):
			sorted_obj[x].reverse()
			for x1 in sorted_obj[x]:
				#x1 = x1.map_to_host(queue)
				#if i < 20:
				temp=x*tot_components_per_iter + x1['feature']
				j=0
				for y in input_column_names:
					if(agg_coulmn_name[j]==new_input_column_names_iter[x*tot_components_per_iter + x1['feature']]):
						sum11[j] = sum11[j] + float(x1['dist']) 
					j=j+1
				#print("Feature: "+str(new_input_column_names_iter[x*tot_components_per_iter + x1['feature']])+"\tDistance: "+str(float(x1['dist']))+ "  Iteration: "+str(int(x/no_hyper)) +" ")
				i = i + 1
			i = 0
			#print("\n")


		#POst this id for avg calculations for agg results
		j=0
		for y in input_column_names:
			sum11[j] = sum11[j]/(no_hyper*number_of_iterations)
			j=j+1
		#print("Agg REsults Sorted: ")
		##Post this is for mapping agg results
		i = 0
		colmn_agg_results =[]
		temp_agg = sum11

		#print(temp_agg)

		sum11 = sorted(sum11)
		sum11.reverse()

		epsilon = 1e-10
		i11 = 0
		for x11 in sum11:
			j11 = 0
			for y11 in sum11:
				if x11 == y11 and i11 != j11:
					sum11[j11] = sum11[j11] + epsilon
				j11 = j11 + 1

			i11 = i11 + 1

		#print("--------------------------")
		#print(sum11)
		#print(temp_agg)
		for x in sum11:
			j=0
			for y in temp_agg:
				if y ==sum11[i]:
					colmn_agg_results.append(input_column_names[j])
					#print(str(j)+" "+str(input_column_names[j])+" "+str(y)+" "+str(sum11[j]))
				j=j+1
			i=i+1
		#print("\n")
		#print("Top features in the order")
		#print(colmn_agg_results)


		j1=0
		for x in sum11:
			#print(str(colmn_agg_results[j1])+"  :  "+str(sum11[j1]))
			j1 = j1+1
		#print("\n")
		
		#print("BOOST Time for phase 1: %s seconds" %(time.time() - start_time) )
		#my_results_file.write("\nTime for phase 1: %s seconds" %(time.time() - start_time) )
		

		############################################################################################################################################


		#############################################PHASE 2#################################################


		############################################################################################################################################


		start_time = time.time()

		#print(tot_components_iteration)
		tot_components = tot_components_iteration

		increment = 1

		#print(tot_components_iteration)
		tot_components = tot_components_iteration

		if tot_components <= 100:
			if tot_components/4 == 0:
				first_term = 1
			else:
				first_term = tot_components/3

			if tot_components/3 == 0:
				second_term = 1
			else:
				second_term = tot_components/2
	

			if tot_components/2 == 0:
				third_term = 1
			else:
				third_term = tot_components
	
			no_attributes_per_part = [first_term, second_term, third_term]
			increment = 1
			no_attributes_per_part = np.arange(1,(tot_components+1))
			

		else:
			no_attributes_per_part = [tot_components/6, tot_components/5, tot_components/4, tot_components/3, tot_components/2, tot_components]#[20, 30, 50, 80, 100, 150, 200]

			increment = 50
		
			for i in range(1,100):
				no_attributes_per_part.append(i)
		
			for i in range(100, tot_components, increment):
				no_attributes_per_part.append(i)


		#no_attributes_per_part = [1,2,3]#, 75, 100]

		#print(no_attributes_per_part)

		learning_rate_list = [0.1]
		net_lattice = [3,4,5]
		no_partitions = len(no_attributes_per_part)
		no_neurons_per_net = [ i * i for i in net_lattice]
		no_neurons = sum(no_neurons_per_net) # 3*3 + 4*4 + 5*5 = 50 neurons
		no_nets = len(no_neurons_per_net)
		no_hyper = 1 # number of HP trials


		i = 0
		#sorted_obj[0].reverse()
		col_names_phase2 = []

		for j in range(0, len(no_attributes_per_part)):
			i = 0
			temp = []
			for x1 in colmn_agg_results:
				temp.append(x1)#new_input_column_names[x1['feature']])
				i = i + 1
				if i == no_attributes_per_part[j]:
					break
			col_names_phase2.append(temp)

		#print(col_names_phase2)

		#col_names_phase2 = [['A', 'B', 'E'],['A', 'B', 'E']]

		phase2_input = input_data_df[col_names_phase2[0]]
		for l in range(1, len(col_names_phase2)):
			#print(input_data[col_names_phase2[l]])
			phase2_input = pd.concat([phase2_input, input_data_df[col_names_phase2[l]]], axis = 1)

		#print("Input:\n"+str(phase2_input))

		input_data = np.array(phase2_input.values.ravel(), dtype = np.float32)

		#print("Input:"+str(input_data))


		tot_components = sum(no_attributes_per_part)
		no_tuples = (input_data_df.shape[0])

		no_work_units = tot_components * no_neurons # one row for HP tuning

		map_data = np.array(np.zeros(shape=(no_hyper, no_work_units),), dtype = np.float32)
		#np.array(np.random.uniform(0, 1, size=(no_hyper, no_work_units)) , dtype = np.float32).ravel() 

		no_work_units_x = no_neurons * no_partitions
		no_work_units_y = no_hyper

		distances = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)

		len_of_each_part = np.array(no_attributes_per_part, dtype = np.int32)

		no_neurons = np.array(no_neurons, dtype = np.int32)

		tot_components = np.array(tot_components, dtype = np.int32)

		n_partitions = np.array(no_partitions, dtype = np.int32)

		neurons_per_net = np.array(no_neurons_per_net, dtype = np.int32)

		no_nets = np.array(no_nets, dtype = np.int32)

		min_array = np.array(np.full((no_nets * n_partitions * no_hyper,), sys.maxsize), dtype = np.float32)

		min_pos = np.array(np.full((no_nets * n_partitions * no_hyper,), sys.maxsize), dtype = np.int32)

		net_lattice = np.array(net_lattice, dtype = np.int32)

		tot_components = np.array(tot_components)

		map_side_size = np.array(net_lattice, dtype=np.int32)

		number_of_net = np.array(no_nets, dtype = np.int32)

		no_neurons1 = np.array(list(self.running_sum([x*x for x in net_lattice])), dtype = np.int32)

		tot_neurons_per_hp = np.array(no_neurons * no_partitions, dtype = np.int32)

		tot_neurons_per_partition = np.array(no_neurons, dtype = np.int32)

		number_of_net_per_partition = np.array(no_nets, dtype = np.int32)

		number_of_hyper=np.array(no_hyper, dtype = np.int32)

		temp_guass = np.array(np.zeros((no_neurons * no_partitions, no_hyper)).T, dtype = np.float32)

		input_start_point = np.array(0, dtype=np.int32)

		number_of_datapoints = np.array(train_split * no_tuples, dtype=np.int32)

		j = 0
		learning_rate_list_per_part = []
		for l in learning_rate_list:
			for i in range(0, no_nets * n_partitions):
				learning_rate_list_per_part.append(l)
			if j == no_hyper-1:
				break
			j = j + 1

		#print(learning_rate_list_per_part)
		y =  np.array(learning_rate_list_per_part).astype(np.float32)

		w =  np.array(no_neurons).astype(np.int32)

		culumative_differnce_first = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)

		culumative_differnce_average = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)

		culumative_differnce_average_first = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)

		culumative_differnce_per_neuron = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
	
		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)

		a_init = a

		map_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = map_data, size = map_data.nbytes)
		input_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = input_data)
		distances_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances.nbytes)
		len_buffer_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = len_of_each_part)
		no_neurons_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_neurons)
		no_components_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_components)
		no_partitions_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = n_partitions)
		neurons_per_net_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = neurons_per_net)
		no_nets_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_nets)
		min_array_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = min_array.nbytes)
		min_pos_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = min_pos.nbytes)
		net_lattice_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = net_lattice)
		no_components_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_components)
		map_side_size_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_side_size)
		number_of_net_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_net)
		no_neurons1_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_neurons1)
		tot_neurons_per_hp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_neurons_per_hp)
		tot_neurons_per_partition_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = tot_neurons_per_partition)
		number_of_net_per_partition_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_net_per_partition)
		number_of_hyper_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_hyper)
		temp_guass_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, distances.nbytes)
		input_start_point_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = input_start_point)
		number_of_datapoints_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_datapoints)
		learning_rate =  pycl_array.to_device(queue, y )
		c = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = a, size = a.nbytes)
		number_of_neurons =  pycl_array.to_device(queue, w )

		culumative_differnce_first_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = culumative_differnce_first.nbytes)

		culumative_differnce_average_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = culumative_differnce_average.nbytes)

		culumative_differnce_average_first_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = culumative_differnce_average_first.nbytes)

		culumative_differnce_per_neuron_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = culumative_differnce_per_neuron.nbytes)

		neigh_rate = np.array(np.ones(no_nets  * n_partitions * no_hyper), dtype = np.float32)
		neigh_rate_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = neigh_rate)

	
		#print(no_work_units_x)
		#print(no_work_units_y)
	
		program.phase1_neuron(queue, (no_work_units_x, no_work_units_y), None, map_buf_1, input_buf_1, distances_buf_1, len_buffer_1, no_neurons_buf, no_components_buf, no_partitions_buf, neurons_per_net_buf, no_nets_buf, net_lattice_buf, min_array_buf, min_pos_buf,no_components_buf,map_side_size_buf,number_of_net_buf,no_neurons1_buf,tot_neurons_per_hp_buf,tot_neurons_per_partition_buf, number_of_net_per_partition_buf,number_of_hyper_buf,temp_guass_buf_1,c,input_start_point_buf,number_of_datapoints_buf, culumative_differnce_first_buf,culumative_differnce_per_neuron_buf,culumative_differnce_average_buf,culumative_differnce_average_first_buf,learning_rate.data, neigh_rate_buf)

		map_data_after = map_data[:]
		#print("Before:"+str(map_data))
		cl.enqueue_copy(queue, map_data, map_buf_1)
	
	
		radius_map = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.float32)
		radius_map_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = radius_map.nbytes)

		distances = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		distances_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances.nbytes)

		active_centers = np.array(np.zeros((no_hyper * no_neurons * no_partitions,)).T, dtype = np.int32)	
		active_centers_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers, size = active_centers.nbytes)
	
		#active_centers_phase1 = mod.get_function("active_centers_phase1")	

		#active_centers_phase1(drv.InOut(map_data), drv.InOut(tot_components), drv.InOut(overall_distances), drv.InOut(no_neurons), drv.InOut(n_partitions), drv.InOut(number_of_hyper), drv.InOut(no_nets), drv.InOut(neurons_per_net), drv.InOut(len_of_each_part),drv.InOut(number_of_datapoints),drv.InOut(input_data),drv.InOut(distances),drv.InOut(active_centers),drv.InOut(no_neurons1),drv.InOut(map_side_size),drv.InOut(min_pos),drv.InOut(min_array),drv.InOut(neurons_per_net), grid=(no_work_units_x_cuda, no_work_units_y_cuda), block=(no_neurons_lvalue,1,1))
	
	
		########################################Django_Change
	
		no_target_splits = 2 #Django_Change
	
		active_centers_split = np.array(np.zeros((no_target_splits * no_hyper * no_neurons * no_partitions,)).T, dtype = np.int32)
	
		active_centers_split_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_split, size = active_centers_split.nbytes)

		target_arr_val = np.array(target_arr, dtype = np.float32).ravel()
	
		target_arr_val_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = target_arr_val, size = target_arr_val.nbytes)
	
		splitting_point_target = float(max_denormalize - min_denormalize) / no_target_splits #Django_Change
	
		splitting_point_target = float((splitting_point_target - min_denormalize) / (max_denormalize - min_denormalize) ) 
	
		#print(splitting_point_target)
	
		splitting_point_target_val = np.array(splitting_point_target, dtype = np.float32)	
	
		splitting_point_target_val_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = splitting_point_target_val)
	
		#print(target_arr_val)
	
		#print(max_denormalize)
		#print(min_denormalize)
		#print(no_target_splits)
	
		########################################Django_Change
	
		#Calculate radii of trained neurons to form gaussians in phase 3

		program.calculate_radius_phase2(queue,(no_work_units_x, no_work_units_y), None, map_buf_1, no_components_buf, overall_distances_buf, no_neurons_buf, no_partitions_buf, number_of_hyper_buf, no_nets_buf, neurons_per_net_buf, len_buffer_1,number_of_datapoints_buf,input_buf_1,distances_buf_1,active_centers_buf,no_neurons1_buf,map_side_size_buf,min_pos_buf,min_array_buf,neurons_per_net_buf, radius_map_buf, splitting_point_target_val_buf, active_centers_split_buf, target_arr_val_buf)

		#(queue,(no_work_units_x, no_work_units_y), None, map_buf_1, input_buf_1, distances_buf_1, len_buffer_1, no_neurons_buf, number_of_net_buf, no_components_buf, no_partitions_buf, active_centers_buf, input_start_point_buf,number_of_datapoints_buf, min_array_buf, min_pos_buf, neurons_per_net_buf, radius_map_buf)  

		#print("Before:"+str(radius_map))
		cl.enqueue_copy(queue, radius_map, radius_map_buf)
		#print("After radius:"+str(radius_map))

		#print("Before:"+str(active_centers))
		cl.enqueue_copy(queue, active_centers, active_centers_buf)
	
		#print(active_centers)
	
		cl.enqueue_copy(queue, active_centers_split, active_centers_split_buf)

		map_data_phase2 = map_data[0]

		#Correct
		#print("Map:"+str(len(map_data)))

		#print("Map:"+str(map_data))

		#print("BOOST Time for phase 2: %s seconds" %(time.time() - start_time) )
		#my_results_file.write("\nTime for phase 2: %s seconds" %(time.time() - start_time) )

		####################################### PHASE 3 #############################################

		start_time = time.time()


		no_hyper = 20
		number_of_hyper=np.array(no_hyper, dtype = np.int32)
		number_of_hyper_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_hyper)


		f_array = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()
		f_array_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = f_array.nbytes)

		#print(f_array.shape)

		weights_array = np.array(np.zeros((no_partitions * no_hyper * no_neurons,)).T, dtype = np.float32).ravel()
		weights_array_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = weights_array.nbytes, hostbuf = weights_array)

		#print(weights_array.shape)


		len_of_each_part = np.array(no_attributes_per_part, dtype = np.int32)
		len_buffer_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = len_of_each_part)


		#target_arr_one = target_arr
		#for l in range(1, no_hyper * no_partitions):
			#print(input_data[col_names_phase2[l]])
		#	target_arr = pd.concat([target_arr, target_arr_one], axis = 0)

		#print((sum(target_arr.values)[0]))
		#print((len(target_arr)))
	
	
		average_target_value = (sum(target_arr.values)[0]) / len(target_arr)

		target_arr_full = []
		for t in target_arr.values:
			for l in range(0, no_hyper * no_partitions):
				target_arr_full.append(t)

		target_arr = target_arr_full

		number_of_datapoints = np.array(no_tuples, dtype=np.int32)
		number_of_datapoints_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_datapoints)

	
		#print(target_arr)

		error_tp = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		#print(error_tp)
		error_tp_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_tp)

		error_train = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_train_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_train)

		error_train_prev = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_train_prev_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_train_prev)


		error_test = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_test_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_test)


		error_test_prev = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_test_prev_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_test_prev)


		f_array_train = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()
		f_array_train_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = f_array.nbytes)


		f_array_test = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()
		f_array_test_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = f_array_test.nbytes)


		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)
		c = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = a, size = a.nbytes)

		distances = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		distances_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances.nbytes)

		distances_train = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		distances_train_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances_train.nbytes)

		distances_test = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		distances_test_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances_test.nbytes)



		convergence_threshold = [0,0,0,0,0,0]#0.005, 0.004, 0.009, 0.007, 0.002, 0.001]
		learning_rate_list = [0.5, 0.15, 0.06, 0.06, 0.06, 0.06,0.06, 0.001, 0.004, 0.02, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
		learning_rate_drop = [5,4,5,4,5,7,8,3,4,9,4,5,6,4,5,6,4,5,5,4]


		j = 0
		learning_rate_list_per_part = []
		for l in learning_rate_list:
			for i in range(0, no_nets * n_partitions):
				learning_rate_list_per_part.append(l)
			if j == no_hyper-1:
				break
			j = j + 1

		#print(learning_rate_list_per_part)
		learning_rate_list_per_part_original = learning_rate_list_per_part
		y =  np.array(learning_rate_list_per_part).astype(np.float32)

		learning_rate =  pycl_array.to_device(queue, y )

		learning_rate_init = y

		learning_rate_list = [0.0001, 0.002, 0.0003]

		j = 0
		learning_rate_list_per_part = []
		for l in learning_rate_list:
			for i in range(0, no_nets * n_partitions):
				learning_rate_list_per_part.append(l)
			if j == no_hyper-1:
				break
			j = j + 1

		y_end =  np.array(learning_rate_list_per_part).astype(np.float32)
		learning_rate_end =  pycl_array.to_device(queue, y_end)

		error_weight_change_per_part = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_weight_change_per_part_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_weight_change_per_part)


		error_weight_change_per_neuron = np.array(np.zeros((no_hyper * no_partitions * no_neurons),).T,dtype = np.float32)
		error_weight_change_per_neuron_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_weight_change_per_neuron)



		error_pass = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_pass_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_pass)


		error_train_init = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_train_init_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_train_init)


		error_test_init = np.array(np.zeros((no_hyper * no_partitions),).T,dtype = np.float32)
		error_test_init_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = error_test_init)

		x = 0
		#for t in target_arr:
		#	print(str(x)+" "+str(t))
		#	x = x + 1

		#print(target_arr[22])

		converge_var = np.array(1, dtype = np.int32)
		converge_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = converge_var.nbytes)


		f_array_init = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()
		f_array_init_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = f_array_init.nbytes)


		convergence_threshold = np.array(convergence_threshold, dtype = np.float32).ravel()
		convergence_threshold_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = convergence_threshold)

		learning_rate_drop = np.array(learning_rate_drop, dtype = np.int32).ravel()
		learning_rate_drop_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = learning_rate_drop)


		no_work_units_x = no_neurons * no_partitions
		no_work_units_y = no_hyper

		#print("Distances phase 3 crossval:"+str(distances_train))

		start_time = time.time()
	
		DATA_RESULTS_SINGLE = []

		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)
		c = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = a, size = a.nbytes)

		weights_full_array = []

		f_array_final = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()

		target_arr_orig = target_arr
		target_arr = np.array(target_arr, dtype = np.float32).ravel()
		target_arr_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = target_arr)
		
		train_split = np.array(train_split, dtype = np.float32)
		train_split_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = train_split.nbytes, hostbuf = train_split)
	
		passes_max = np.array(50, dtype=np.int32)
		passes_max_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = passes_max.nbytes, hostbuf = passes_max)
	
		weights_array = np.array(np.zeros((no_partitions * no_hyper * no_neurons,)).T, dtype = np.float32).ravel()
		weights_array_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = weights_array.nbytes, hostbuf = weights_array)
	
		converge_partition_array = np.array(np.zeros(no_hyper * no_partitions,).T, dtype = np.int32).ravel()
		converge_partition_array_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size = converge_partition_array.nbytes, hostbuf = converge_partition_array)

		program.phase3_neuron(queue, (no_work_units_x, no_work_units_y), None, map_buf_1, input_buf_1, distances_buf_1, len_buffer_1, no_neurons_buf, no_components_buf, no_partitions_buf, neurons_per_net_buf, no_nets_buf, net_lattice_buf, min_array_buf, min_pos_buf,no_components_buf,map_side_size_buf,number_of_net_buf,no_neurons1_buf,tot_neurons_per_hp_buf,tot_neurons_per_partition_buf, number_of_net_per_partition_buf,number_of_hyper_buf,temp_guass_buf_1,c,input_start_point_buf,number_of_datapoints_buf, culumative_differnce_first_buf,culumative_differnce_per_neuron_buf,culumative_differnce_average_buf,culumative_differnce_average_first_buf,learning_rate.data, neigh_rate_buf, f_array_buf, weights_array_buf, radius_map_buf, active_centers_buf, target_arr_buf, error_tp_buf, error_pass_buf, learning_rate.data, learning_rate_end.data, error_train_buf, error_test_buf, f_array_train_buf, distances_train_buf_1, error_train_prev_buf, error_test_prev_buf, error_train_init_buf, error_test_init_buf, f_array_init_buf, converge_buf, convergence_threshold_buf, learning_rate_drop_buf, converge_partition_array_buf, train_split_buf, passes_max_buf)

		
		cl.enqueue_copy(queue, error_tp, error_tp_buf)
		cl.enqueue_copy(queue, f_array, f_array_buf)
		cl.enqueue_copy(queue, f_array_train, f_array_train_buf)
		cl.enqueue_copy(queue, f_array_test, f_array_test_buf)
		cl.enqueue_copy(queue, error_train, error_train_buf)
		cl.enqueue_copy(queue, error_test, error_test_buf)
		cl.enqueue_copy(queue, weights_array, weights_array_buf)
		cl.enqueue_copy(queue, error_train_init, error_train_init_buf)
		cl.enqueue_copy(queue, error_test_init, error_test_init_buf)
		cl.enqueue_copy(queue, f_array_init, f_array_init_buf)
		cl.enqueue_copy(queue, converge_var, converge_buf)
		
		weights_full_array.append(weights_array)

		target_value_original = target_arr[:]
		
		#print(target_arr)

		f_array_final = f_array_final + f_array_train 



		learning_rate_list_per_part = learning_rate_list_per_part_original
		#print(learning_rate_list_per_part)
		y =  np.array(learning_rate_list_per_part).astype(np.float32)


		#f_array_train = f_array_final

		#target_arr = target_arr_orig

		#print("PREDICTED, ERROR, TARGET")
		#for tt in range(0, 10):
		#	print(str(f_array_train[tt])+" "+str(target_arr[tt])+" "+str(target_value_original[tt]))

		#print("--------------")

		#print("BOOST Time for Phase 3: %s seconds" %(time.time() - start_time) )

		f_array_train = f_array_final

		target_arr = target_arr_orig


		from sklearn.metrics import mean_squared_error
		from math import sqrt

		#f_array_train = f_array
		#print(f_array)
		#print(weights_array)
		#print(f_array_train)
		#print(f_array_init)
		#print(error_tp)
		#print(f_array_train.shape)
		#print(f_array_test)
		#print("Initial Train:"+str(error_train_init))
		#print("Initial Test:"+str(error_test_init))
		#print("Final Train:"+str(error_train))
		#print("Final Test:"+str(error_test))

		#print("F array Initial: "+str(f_array_init))

		target_arr_1 = input_data_df_rmse.loc[:, input_data_df_rmse.columns == target].values

		max_denormalize = max(target_arr_1)
		min_denormalize = min(target_arr_1)

		i = 0
		target_part = []
		for x in range(0, no_tuples):
			target_part.append(min_denormalize + (target_arr[i] * (max_denormalize - min_denormalize)))
			i = i + no_hyper * no_partitions

		#print(target_part)

		#print("min target: "+str(min(target_part)))

		f_array_part = []

		for j in range(0, no_partitions * no_hyper):
			f_array_per_part = []
			i = 0
			for x in range(j, len(f_array_train), no_partitions * no_hyper):
				f_array_per_part.append(min_denormalize + (f_array_train[x] * (max_denormalize - min_denormalize)))
				i = i + 1
			f_array_part.append(f_array_per_part)

		#print(len(f_array_part))

		'''
		f_array_part_init = []

		for j in range(0, no_partitions * no_hyper):
			f_array_per_part_init = []
			i = 0
			for x in range(j, len(f_array_init), no_partitions * no_hyper):
				f_array_per_part_init.append(f_array_init[x])
				i = i + 1

			if np.isnan(f_array_per_part_init).any() == False:
				f_array_part_init.append(f_array_per_part_init)
				rms = sqrt(mean_squared_error(target_part, f_array_per_part_init))
			#print("(Normalized) Initial for Top "+str(no_attributes_per_part[j])+" : "+str(rms))

			################Normalization#################
			#rms = rms  * (max_denormalize - min_denormalize) #* std_denormalize	
			################Normalization#################
			#print("(Un-Normalized) Initial for Top "+str(no_attributes_per_part[j])+" : "+str(rms))

		'''

		#print((f_array_part))
		#print(target_part)
		#print(weights_array)


		#target_part = target_norm_scale.inverse_transform(pd.DataFrame(target_part).values)

		#for i in range(0, len(f_array_part)):
		#	f_array_part[i] = target_norm_scale.inverse_transform(pd.DataFrame(f_array_part[i]).values)


		#print(max_denormalize)
		#print(min_denormalize)

		#for i in range(0, len(f_array_part)):
		#	f_array_part[i] = [ (max_denormalize-min_denormalize)* f_part + min_denormalize for f_part in f_array_part[i] ]


		#print("\n\n\n\n\n")

		train_part_errors = []
		test_part_errors = []

		min_rmse_val = 1e+20
		min_rmse_tot_val = 1e+20
		min_rmse_part = -1
		
		min_rmse_train_val = 1e+20
		min_rmse_train_part = -1
		
		
		min_rmse_cv_val = 1e+20
		min_rmse_cv_part = -1				

		j = 0

		k = 0

		for f in f_array_part:
			train_rms = error_train[k] * (max_denormalize - min_denormalize) #* std_denormalize
			test_rms = error_test[k] * (max_denormalize - min_denormalize)
			if np.isnan(f).any() == False and np.isnan(train_rms) == False and np.isnan(test_rms) == False:
				rms = sqrt(mean_squared_error(target_part, f)) #* (max_denormalize - min_denormalize)	
				#print("(Normalized) Final for Top "+str(no_attributes_per_part[j])+" : "+str(rms))
				################Normalization#################
				#rms = rms  * (max_denormalize - min_denormalize) #* std_denormalize
				################Normalization#################
				if (train_rms + test_rms) < min_rmse_tot_val:
					min_rmse_tot_val = train_rms + test_rms
					min_rmse_val = test_rms
					min_rmse_part = j
					
				if train_rms < min_rmse_train_val:
					min_rmse_train_val = train_rms
					min_rmse_train_part = j
					
				if test_rms < min_rmse_cv_val:
					min_rmse_cv_val = test_rms
					min_rmse_cv_part = j					
					

				#print("-----------------BOOST ERROR: Hyper: "+str(k / no_partitions)+" Partition: "+str(k % no_partitions)+" Errors: "+str(train_rms)+" "+str(error_train[k])+" "+str(rms)+" "+str(error_test[k])+" "+str(test_rms))
				train_part_errors.append(train_rms)
				test_part_errors.append(test_rms) #* (max_denormalize - min_denormalize) * std_denormalize)
				j = j + 1

			k = k + 1


		total_rmse_denorm = np.array((0.5 * (error_train + error_test) + (error_train - error_test) )  * (max_denormalize - min_denormalize) )

		min_rmse_part = np.argpartition(total_rmse_denorm, 1)[:1]
		


		#print("Min Boost RMSE PArt: "+str(min_rmse_part)+" RMSE Value: "+str(min_rmse_val)+" Min with Train alone: "+str(min_rmse_train_part)+" "+str(min_rmse_train_val)+" Min with Test alone: "+str(min_rmse_cv_val)+" "+str(min_rmse_cv_part))

		#print(f_array_part[min_rmse_part[0]])

		#print(target_part)

		x_axis = np.arange(0,len(target_part))
		#plt.plot(x_axis, f_array_part[min_rmse_part], 'red', x_axis, target_part, 'y')
		#plt.plot(x_axis, f_array_part[0], 'red', x_axis, f_array_part[1], 'blue', x_axis, f_array_part[2], 'green', target_part, 'y')
		#plt.axis([0 * no_tuples, no_tuples, min_denormalize-2, max_denormalize+2])
		#plt.show()
		
		
		#print("--------------------------------------------INSIDE STACKING TESTING PROCESS-------------------------------------------------")
		

		##############################################################################################################TEST INFERENCE PART######################################################################################################################


		################Encoding#################

		X = test_data_df

		for i in X.columns:
			#print(str(i)+" zzz "+str((test_data_df.dtypes[i])))
			if (X.dtypes[i]) == (object):
				#print("Encoding:")
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]

		test_input_column_names = [x for x in X.columns]

		test_data_cleaned = X

		################Encoding#################

		#test_data_cleaned = test_data_df
		#print("-------------IN NORMAL TEST DATA---------\n"+str(test_data_cleaned))
		#Exclude target if present
		if target in test_data_cleaned.columns:
			test_data_cleaned = test_data_cleaned.loc[:, test_data_cleaned.columns != target]

		#print("-------------TEST DATA---------\n"+str(test_data_cleaned))

		no_tuples = (test_data_df.shape[0])

		f_array_test_data = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()
		f_array_test_data_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size = f_array_test_data.nbytes)

		#print(no_attributes_per_part)

		#print(test_data_cleaned.columns)

		test_data_input = test_data_cleaned[col_names_phase2[0]]
		for l in range(1, len(col_names_phase2)):
			#print(input_data[col_names_phase2[l]])
			test_data_input = pd.concat([test_data_input, test_data_cleaned[col_names_phase2[l]]], axis = 1)

		#print(test_data_input)

		X = pd.DataFrame(test_data_input)

		#print(X)

		#print(len_of_each_part)

		test_col_names = []

		for name in col_names_phase2:
			for n in name:
				test_col_names.append(n)

		#print(test_col_names)
		################Normalization#################
		from sklearn.preprocessing import MinMaxScaler
		test_norm_scale = MinMaxScaler()
		#test_norm_data = test_norm_scale.fit_transform(X.values)
		#X = pd.DataFrame(test_norm_data, columns = test_col_names)
		################Normalization#################

		#print(X)

		test_data_input = X

		test_data_input = np.array(test_data_input.values.ravel(), dtype = np.float32)
		test_input_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = test_data_input)

		#print(test_data_input)

		a=np.array(np.ones(no_hyper * no_partitions * no_neurons).T, dtype = np.float32)
		c = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = a, size = a.nbytes)
		
		number_of_datapoints = np.array(no_tuples, dtype=np.int32)
		number_of_datapoints_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = number_of_datapoints)

		distances = np.array(np.zeros((no_neurons * no_partitions * no_hyper,)).T, dtype = np.float32)
		distances_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = distances.nbytes)

		f_array_test_final = np.array(np.zeros(no_tuples * no_hyper * no_partitions,).T, dtype = np.float32).ravel()

		program.inference_test_set(queue, (no_work_units_x, no_work_units_y), None, map_buf_1, test_input_buf_1, distances_buf_1, weights_array_buf, radius_map_buf, no_partitions_buf, no_neurons_buf, no_nets_buf, len_buffer_1, no_components_buf, input_start_point_buf, active_centers_buf, number_of_hyper_buf, f_array_test_data_buf, number_of_datapoints_buf, no_components_buf, neurons_per_net_buf, c)
		
		cl.enqueue_copy(queue, f_array_test_data, f_array_test_data_buf)

		f_array_test_final = f_array_test_final + f_array_test_data

		#print(f_array_test_final)
		
		#print(len(f_array_test_final))

		f_array_test_part = []
	
		min_rmse_val = 1e+20
	
		#print("-----------------------------------------")

		for j in range(0, no_partitions * no_hyper):
			f_array_per_part = []
			f_array_per_part_unnorm = []
			i = 0
			for x in range(j, len(f_array_test_data), no_partitions * no_hyper):
				f_array_per_part_unnorm.append(f_array_test_data[x])
				f_array_per_part.append(min_denormalize + (f_array_test_data[x] * (max_denormalize - min_denormalize)))
				i = i + 1
			f_array_test_part.append(f_array_per_part_unnorm)

			if target in test_input_column_names:
				test_target_arr = test_data_df.loc[:, test_data_df.columns == target]
				#print("Boost:"+str(test_target_arr))
				if test_target_arr.dtypes[0] == object:
					print("Wrong test file given!")
					exit(1)	
	
				if np.isnan(f_array_per_part).any() == False:
					rms = sqrt(mean_squared_error(test_target_arr, f_array_per_part))
						
					if rms < min_rmse_val and np.isnan(rms) == False:
						min_rmse_val = rms
						min_rmse_part = j
					
					#print("Hyper:"+str((j/no_partitions))+" Partition: "+str((j%no_partitions))+" RMSE: "+str(rms))
		
		#print("Boosted Min RMSE PArt: "+str(min_rmse_part)+" RMSE Value: "+str(min_rmse_val))			


		#print("--------------------------------------------EXITING STACKING TESTING PROCESS-------------------------------------------------")
		
		#print(len(train_part_errors))
		
		stack_errors_str = str(error_train[min_rmse_part]) + ":" + str(error_test[min_rmse_part])
		
		train_rms = error_train[min_rmse_part] * (max_denormalize - min_denormalize) #* std_denormalize
		test_rms = error_test[min_rmse_part] * (max_denormalize - min_denormalize)
		
		#print(f_array_test_part[min_rmse_part])
		print("Complete pyopencl_boost")
		return f_array_test_part[min_rmse_part], [train_rms, test_rms]
		

