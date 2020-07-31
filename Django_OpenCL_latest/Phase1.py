'''
This code is licensed and documented by Teuvonet Technologies. 
Any use of this code, proprietary or personal, needs approval from the company.

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import time
import imp
from pandas import *
import pyopencl as cl
import pyopencl.algorithm as algo
import pyopencl.array as pycl_array
import sklearn.metrics as met
from collections import defaultdict
import distEncode
import copy
#import pdb

def phase1(NET_LATTICE, LEARNING_RATES_PHASE1, Input_Data, NUM_ITERATIONS, NUM_DATAPOINTS_OVERALL, 
no_data_Passes_kernel_Phase1, NUM_CLASSES, target_arr, cumulative_no_features_partition, 
total_num_features_all_featurespaces, NUM_PARTITIONS, list_num_features_featurespaces, TOTAL_NUM_FEATURES, NUM_DATASET_PASSES, CHUNK_SIZE, platforms, context, queue):
	np.set_printoptions(threshold=sys.maxsize)
        os.environ['PYOPENCL_NO_CACHE'] = '1'
	#Initial Parameters
	NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE] # [9,16,25]
	TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
	NUM_NETS_PER_BLOCK = len(NUM_NEURONS_PER_NET) #If [3,4,5] = 3
	NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE1) # Number of Learning Rates
	NUM_NETS = len(NET_LATTICE) #[3,4,5] = 3
	NUM_DATAPOINTS = Input_Data.shape[0] #Number of datapoints

	print("\n")
	print("Now doing Phase 1 Details Below\n")
	print("Number of Datapoints :"+str(NUM_DATAPOINTS_OVERALL))
	print("Number of classes in the given data :"+str(NUM_CLASSES))
	print("Net_lattices : " + str(NET_LATTICE))
	print("TOTAL_NUM_FEATURES : " + str(TOTAL_NUM_FEATURES))
	print("Total Number of learning rates : " + str(NUM_LEARNING_RATES))
	print("Number of Partitions : "+str(NUM_PARTITIONS))
	print("Number of Passes of Data in kernel : {}".format(no_data_Passes_kernel_Phase1))
	print("total_num_features_all_featurespaces : {}".format(total_num_features_all_featurespaces))
	print("cumulative_no_features_partition : {}".format(cumulative_no_features_partition))
	print("list_num_features_featurespaces : {}".format(list_num_features_featurespaces))
	print("\n")
        #pdb.set_trace()

	#Reading the kernel File
	cl_filename = "./trainingNeurons.cl" #Reading the file
	with open(cl_filename, 'r') as fd:
		clstr = fd.read()
        #platforms = cl.get_platforms()
	#context = cl.create_some_context()
	#queue = cl.CommandQueue(context)
        mem_flags = cl.mem_flags
	myFloatType = np.float32 #Global Float Type for Kernel
	myIntType = np.int32 #Global int Type for Kernel
	program = cl.Program(context, clstr).build()
	

	# KERNEL BUFFER ARRAYS
	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	
	#Distance map array stores the distance between neuron center and Input data of every feature
	distance_base_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS ), dtype = myFloatType)#write
        #print("distance_base_map", np.shape(distance_base_map))

        distance_base_map_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = distance_base_map)#write buffer
	#Base Map will have the neuron centers. Size = NoOfneuron * NoOfFeatures
	map_data_size = NUM_LEARNING_RATES * TOTAL_NUM_NEURONS * TOTAL_NUM_FEATURES	
	map_data = np.random.uniform(low=0, high=0, size=(map_data_size,)).astype(myFloatType)#write

        map_data_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#write buffer
        #print("map_data", np.shape(map_data))
	#Number of features in each Partition
	NUM_FEATURES_PER_PARTITION = np.array(list_num_features_featurespaces, dtype =  myIntType)#read

        NUM_FEATURES_PER_PARTITION_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)#read buffer
        #print("NUM_FEATURES_PER_PARTITION", np.shape(NUM_FEATURES_PER_PARTITION))
	#learning rates for phase1
	LEARNING_RATES_PHASE1 = np.array(LEARNING_RATES_PHASE1, dtype =  myFloatType)#not passed in kernel
        #print("LEARNING_RATES_PHASE1", np.shape(LEARNING_RATES_PHASE1))
        #LEARNING_RATE = pycl_array.to_device(queue, LEARNING_RATES_PHASE1 )
	#Cumulative Number of features 
	cumulative_no_features_partition = np.array(cumulative_no_features_partition, dtype = myIntType)#read
        
        cumulative_no_features_partition_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_partition)#read buffer
	#Number of Neurons in each net
	NUM_NEURONS_PER_NET = np.array(NUM_NEURONS_PER_NET, dtype = myIntType)#read

        NUM_NEURONS_PER_NET_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
        #print("NUM_NEURONS_PER_NET", np.shape(NUM_NEURONS_PER_NET))
	#Array to store minimum distance neuron for each net.
	min_dist_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myIntType)#write

        min_dist_pos_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = min_dist_pos)#write buffe
	#print("min_dist_pos",np.shape(min_dist_pos))
	#Array to store minimum distance position of a neuron in each net
	min_dist_neuron = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myFloatType)#write

        min_dist_neuron_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = min_dist_neuron)#write buffer
        #print("min_dist_neuron",np.shape(min_dist_neuron))
	#NeighBourRate Array
	neigh_rate = np.array(np.ones(NUM_NETS * NUM_PARTITIONS * NUM_LEARNING_RATES) , dtype = myFloatType) 
	new_neigh_rate = copy.deepcopy(neigh_rate)#write

        new_neigh_rate_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = new_neigh_rate)#write buffer#write buffer
	new_learning_rate_list = copy.deepcopy(LEARNING_RATES_PHASE1)#read

        new_learning_rate_list_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = new_learning_rate_list)#read buffer
	# cumulative weight change
	cumulative_weight_change_per_neuron = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)#write
        
        cumulative_weight_change_per_neuron_buf_1 = cl.Buffer(context, mem_flags.WRITE_ONLY, size = cumulative_weight_change_per_neuron.nbytes)#write buffer
	#This array is helpful for indexing of the Input data at kernel
	InputIndex_Per_Partition = cumulative_no_features_partition
	InputIndex_Per_Partition = np.array(InputIndex_Per_Partition, dtype = myIntType)#read
        InputIndex_Per_Partition_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = InputIndex_Per_Partition)#read buffer
        
        #Buffer array to hold total number of neurons
        TOTAL_NUM_NEURONS = np.array(TOTAL_NUM_NEURONS, dtype = np.int32)
	#TOTAL_NUM_NEURONS_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = TOTAL_NUM_NEURONS)

        #Buffer array to hold number of nets
        NUM_NETS = np.array(NUM_NETS, dtype = np.int32)
	#NUM_NETS_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NETS)

        #Buffer array to hold the number of partitions
        NUM_PARTITIONS = np.array(NUM_PARTITIONS, dtype = np.int32)
        #NUM_PARTITIONS = pycl_array.to_device(queue, NUM_PARTITIONS )
	#NUM_PARTITIONS_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_PARTITIONS)
        

        #Buffer array to hold total number of features
        TOTAL_NUM_FEATURES = np.array(TOTAL_NUM_FEATURES, dtype = np.int32)
	#TOTAL_NUM_FEATURES_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = TOTAL_NUM_FEATURES)

        #Buffer array to hold total number of features in all feature spaces
        total_num_features_all_featurespaces = np.array(total_num_features_all_featurespaces, dtype = np.int32)
	#total_num_features_all_featurespaces_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = total_num_features_all_featurespaces)

        #Buffer array to hold total number of data passes to the kernel of phase 1
        no_data_Passes_kernel_Phase1 = np.array(no_data_Passes_kernel_Phase1, dtype = np.int32)
	#no_data_Passes_kernel_Phase1_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = no_data_Passes_kernel_Phase1)

        NET_LATTICE = np.array(NET_LATTICE, dtype = np.int32)
	NET_LATTICE_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NET_LATTICE)

        NUM_DATASET_PASSES = np.array(NUM_DATASET_PASSES, dtype = np.int32)
	NUM_DATASET_PASSES_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_DATASET_PASSES)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------
        
	#---------------------------------------------------------------------------------------------------------------------------------------------------------
	
	#Define the sizes of the blocks and grids
	#grid_X, grid_Y, grid_Z = NUM_PARTITIONS, NUM_LEARNING_RATES, 1
	#workUnits_X, workUnits_Y, workUnits_Z = TOTAL_NUM_NEURONS, 1, 1
        #Input_Data = Input_Data.values.ravel()
        

	phase1_time = time.time()

	for l in range(NUM_DATASET_PASSES):
		# print("Dataset Pass ", l+1)
		sum_cum_wt_change = 0
		for i in range(0,len(Input_Data),CHUNK_SIZE):
			chunk_inp = Input_Data.iloc[i:i+CHUNK_SIZE,:]# extract the input data with the row number.
			Final_Num_DataPoints = min(len(Input_Data), len(chunk_inp))
                        Final_Num_DataPoints = np.array(Final_Num_DataPoints, dtype = np.int32)
                        #print("Final_Num_DataPoints", Final_Num_DataPoints)
                        #Buffer array to hold the final number of datapoints
	                #Final_Num_DataPoints_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Final_Num_DataPoints)
			#Define the sizes of the blocks and grids
		        grid_X, grid_Y = NUM_PARTITIONS*TOTAL_NUM_NEURONS,NUM_LEARNING_RATES
			#workUnits_X, workUnits_Y, workUnits_Z = , 1, 1
			chunk_inp = chunk_inp.values.ravel() #1D array of the input of 5 chunks.
			chunk_inp = np.array(chunk_inp, dtype =  myFloatType)#read
			chunk_inp_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = chunk_inp)#read buffer
                        #print("Input_Data",chunk_inp)
                
		        phase_1_train = program.trainingNeurons(queue,(grid_X, grid_Y ),None,
		            TOTAL_NUM_NEURONS,
			    NUM_NETS,
			    Final_Num_DataPoints,
			    NUM_PARTITIONS,
			    TOTAL_NUM_FEATURES,
			    total_num_features_all_featurespaces, 
			    no_data_Passes_kernel_Phase1,
			    NUM_FEATURES_PER_PARTITION_buf_1,
			    cumulative_no_features_partition_buf_1,
			    cumulative_weight_change_per_neuron_buf_1,
			    NUM_NEURONS_PER_NET_buf_1,	
			    map_data_buf_1,
			    min_dist_pos_buf_1,
			    min_dist_neuron_buf_1,
			    chunk_inp_buf_1,
			    distance_base_map_buf_1,
			    new_neigh_rate_buf_1,
			    new_learning_rate_list_buf_1,
			    NET_LATTICE_buf_1,
			    InputIndex_Per_Partition_buf_1)
                                #LEARNING_RATE.data)
				#block=(workUnits_X, workUnits_Y, workUnits_Z), grid=(grid_X,grid_Y,grid_Z)) 

			# cumulative_weight_change_per_neuron_new = cumulative_weight_change_per_neuron.reshape((NUM_LEARNING_RATES,-1))
			# sum_cum_wt_change+=np.sum(cumulative_weight_change_per_neuron_new,axis=1)
                        #print("back_to_host")
		if(l%2==0):
			new_neigh_rate = np.array([lr*0.9 for lr in new_neigh_rate], dtype=np.float32)
                        new_neigh_rate_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = new_neigh_rate)#write buffer#write buffer
			new_learning_rate_list = np.array([lr*0.9 for lr in new_learning_rate_list], dtype=np.float32)
                        new_learning_rate_list_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = new_learning_rate_list)#read buffer
                cl.enqueue_copy(queue, map_data, map_data_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
                #print("back_to_host")
                #print("min_dist_pos_buf_1",map_data) 
                cl.enqueue_copy(queue, cumulative_weight_change_per_neuron, cumulative_weight_change_per_neuron_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
                 
                cl.enqueue_copy(queue, min_dist_pos, min_dist_pos_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
                cl.enqueue_copy(queue, min_dist_neuron, min_dist_neuron_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
                cl.enqueue_copy(queue, distance_base_map, distance_base_map_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
                cl.enqueue_copy(queue, new_neigh_rate, new_neigh_rate_buf_1,is_blocking=True, wait_for=[phase_1_train]).wait()
        
	#print("\nTime for phase 1 : {0:.5f} secs".format(time.time()-phase1_time))
        #map_data = np.random.uniform(low=0, high=0.8, size=(map_data_size,)).astype(myFloatType)#write
        #map_data = np.zeros(NUM_LEARNING_RATES * TOTAL_NUM_NEURONS/2 * TOTAL_NUM_FEATURES/4)#write
        #print("map_data",map_data)
        #print("min_dist_pos",min_dist_pos)
        #print("min_dist_neuron",min_dist_neuron)
        #print("map_data_shape",np.shape(map_data))
	# temp = map_data.reshape(NUM_LEARNING_RATES, TOTAL_NUM_NEURONS, TOTAL_NUM_FEATURES)
	# # print(map_data.reshape(NUM_LEARNING_RATES, TOTAL_NUM_NEURONS,  total_num_features_all_featurespaces))
	temp = map_data.reshape(NUM_LEARNING_RATES, TOTAL_NUM_NEURONS, total_num_features_all_featurespaces)
	# print(temp)
	# index = 0
	# for i in temp:
	# 	if index == 1:
	# 		for k, j in enumerate(i):
	# 			print(k, j)
	# 	index += 1
	

	#ACTIVE CENTERS
	##################################################################################################################################################
 
	#Reading the kernel File
	cl_filename = "./activecenters.cl" #Reading the file
	with open(cl_filename, 'r') as fd:
		clstr = fd.read()
        #platforms = cl.get_platforms()
	#context = cl.create_some_context()
	#queue = cl.CommandQueue(context)
        mem_flags = cl.mem_flags
	myFloatType = np.float32 #Global Float Type for Kernel
	myIntType = np.int32 #Global int Type for Kernel
	program = cl.Program(context, clstr).build()

	cumulative_no_neurons_per_net = []
	cumulative = 0
	cumulative_no_neurons_per_net.append(0)
	for val in NUM_NEURONS_PER_NET:
		cumulative += val
		cumulative_no_neurons_per_net.append(val)
     

	#KERNEL BUFFER ARRAYS
	#----------------------------------------------------------------------------------------------------------------------

	Input_Data = Input_Data.values.ravel() #Taking Input in the order of shuffled columns
	Input_Data = np.array(Input_Data, dtype = myFloatType) #Changing to a single dimensional array
        Input_Data_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Input_Data)
	#Cumulative Array of features in each partition array
	cumulative_no_features_partition = np.array(cumulative_no_features_partition, dtype= myIntType)
        #cumulative_no_features_partition_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_partition)
	#Cumulative number of neurons per net
	cumulative_no_neurons_per_net = np.array(cumulative_no_neurons_per_net, dtype= myIntType)
        cumulative_no_neurons_per_net_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_neurons_per_net)
	#Buffer Array to store cumulative sum of distances from each neuron to its feature
	distance_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)
        distance_map_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = distance_map)
	#Converting Target array to int
	target_arr_int = np.array(target_arr.values.ravel(), dtype = myIntType)
        target_arr_int_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = target_arr_int)
	#Array to store minimum distance neuron for each net.
	min_dist_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myIntType)
        min_dist_pos_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_dist_pos)
	#Array to store minimum distance position of a neuron in each net
	min_dist = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_NETS), sys.maxsize), dtype = myFloatType)
        min_dist_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_dist)
	#Array to store how many datapoints for each class for each neuron
	active_centers_per_class = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS * NUM_CLASSES,)).T, dtype = myIntType)
        active_centers_per_class_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_per_class)#write buffer
	#Arrayto store dominating class for each neuron
	active_centers_dominant_class = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)
        active_centers_dominant_class_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_dominant_class)#write buffer#write buffer
	#Active Centers Total Count
	active_centers_total_count = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)
        active_centers_total_count_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_total_count)#write buffer
	#Active Centers Total Count
	active_dominant_class_count = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS,)).T, dtype = myIntType)
        active_dominant_class_count_buf_2 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = active_dominant_class_count)#write buffer

	#Number of features in each partition
	NUM_FEATURES_PER_PARTITION = np.array(list_num_features_featurespaces, dtype = myIntType)
        #NUM_FEATURES_PER_PARTITION_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)
        # conversion to fp32 from argument of phase1 
        NUM_DATAPOINTS_OVERALL = np.array(NUM_DATAPOINTS_OVERALL, dtype = myIntType)
        NUM_CLASSES = np.array(NUM_CLASSES, dtype = myIntType)

        map_data_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#write buffer
        #NUM_NEURONS_PER_NET_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
        #InputIndex_Per_Partition_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = InputIndex_Per_Partition)#read buffer
	#Define the sizes of the blocks and grids
	grid_X, grid_Y = NUM_PARTITIONS*TOTAL_NUM_NEURONS, NUM_LEARNING_RATES
	#noofThreads = int(TOTAL_NUM_NEURONS) #CUDA Doesn't allow np.int64 variables so doing a typecase to int
	#workUnits_X, workUnits_Y, workUnits_Z = noofThreads, 1, 1

	activeCentersTime = time.time()
	# print(NUM_FEATURES_PER_PARTITION)
        #print("Input_Data",Input_Data)
	#Kernel Function call with Parameters , Required In and outs
	phase_1_active_centers = program.activecenters(queue, (grid_X, grid_Y), None,
		TOTAL_NUM_FEATURES,
		NUM_PARTITIONS,
		NUM_DATAPOINTS_OVERALL,
		TOTAL_NUM_NEURONS,
		NUM_NETS,
		NUM_CLASSES,
		total_num_features_all_featurespaces,
		NUM_FEATURES_PER_PARTITION_buf_1,
		cumulative_no_features_partition_buf_1,
		cumulative_no_neurons_per_net_buf_1,
		Input_Data_buf_1,
		map_data_buf_2,
		distance_map_buf_2,
		target_arr_int_buf_1,
		min_dist_pos_buf_1,
		min_dist_buf_1,
		active_centers_per_class_buf_2,
		active_centers_dominant_class_buf_2,
		active_centers_total_count_buf_2,
		active_dominant_class_count_buf_2,
		NUM_NEURONS_PER_NET_buf_1,
		InputIndex_Per_Partition_buf_1)
		#block=(workUnits_X, workUnits_Y, workUnits_Z), grid=(grid_X,grid_Y,grid_Z))
       # print("back_to_host")
       #The write buffer is enqueued for all the argument of type Inout
        
        cl.enqueue_copy(queue, active_centers_per_class, active_centers_per_class_buf_2,is_blocking=True, wait_for=[phase_1_active_centers]).wait()
        cl.enqueue_copy(queue, active_centers_dominant_class, active_centers_dominant_class_buf_2,is_blocking=True, wait_for=[phase_1_active_centers]).wait()
        cl.enqueue_copy(queue, active_centers_total_count, active_centers_total_count_buf_2,is_blocking=True, wait_for=[phase_1_active_centers]).wait()
        cl.enqueue_copy(queue, active_dominant_class_count, active_dominant_class_count_buf_2,is_blocking=True, wait_for=[phase_1_active_centers]).wait()
        
        #print("dominant class",active_dominant_class_count )
        #print("dominant class shape",np.shape(active_dominant_class_count))
        #print("active_centers_per_class",active_centers_per_class)
        #print("active_centers_dominant_class",active_centers_dominant_class)
        #print("active_centers_total_count",active_centers_total_count)
        #print("active_dominant_class_count",active_dominant_class_count)

               
	print("\nTime for Active Centers phase 1 : {0:.5f} secs".format(time.time()- activeCentersTime))
	

	# print("\n")
	# print("Active Centers")
	# print(np.reshape(active_centers, (NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS, NUM_CLASSES)))
	# print("\n active_centers_orig")
	# print(active_centers_orig)
	# index = 0
	# active_centers_per_class = active_centers_per_class.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS, NUM_CLASSES)
	# for i in active_centers_per_class:
	# 	for j in i:
	# 		if index < 2:
	# 			for t, k in enumerate(j):
	# 				print(t, k, end = "\n")
	# 				# if(t == t + NUM_NEURONS_PER_NET[index]):
	# 				# 	print("\n")
	# 		index += 1




	# DISTANCE COMPUTATION
	# ##################################################################################################################################################
	#Reading the kernel File
	cl_filename = "./phase1_distanceComputation.cl" #Reading the file
	with open(cl_filename, 'r') as fd:
		clstr = fd.read()
        #platforms = cl.get_platforms()
	#context = cl.create_some_context()
	#queue = cl.CommandQueue(context)
        mem_flags = cl.mem_flags
	myFloatType = np.float32 #Global Float Type for Kernel
	myIntType = np.int32 #Global int Type for Kernel
	program = cl.Program(context, clstr).build()

        print("starting the phase1 dist computation")
	#KERNEL BUFFER ARRAYS
	#----------------------------------------------------------------------------------------------------------------------
	#An Array to capture distances
	overall_distances = np.array(np.zeros((NUM_LEARNING_RATES * TOTAL_NUM_FEATURES * NUM_CLASSES,)).T, dtype = myFloatType)
        #print("overall_dist",np.shape(overall_distances))
        overall_distances_buf_3 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = overall_distances)#write buffer
        
        #map_data_buf_3 =cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_data) #write buffer
        NUM_NEURONS_PER_NET_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_NEURONS_PER_NET)#read buffer
        active_centers_dominant_class_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_dominant_class)#read buffer
        #NUM_FEATURES_PER_PARTITION_buf_3 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)
        NUM_LEARNING_RATES = np.array(NUM_LEARNING_RATES, dtype = myIntType)
        #print("NUM_LEARNING_RATES",NUM_LEARNING_RATES)
        #NUM_LEARNING_RATES = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_LEARNING_RATES)#read buffer
	grid_X, grid_Y = TOTAL_NUM_FEATURES, NUM_LEARNING_RATES
	#workUnits_X, workUnits_Y, workUnits_Z = 1, 1, 1

	distanceCompTime = time.time()

	phase_1_dis_comp = program.phase1_distanceComputation(queue,(grid_X, grid_Y),None,
		TOTAL_NUM_FEATURES,
		NUM_PARTITIONS,
		TOTAL_NUM_NEURONS,
		NUM_LEARNING_RATES,
		NUM_NETS,
		NUM_CLASSES,
		map_data_buf_1,
		overall_distances_buf_3,
		NUM_NEURONS_PER_NET_buf_1,
		NUM_FEATURES_PER_PARTITION_buf_1,
		active_centers_dominant_class_buf_2)
	cl.enqueue_copy(queue, overall_distances, overall_distances_buf_3)
        cl.enqueue_copy( queue, map_data, map_data_buf_1)	
        #print("Overall distance in Phase1", cl.enqueue_copy( queue, active_centers_per_class, active_centers_per_class_buf_2))
        #print("\nTime for Distance Computation phase 1 : {0:.5f} secs".format(time.time()- distanceCompTime))

      
       	return overall_distances


