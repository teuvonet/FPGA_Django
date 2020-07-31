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
import Phase2 as Phase2_File
import Phase1 as Phase1_File
import phase3 as Phase3_File
from scipy import stats as s
#import os
np.set_printoptions(threshold=sys.maxsize)
# TO - DO###################################################################################################
os.environ['PYOPENCL_NO_CACHE'] = '1'
# Master Code:
# Class Wise Feature Ranking Check
# Test Throughly - Diff learning rates, diff partitions
# Check Inline Comments
# Consistent naming convention
# Check with Nitish on chnages in phase 1 "+" instead of "-"
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#platforms = cl.get_platforms()
#context = cl.create_some_context()
#queue = cl.CommandQueue(context)



#Encoding Target into differnt classes. This is used to convert unique string values into integers.
def encode_target(df, target_column, label_dict):

	df_mod = df.copy() #Make DeepCopy
	targets = df_mod[target_column].unique() #Take Unique Values among Target
	for n, name in enumerate(targets):
		label_dict[n] = name

	map_to_int = {name: n for n, name in enumerate(targets)}
	df_mod["Encode "+str(target_column)] = df_mod[target_column].replace(map_to_int)

	return (df_mod, targets)

def call_phase1(Input_Data, Target_Data, Num_Classes, NUM_ITERATIONS, LEARNING_RATES_PHASE_1, NET_LATTICE_PHASE_1, no_data_Passes_kernel_Phase_1, SIZE_OF_PARTITIONS, NUM_DATASET_PASSES, CHUNK_SIZE, platforms, context, queue):
    #platforms = cl.get_platforms()
    #context = cl.create_some_context()
    #queue = cl.CommandQueue(context)
    #Calculated Values
    TOTAL_NUM_FEATURES = Input_Data.shape[1] #Number of features in dataset
    NUM_DATAPOINTS_OVERALL = Input_Data.shape[0] #Number of tuples/datapoints
    NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_1) #Number of learning rates
    myIntType = np.int32 #int Type
    myFloatType = np.float32 #float Type

    #Creating different Partition
    FeatureSpacesPartitions = int(np.ceil(float(TOTAL_NUM_FEATURES)/SIZE_OF_PARTITIONS)) # Number of Partitions


    #Allocating Features per partition based on max size
    if TOTAL_NUM_FEATURES%FeatureSpacesPartitions==0:
        list_num_features_partitions = [int(np.ceil(float(TOTAL_NUM_FEATURES)/FeatureSpacesPartitions)) for _ in range(FeatureSpacesPartitions)]
    else:
        list_num_features_partitions = [int(np.ceil(float(TOTAL_NUM_FEATURES)/FeatureSpacesPartitions)) for _ in range(FeatureSpacesPartitions-1)]+[TOTAL_NUM_FEATURES % SIZE_OF_PARTITIONS]

    NUM_PARTITIONS = len(list_num_features_partitions)

    #Finding cumulative number of features per partition --- Easy for Indexing in Kernel Code
    cumulative_num_features_partition = []
    cum = 0
    cumulative_num_features_partition.append(0)
    for v in range(NUM_PARTITIONS):
        cum += list_num_features_partitions[v]
        cumulative_num_features_partition.append(cum)

    total_num_features_all_partitions = sum(list_num_features_partitions) #TOTAL_NUM_FEATURES

    overall_distances_all = [] #List of list ->  distance for every shuffle
    feature_order = np.arange(0,Input_Data.shape[1],1) #Initial Feature Order
    #print("Shape of feature order",np.shape(feature_order))
    #print("The feature order are ",feature_order)

    

    #Call Phase1 for every shuffle and maintain distances from all shuffle trails
    for it in range(NUM_ITERATIONS):
        #print("\nIteration {}".format(it+1))
        np.random.shuffle(feature_order)
        Input_Data = Input_Data.iloc[:,feature_order]
        #print("Feature Order : {}".format(feature_order))
        #print("Shape of feature order",np.shape(feature_order))
        # print(np.argsort(feature_order))
        temp = Phase1_File.phase1(NET_LATTICE_PHASE_1, LEARNING_RATES_PHASE_1, Input_Data, NUM_ITERATIONS, NUM_DATAPOINTS_OVERALL,
                        no_data_Passes_kernel_Phase_1, Num_Classes, Target_Data, cumulative_num_features_partition,
                        total_num_features_all_partitions, NUM_PARTITIONS, list_num_features_partitions, TOTAL_NUM_FEATURES, NUM_DATASET_PASSES, CHUNK_SIZE, platforms, context, queue)
        #print("value of temp before reshape",temp)
        
        #print("shape of temp overall distance from phase1",np.shape(temp))
        temp = np.reshape(temp, (NUM_LEARNING_RATES, TOTAL_NUM_FEATURES, Num_Classes))
        #print("value of temp",temp)
        #print("shape of temp",np.shape(temp))
        overall_distances_all.append(temp[:, np.argsort(feature_order), :])

    #Taking the mean of all values
    final_overall_distances = np.mean(np.array(overall_distances_all, dtype=np.float32),axis=0)
    #print("final_over_all_distances",final_overall_distances )
    avg_learningRates = np.mean(np.array(final_overall_distances, dtype=np.float32),axis=0)


    class_wise_ranking = defaultdict(list)

    for i, feature in enumerate(avg_learningRates):
        for j, clas in enumerate(feature):
            class_wise_ranking[j].append(clas)

    ranks = defaultdict(list)
    for k in range(Num_Classes):
        distances_from_this_class = class_wise_ranking[k]
        d_ranking = np.sort(distances_from_this_class)[::-1]
        #print("shape of d_ranking",np.shape(d_ranking))
        #print("class : {}".format(k))
        #print(d_ranking[:10])
        f_ranking = np.argsort(distances_from_this_class)[::-1]
        #print(f_ranking[:10])
        ranks[k] = f_ranking
    print("rank",ranks[k])
    # ranks = defaultdict(list)
    # for k in range(Num_Classes):
    #     distances_from_this_class = class_learning_distances[k]

    #     d_ranking = np.array(np.zeros((TOTAL_NUM_FEATURES * NUM_LEARNING_RATES)), dtype = np.float32)
    #     f_ranking = np.array(np.zeros((TOTAL_NUM_FEATURES * NUM_LEARNING_RATES)), dtype = np.int32)
    #     for i in range(NUM_LEARNING_RATES):
    #         d_ranking[i*total_num_features_all_partitions:(i+1)*total_num_features_all_partitions] = np.sort(distances_from_this_class[i*total_num_features_all_partitions:(i+1)*total_num_features_all_partitions])[::-1][:total_num_features_all_partitions]
    #         f_ranking[i*total_num_features_all_partitions:(i+1)*total_num_features_all_partitions] = np.argsort(distances_from_this_class[i*total_num_features_all_partitions:(i+1)*total_num_features_all_partitions])[::-1][:total_num_features_all_partitions]

    #     index = 0
    #     for i in range(total_num_features_all_partitions):
    #         l = []
    #         p = []
    #         for j in range(NUM_LEARNING_RATES):
    #             l.append(f_ranking[j*total_num_features_all_partitions+i])
    #             p.append(d_ranking[j*total_num_features_all_partitions+i])
    #         # if index < 50:
    #         #     print(len(set(l)))
    #         #     index += 1
    #         ranks[k].append(int(s.mode(l)[0]))


    # print("\nFeature ranking")
    # for key in ranks:
    #     print("Class : {}".format(key), end = " ")
    #     print("Ranking : {}".format(len(set(ranks[key]))))
  
    return ranks




def call_phase2(Input_Data, Target_Data, FeatureRankingByClass, Num_Classes, LEARNING_RATES_PHASE_2, NET_LATTICE_PHASE_2, no_data_Passes_kernel_Phase_2,
Max_FeatureSpaces_Size, train_crossvalidation_percentage, NUM_DATASET_PASSES, CHUNK_SIZE, platforms, context, queue):
    #print("Feature ranking by class",FeatureRankingByClass)
    #platforms = cl.get_platforms()
    #context = cl.create_some_context()
    #queue = cl.CommandQueue(context)
    #Feature Order Arrangement based on Feature Ranking
    #Class A : ClassA_Feature1, ClassA_Feature2, ClassA_Feature3
    #Class B : ClassB_Feature1, ClassB_Feature2, ClassB_Feature3
    #New Feature Order: ClassA_Feature1, ClassB_Feature1, ClassA_Feature2, ClassB_Feature2, ClassA_Feature3, ClassB_Feature3
    maxLimit = 0
    for clas in FeatureRankingByClass:
        maxLimit = max(maxLimit, len(FeatureRankingByClass[clas])) #Determine maxfeatures among all classes

    New_Feature_Order = []
    seen = set()


    for i in range(maxLimit):
        for clas in FeatureRankingByClass:
            if FeatureRankingByClass[clas][i] not in seen:
                val = FeatureRankingByClass[clas][i]
                New_Feature_Order.append(val)
                seen.add(val)
    print("New_Feature_Order",New_Feature_Order)
    TOTAL_NUM_FEATURES = len(New_Feature_Order) #New Feature Order
    list_num_feature_featurespaces = [i + 1 for i in range(min(TOTAL_NUM_FEATURES, Max_FeatureSpaces_Size))] #Feature Partitions
    total_num_features_all_featurespaces = sum(list_num_feature_featurespaces) #TOTAL_NUM_FEATURES
    NUM_PARTITIONS = len(list_num_feature_featurespaces)


    #Finding cumulative number of features per partition --- Easy for Indexing in Kernel Code
    cumulative_no_features_featurespace = []
    cum = 0
    cumulative_no_features_featurespace.append(0)
    for v in range(NUM_PARTITIONS):
        cum += list_num_feature_featurespaces[v]
        cumulative_no_features_featurespace.append(cum)


    NUM_DATAPOINTS = Input_Data.shape[0] #Number of tuples/datapoints
    crossValidation_percentage = 1 - train_crossvalidation_percentage
    Input_Train_Data = Input_Data.head(int(train_crossvalidation_percentage * NUM_DATAPOINTS)) #Get the Train data
    Train_Target = Target_Data.head(int(train_crossvalidation_percentage * NUM_DATAPOINTS)) #Get the target data for train data
    #print("new feature order is here")
    #Call Phase 2 Training
    map_data, active_centers_dominant_class, active_centers_total_count, active_centers_per_class, active_dominant_class_count, radius_map = \
        Phase2_File.phase2(Train_Target,
                            NET_LATTICE_PHASE_2,
                            LEARNING_RATES_PHASE_2,
                            Input_Train_Data,
                            FeatureRankingByClass,
                            Num_Classes,
                            Max_FeatureSpaces_Size,
                            no_data_Passes_kernel_Phase_2,
                            NUM_DATASET_PASSES,
                            CHUNK_SIZE,
                            train_crossvalidation_percentage,
                            New_Feature_Order,
                            list_num_feature_featurespaces,
                            total_num_features_all_featurespaces,
                            cumulative_no_features_featurespace, platforms, context, queue)
    #print("map_data",np.shape(map_data))
    #print("active_centers_dominant_class", np.shape(active_centers_dominant_class))
    #print("active_centers_total_count",np.shape(active_centers_total_count) )
    #print("active_centers_per_class", np.shape(active_centers_per_class))
    #print(" active_dominant_class_count",np.shape(active_dominant_class_count))
    #print("radius_map", np.shape(radius_map))

    #commenter("Phase 3")
    #Phase3 - Train Data - Calculates new prediction and Probability
    Train_Accuracies, Train_Predictions,Train_prec,Train_recall,Train_fscore = Phase3_File.Confidence_KNN(Input_Train_Data,
                                New_Feature_Order,
                                NET_LATTICE_PHASE_2,
                                Num_Classes,
                                active_centers_dominant_class,
                                active_centers_total_count,
                                active_dominant_class_count,
                                active_centers_per_class,
                                map_data,
                                radius_map,
                                LEARNING_RATES_PHASE_2,
                                Train_Target,
                                list_num_feature_featurespaces,
                                total_num_features_all_featurespaces,
                                cumulative_no_features_featurespace,platforms, context, queue)

    if crossValidation_percentage == 0:
        print("Train Accuracies")
        for i in zip(np.sort(Train_Accuracies)[::-1], np.argsort(Train_Accuracies)[::-1]):
            print(i)
        bestFeatureSpace = np.argsort(Train_Accuracies)[::-1][0]
        print("best Feature Spacae : {}".format(bestFeatureSpace))
        print("Learning rate : {}".format(bestFeatureSpace // NUM_PARTITIONS))
        print("Best Partition : {}".format(bestFeatureSpace % NUM_PARTITIONS))
        #print(Train_Predictions[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])

        index = 0
        # bestFeatureSpace = len(LEARNING_RATES_PHASE_2)
        #Intial Parameters
        print("Active cenetrs count for the best feature space")
        NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2)
        NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE_PHASE_2] # [9,16,25]
        TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
        active_centers_per_class = active_centers_per_class.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS, Num_Classes)

        
        active_dominant_django=active_dominant_class_count.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS)
        active_centers_dominant_class_django=active_centers_dominant_class.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS)
        active_centers_total_count_django=active_centers_total_count.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS)

        active_dominant_django_val=(active_dominant_django[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])
        active_centers_dominant_class_django_val=(active_centers_dominant_class_django[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])
        active_centers_total_count_django_val=(active_centers_total_count_django[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])
        print("BEST DOMINANT CLASS",active_centers_total_count_django_val)
        for i in active_centers_per_class:
            for j in i:
                if index == bestFeatureSpace:
                    for t, k in enumerate(j):
                        print(t, k)
                        # if(t == t + NUM_NEURONS_PER_NET[index]):
                        #print("\n")
                    print("ok")
	            index += 1
            print("ok")
        print(bestFeatureSpace)



        #
        # NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2)
        # NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE_PHASE_2] # [9,16,25]
        # TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
        # active_centers = active_centers.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS, Num_Classes)
        # print(active_centers[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])

        return New_Feature_Order, NET_LATTICE_PHASE_2, Num_Classes, active_centers_dominant_class, active_centers_total_count, active_dominant_class_count, active_centers_per_class, \
            map_data, radius_map, LEARNING_RATES_PHASE_2, list_num_feature_featurespaces, total_num_features_all_featurespaces, cumulative_no_features_featurespace, bestFeatureSpace, \
            active_dominant_django_val,active_centers_dominant_class_django_val,active_centers_total_count_django_val
    else:
        #Phase3 - Cross Validation - Calculates new prediction and Probability
        Input_CrossValidation_Data = Input_Data.tail(int(crossValidation_percentage * NUM_DATAPOINTS)) #Get the Cross Validation data
        CrossValidation_Target = Input_Target_Encoded.tail(int(crossValidation_percentage * NUM_DATAPOINTS)) #Get the Target data for Cross Validation data
        CrossValidation_Accuracies, CrossValidation_Predictions,CrossValidation_prec,CrossValidation_recall,CrossValidation_fscore  = Phase3_File.Confidence_KNN(Input_CrossValidation_Data,
                                                                New_Feature_Order,
                                                                NET_LATTICE_PHASE_2,
                                                                Num_Classes,
                                                                active_centers_dominant_class,
                                                                active_centers_total_count,
                                                                active_dominant_class_count,
                                                                active_centers_per_class,
                                                                map_data,
                                                                radius_map,
                                                                LEARNING_RATES_PHASE_2,
                                                                CrossValidation_Target,
                                                                list_num_feature_featurespaces,
                                                                total_num_features_all_featurespaces,
                                                                cumulative_no_features_featurespace)


        temp = []
        Train_CrossValidation_sum = 0
        Train_CrossValidation_diff = 0

        #print("Length of Train Accuracies", len(Train_Accuracies))
        #print("Length of CrossValidation_Accuracies", len(CrossValidation_Accuracies))
        #print("list_num_feature_featurespaces",len(list_num_feature_featurespaces))

        for i in range(len(Train_Accuracies)):
            Train_CrossValidation_sum = Train_Accuracies[i] + CrossValidation_Accuracies[i]
            Train_CrossValidation_diff = abs(Train_Accuracies[i] - CrossValidation_Accuracies[i])
            current_Acuuracy = train_crossvalidation_percentage * (Train_CrossValidation_sum / 2) + crossValidation_percentage * Train_CrossValidation_diff
            temp.append((Train_Accuracies[i], CrossValidation_Accuracies[i], abs(Train_Accuracies[i]-CrossValidation_Accuracies[i]), i % TOTAL_NUM_FEATURES, current_Acuuracy))

        temp = sorted(temp, key = lambda k : k[-1])
        for i in temp:
            print(i)

        bestFeatureSpace = temp[-1][-2]
        #print("Active cenetrs count for the best feature space")
        NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2)
        NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE_PHASE_2] # [9,16,25]
        TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
        active_centers = active_centers.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS, Num_Classes)
        for i in active_centers:
            for j in i:
                if index == 0:
                    for t, k in enumerate(j):
                        print(t, k)
                        # if(t == t + NUM_NEURONS_PER_NET[index]):
                        # print("\n")

                index += 1

        print(bestFeatureSpace)
        # print("bestFeatureSpace : {}".format(bestFeatureSpace))
        # print("Active cenetrs count for the best feature space")
        # NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2)
        # NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE_PHASE_2] # [9,16,25]
        # TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
        # active_centers = active_centers.reshape(NUM_LEARNING_RATES , NUM_PARTITIONS , TOTAL_NUM_NEURONS, Num_Classes)
        # print(active_centers[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])

        return New_Feature_Order, NET_LATTICE_PHASE_2, Num_Classes, active_centers_dominant_class, active_centers_total_count, active_dominant_class_count, active_centers_per_class, \
            map_data, radius_map, LEARNING_RATES_PHASE_2, list_num_feature_featurespaces, total_num_features_all_featurespaces, cumulative_no_features_featurespace, bestFeatureSpace, \
            active_dominant_django_val,active_centers_dominant_class_django_val,active_centers_total_count_django_val


def testing(Input_TestData, New_Feature_Order, NET_LATTICE_PHASE_2, Num_Classes, active_centers_dominant_class, active_centers_total_count, active_dominant_class_count, active_centers_per_class,
map_data, radius_map, LEARNING_RATES_PHASE_2, Input_Test_Target_Encoded, list_num_feature_featurespaces, total_num_features_all_featurespaces, cumulative_no_features_featurespace, bestFeatureSpace, platforms, context, queue):

    Test_Accuracies, Test_Predictions,Test_prec,Test_recall,Test_fscore = Phase3_File.Confidence_KNN(Input_TestData,
                                                            New_Feature_Order,
                                                            NET_LATTICE_PHASE_2,
                                                            Num_Classes,
                                                            active_centers_dominant_class,
                                                            active_centers_total_count,
                                                            active_dominant_class_count,
                                                            active_centers_per_class,
                                                            map_data,
                                                            radius_map,
                                                            LEARNING_RATES_PHASE_2,
                                                            Input_Test_Target_Encoded,
                                                            list_num_feature_featurespaces,
                                                            total_num_features_all_featurespaces,
                                                            cumulative_no_features_featurespace,platforms, context, queue)

   
    print("bestFeatureSpace",bestFeatureSpace)
    NUM_PARTITIONS = len(list_num_feature_featurespaces)
    print("Test Accuracies")
    for i in zip(np.sort(Test_Accuracies)[::-1], np.argsort(Test_Accuracies)[::-1]):
        print("zip output",i)
    #print("Input_TestData",Input_TestData)
    #print("Test Prediction raw",Test_Predictions)                                                      
    print("Test_predictions",Test_Predictions[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])
    predict=np.array(Test_Predictions[bestFeatureSpace // NUM_PARTITIONS][bestFeatureSpace % NUM_PARTITIONS])
    Input_TestData=np.array(Input_TestData)
    Input_Test_Target_Encoded=np.array(Input_Test_Target_Encoded)
    #print("radius map",radius_map)
    #np.savetxt("MethodB.csv", zip(Input_Test_Target_Encoded,predict), delimiter=",",header="Actual Output, Predict output",fmt="%s", comments='')
    print("Test Accuracies")
    return Test_Accuracies[bestFeatureSpace],Test_prec[bestFeatureSpace],Test_recall[bestFeatureSpace],Test_fscore[bestFeatureSpace]

def load_data_and_targets(path_to_train_file, path_to_test_file):

    #Reading the files
    TrainData = pd.read_csv(path_to_train_file)
    TestData = pd.read_csv(path_to_test_file)

    from sklearn.utils import shuffle
    #TrainData = shuffle(TrainData)
    #TestData = shuffle(TestData)

    #Train and Test
    Input_TrainData = TrainData.iloc[:, :-1]
    Input_TestData = TestData.iloc[:,:-1]
    Target = TrainData.columns[-1]
    #Combining Test and Target data for consistent distributed encoding
    #Combine_Train_Test_DistributedEncoding = pd.concat([Input_TrainData, Input_TestData], keys=[0, 1], sort = False)

    #Target Train Encoding
    Encoding_Map = {} #Encoding Map Reference
    Target_Encoded, col = encode_target(TrainData, Target, Encoding_Map) #calling encode_target function to map the classes
    Input_Target_Encoded = Target_Encoded.loc[:, Target_Encoded.columns == "Encode "+str(Target)] #Copying back the target
    Num_Classes = len(set(Input_Target_Encoded.values.T.tolist()[0])) #Calculating Number of Target Classes
    #print("NUMCLASSESSS",Num_Classes)
    #Target Test Encoding
    Encoding_Map = {} #Encoding Map Reference
    Target_Encoded, col = encode_target(TestData, Target, Encoding_Map) #calling encode_target function to map the classes
    Input_Test_Target_Encoded = Target_Encoded.loc[:, Target_Encoded.columns == "Encode "+str(Target)] #Copying back the target
    Num_Classes = len(set(Input_Test_Target_Encoded.values.T.tolist()[0])) #Calculating Number of Target Classes


    # #Distributed Encoding for the Train Data
    # DE_Train = distEncode.dist_encode("binary")
    # maps, inv_maps, col_maps = DE_Train.fit(Combine_Train_Test_DistributedEncoding)
    # temp = DE_Train.transform(Combine_Train_Test_DistributedEncoding, maps = maps, maps_dir = "./maps.txt", col_maps = col_maps) #Applying Distributed Encoding
    # Combine_Train_Test_DistributedEncoding = temp #copying Back the Data
    # import pickle
    # features = pickle.load(open('col_maps.txt', "rb"))
    # for key in col_maps:
    #     for col in col_maps[key]:
    #         Combine_Train_Test_DistributedEncoding[col] = Combine_Train_Test_DistributedEncoding[col].astype(np.int32)

    # #Divideback into Train and test
    # Input_TrainData = Combine_Train_Test_DistributedEncoding.xs(0)
    # Input_TestData = Combine_Train_Test_DistributedEncoding.xs(1)
    Input_columns = Input_TrainData.columns # Read the column names

    #Normalizing both Train and Test data
    from sklearn.preprocessing import MinMaxScaler
    train_norm_scale = MinMaxScaler()
    Input_TrainData = train_norm_scale.fit_transform(Input_TrainData)
    Input_TrainData = pd.DataFrame(Input_TrainData, columns = Input_columns)
    Input_TestData = train_norm_scale.fit_transform(Input_TestData)
    Input_TestData = pd.DataFrame(Input_TestData, columns = Input_columns)

    #print(Input_TrainData)
    #print(Input_TestData)
    #print(Input_Target_Encoded[:5])
    # print(Input_Test_Target_Encoded)

    return Input_TrainData, Input_TestData, Input_Target_Encoded, Input_Test_Target_Encoded, Target, Num_Classes

