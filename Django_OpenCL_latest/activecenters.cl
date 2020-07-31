/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.

*/

//#include<stdio.h>

__kernel void activecenters(
                        int TOTAL_NUM_FEATURES, //Number of features
                        int No_Partitions, //Number of partitions
                        int NUM_DATAPOINTS_OVERALL, //Toal number of datapoints in the given chunk
                        int TOTAL_NUM_NEURONS, //Total neurons if 3,4,5 nets = 50 neurons
                        int NUM_NETS, //Nets used //3,4,5 = 3
                        int NUM_CLASSES, //Total Number of classes 
                        int total_num_features_all_featurespaces,
                        __global int* NUM_FEATURES_PER_PARTITION, //How many Features are there in each partition [100, 100, 20]
                        __global int* cumulative_no_features_partition, //[0, 100, 200, 220] This is used for Indexing to skip the festures
                        __global int* cumulative_no_neurons_per_net,
                        __global float* Input_Data, //Given Input Data after distributed encoding
                        __global float* map, //base map for every neuron and its corresponding feature value
                        __global float* distance_map, //Buffer Array to store cumulative sum of distances from each neuron to its feature
                        __global int* target_arr, //Target Data
                        __global int* min_pos, //Used for finding the winning neuron - Stores Index
                        __global float* min_array, //Used for finding the winning neuron - Stores distance
                        __global int* active_centers_per_class, //Array to store how many datapoints for each class for each neuron
                        __global int* active_centers_dominant_class, //Arrayto store dominating class for each neuron
                        __global int* active_centers_total_count,
                        __global int *active_dominant_class_count,
                        __global int* neurons_per_nets, //Number of neurons per net [9, 16, 25]
                        __global int *InputIndex_Per_Partition
                    )
{
    // Finding the total number of threads in the program
    //const int block_skip = get_local_size(2)*get_local_size(1)*get_local_size(0);

    // Finding thread index using block and thread index
    const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0);
    
    //Gathering Initial values
    int Partition_Block_Id = get_global_id(0)/TOTAL_NUM_NEURONS; //Partition Number
    int no_of_features_partition = NUM_FEATURES_PER_PARTITION[Partition_Block_Id]; //Get the Number of features that in each feature space
    int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces;
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_partition[Partition_Block_Id];
    map_start_index += (thread_id % TOTAL_NUM_NEURONS) * no_of_features_partition;
    
    
    for(int datapoint = 0; datapoint < NUM_DATAPOINTS_OVERALL; datapoint++)
    {
            //Looping thorugh each feature for every neuron and finding the cumulative sum between the neuron and Input data
            float sum = 0;
            int Features_Skip_Partition = InputIndex_Per_Partition[Partition_Block_Id];
            
            for (int feature = Features_Skip_Partition; feature < Features_Skip_Partition + no_of_features_partition ; feature++)
            {
                sum += fabs(Input_Data[datapoint * TOTAL_NUM_FEATURES  + feature] - map[map_start_index + feature % no_of_features_partition]);
                //printf("\nmap : %f",map[map_start_index + feature % no_of_features_partition]);
            }
            // if(datapoint == 0 && Partition_Block_Id == 71 && learning_rate_Index == 2)
             //printf("\nThread_id : %d distance : %f ", thread_id, sum);
            //Index in the distance = thread_id 
            distance_map[thread_id] = sum;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);



            // _____________________________________________________________________________________________________________________________________________________
            // STEP 2: Calculating minium distance Neuron
            // ----------------------------------------------------------------------------------------------------------------------------------------------------
            //  Find the winner and increase the count of corresponding class of that neuron 

            if(get_global_id(0)%TOTAL_NUM_NEURONS == 0)
            {
                //printf("%2d %2d\n", thread_id, first_thread_from_each_block);
                //Distance Map index adjusting
                int distance_map_Index = thread_id;
                int min_pos_startIndex = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;
                int active_centers_Index = thread_id * NUM_CLASSES;

                //Loop through each net
                int net = 0;
                for(; net < NUM_NETS; net++)
                {
                    min_array[min_pos_startIndex + net] = INT_MAX;
                    //Loop through every neuron in that net
                    for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                    {
                        // if(Partition_Block_Id == 0 && datapoint == 0 && learning_rate_Index == 0){
                            // printf("\ndistance : %f, net :%d, neuron :%d, min_dist :%f", distance_map[distance_map_Index + neuron], net, neuron, min_array[min_pos_startIndex + net]);
                        // }
                        // printf("Chunck %d learningrate %d thread_id %d neuron %d distance %f minDist %f\n", datapoint, learning_rate_Index, thread_id, neuron, distance_map[distance_map_Index + neuron], min_dist_array[min_pos_startIndex + net]);
                        if(distance_map[distance_map_Index + neuron] < min_array[min_pos_startIndex + net])
                        {
                            //Capture min dist array neuron and it's Index
                            min_array[min_pos_startIndex + net] = distance_map[distance_map_Index + neuron];
                            min_pos[min_pos_startIndex + net] = neuron;
                        }
                    }

                    int winner = min_pos[min_pos_startIndex + net];
                    // if(Partition_Block_Id == 0 && datapoint == 0 && learning_rate_Index == 0){
                         //printf("\nnet :%d, neuron :%d, winner :%d", net, thread_id, winner);
                    // }
                    // printf("\nThread_id : %d Winner : %d active center before : %d", thread_id, winner,  active_centers[active_centers_Index + winner * NUM_CLASSES + target_arr[datapoint]]);
                    active_centers_per_class[active_centers_Index + winner * NUM_CLASSES + target_arr[datapoint]] += 1;
                    // if(Partition_Block_Id == 10 && datapoint == 30 && learning_rate_Index == 0){
                         //printf("\n net :%d, winner : %d actStart : %d acticeCentersIndex : %d datapointClass: %d, countClass0: %d countClass1 :%d",net, winner, active_centers_Index, active_centers_Index + winner * NUM_CLASSES, target_arr[datapoint], active_centers_per_class[active_centers_Index + winner * NUM_CLASSES + 0], active_centers_per_class[active_centers_Index + winner * NUM_CLASSES + 1] );
                    // }
                    // printf("\nThread_id: %d Winner : %d active center after : %d", thread_id, winner,  active_centers[active_centers_Index + winner * NUM_CLASSES + target_arr[datapoint]]);
                    
                    //Adjust distance map after every net
                    distance_map_Index += neurons_per_nets[net];
                    active_centers_Index += neurons_per_nets[net] * NUM_CLASSES;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
    }
    
    int active_centers_Index = thread_id * NUM_CLASSES;
    
    int total_datapoints_per_neuron_allclass = 0;
    for(int clas = 0;  clas < NUM_CLASSES; clas++)
    {
        total_datapoints_per_neuron_allclass = total_datapoints_per_neuron_allclass + active_centers_per_class[active_centers_Index + clas];
       
    }

    active_centers_total_count[thread_id] = total_datapoints_per_neuron_allclass;

    if( total_datapoints_per_neuron_allclass == 0)
    {
        active_centers_dominant_class[thread_id] = -1;
        active_dominant_class_count[thread_id] = 0;
    }
    else	
    {
        float maxPercent = 0;
        int maxPercentClass = -1;
        for(int clas = 0; clas < NUM_CLASSES; clas++)
        {
            
            float percent_class = (float)(active_centers_per_class[active_centers_Index + clas]) / (float)(total_datapoints_per_neuron_allclass);
            if(percent_class > maxPercent){
                maxPercent = percent_class;
                maxPercentClass = clas;
            }
        }
        float condition = (float)(1) / (float)(NUM_CLASSES);
        condition += 0.05;
        if(maxPercent >= condition)
        {
            active_dominant_class_count[thread_id] = active_centers_per_class[active_centers_Index + maxPercentClass];
            active_centers_dominant_class[thread_id] = maxPercentClass;
        } 
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
