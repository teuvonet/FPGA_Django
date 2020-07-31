/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.

*/

int neighbourhood(
	int winner_index,
	int map_side_size,
    int current_id,
    int logic)
{
	int a_x, a_y, b_x, b_y;
	a_x = current_id % map_side_size;
	a_y = current_id / map_side_size;
	b_x = winner_index % map_side_size;
    b_y = winner_index / map_side_size;
    if(logic == 0){
        return (abs(a_x-b_x) + abs(a_y-b_y));
    }
    else{
        return max(abs(a_x-b_x) , abs(a_y-b_y));
    }
}


__kernel void trainingNeurons( 
                        int TOTAL_NUM_NEURONS, // [3,4,5] = 50
                        int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
                        int NUM_DATAPOINTS, // datapoints to collect in each streaming operation
                        int NUM_PARTITIONS, //Number of feature spaces
                        int NUM_FEATURES, //Number of features in given dataset
                        int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
                        int no_Passes_Phase1, //Number of passes through the data
                        __global int *NUM_FEATURES_PER_PARTITION, //List of features in each feature spaces
                        __global int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
                        __global float *culumative_differnce_per_neuron, //Cumulative Difference per Neuron
                        __global int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
                        __global float *map, // the centers of all neurons across all nets
                        __global int *min_dist_pos, //Minimum distance array position
                        __global float *min_dist_array,//Minimum distance array
                        __global float *Input_Data, // input data
                        __global float *distance_map, // the distance of each neuron from the data points
                        __global float *neigh_rate,
                        __global float *learning_rate_list,
                        __global int *net_sizes, // [3,4,5] 
                        __global int *InputIndex_Per_Partition
                    )
{
    //printf("%d\n",total_num_features_all_featurespaces);
    // Finding the total number of threads in the program
    //const int block_skip = get_local_size(2)*get_local_size(1)*get_local_size(0);
    
    //printf("local_size %d\n",get_global_size(2));
    //printf("Input_Data %f\n",Input_Data);
    // Finding thread index using block and thread index
    //const int thread_id = get_group_id(0)*block_skip + get_group_id(1)*block_skip*get_num_groups(0) + get_group_id(2)*block_skip*get_num_groups(0)*get_num_groups(1) + get_local_id(0);
      
    const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0) ;
    
   //printf("\n num_group %d ",get_global_size(0));
    //Initial declaration
    
    int feature_space_blockID =get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
   
   
    int no_of_features_featurespace = NUM_FEATURES_PER_PARTITION[feature_space_blockID]; //Get the Number of features that in each feature space

    int learning_rate_Index =get_global_id(1) ; //Learning Rate Index for every thread
    
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces;
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID];
    map_start_index += (thread_id % TOTAL_NUM_NEURONS) * no_of_features_featurespace ;
    //printf("\nmap %d",map);
     
   







    //printf("\n thread %d feature_space_blockID %d learning_rate_Index %d threadIdx %d",thread_id ,feature_space_blockID,learning_rate_Index,get_global_id(0)%TOTAL_NUM_NEURONS); 
    //_____________________________________________________________________________________________________________________________________________________
    //     STEP 1: Calculate the net of each thread and net index
    //----------------------------------------------------------------------------------------------------------------------------------------------------
    //  Find which net the thread belongs to
    int which_net_index = 0;
    int which_net = 0;
    int net_end_index = learning_rate_Index * NUM_PARTITIONS *  TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    //printf("net_index %d\n",net_end_index);
    int net_start_index = learning_rate_Index * NUM_PARTITIONS * TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    int thread_position_in_map = 0; 

    //Finding Thread details - Which net it belongs to?
    for(int net = 0; net < NUM_NETS; net++)
    {
        net_end_index = net_end_index + neurons_per_nets[net];
        if(thread_id >= net_start_index && thread_id < net_end_index)
        {
            which_net_index = net;
            which_net = net_sizes[net]; 
            thread_position_in_map = thread_id - net_start_index;
      //printf("\nnet_start_index%d",net_start_index);
        }
        net_start_index = net_end_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//printf("\nthread_position_in_map %d",thread_position_in_map);

    
    for(int pass = 0; pass < no_Passes_Phase1; pass++){
    
        // Loop for each data point
        
        for(int datapoint = 0; datapoint < NUM_DATAPOINTS; datapoint++)
        {
            
            //______________________________________________________________________________________________________________________________________________________
            // Step 1: Find Cumulative distance between neuron center and input data for each feature
            //------------------------------------------------------------------------------------------------------------------------------------------------------
            
            float sum=0;
            int Features_Skip_Partition = InputIndex_Per_Partition[feature_space_blockID]; 
            
            
            for (int feature = Features_Skip_Partition; feature < Features_Skip_Partition + no_of_features_featurespace; feature++){
                 //if(datapoint == 0 && pass == 0 && feature_space_blockID == 0 && learning_rate_Index == 0 && feature % no_of_features_featurespace== 0 ){
                     //printf("\nthread_id: %d InputIndex: %d InputData:%f mapData :%f feature: %d mapIndex: %d ", thread_id, datapoint * NUM_FEATURES + feature, Input_Data[datapoint * NUM_FEATURES + feature], map[map_start_index + feature % no_of_features_featurespace], feature_space_blockID, map_start_index + feature % no_of_features_featurespace);
                 //}
                sum += fabs(Input_Data[datapoint * NUM_FEATURES + feature] - map[map_start_index + feature % no_of_features_featurespace]);
                //printf("\n thread_id %d Input_dat %f sum %f index %d ",thread_id, Input_Data[datapoint * NUM_FEATURES + feature],sum,datapoint * NUM_FEATURES + feature);
            }
            
            distance_map[thread_id] = sum;
            //printf("\n threeadid %d distancemap%f",thread_id,sum);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            //______________________________________________________________________________________________________________________________________________________
            //END OF STEP 1
            //----------------------------------------------------------------------------------------------------------------------------------------------------------




            //_____________________________________________________________________________________________________________________________________________________
            //STEP 2: Calculating minium distance Neuron and it's index in that Net
            //----------------------------------------------------------------------------------------------------------------------------------------------------
                        
            if(get_global_id(0)%TOTAL_NUM_NEURONS == 0)
            {   
                //printf("\n treadId %d",get_global_id(0)%TOTAL_NUM_NEURONS);
                //Distance Map index adjusting
                int distance_map_Index = thread_id;
                int min_pos_startIndex = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;

                //Loop through each net
                int net = 0;
                for(; net < NUM_NETS; net++)
                {
                    min_dist_array[min_pos_startIndex + net] = INT_MAX;

                    //Loop through every neuron in that net
                    for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                    {
                        // if(feature_space_blockID == 0 && datapoint == 0 && pass == no_Passes_Phase1 -1 && learning_rate_Index == 0){
                        //     printf("\ndistance : %f, net :%d, neuron :%d, min_dist :%f", distance_map[distance_map_Index + neuron], net, neuron, min_dist_array[min_pos_startIndex + net]);
                        // }
                        if(distance_map[distance_map_Index + neuron] < min_dist_array[min_pos_startIndex + net])
                        {
                            //Capture min dist array neuron and it's Index
                            min_dist_array[min_pos_startIndex + net] = distance_map[distance_map_Index + neuron];
                            min_dist_pos[min_pos_startIndex + net] = neuron;  // sequental traversal of distance map array to find the winner neuron.
                        }
                    }
                    //Adjust distance map after every net
                    //printf("\n Winner: %d winner dis: %f thread_id: %d ", min_dist_pos[min_pos_startIndex + net],min_dist_array[min_pos_startIndex + net], net);
                    distance_map_Index += neurons_per_nets[net];
                }
            }
            

            // if(threadIdx.x == 0 && datapoint == 2){
            //     int min_pos_startIndex = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;
            //     for(int i=0;i< NUM_NETS; i++ ){
            //         printf("\n%f %d",  min_dist_array[min_pos_startIndex + i] , min_dist_pos[min_pos_startIndex + i]);
            //     }
            // }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            
            //______________________________________________________________________________________________________________________________________________________
            //END OF STEP 2
            //----------------------------------------------------------------------------------------------------------------------------------------------------------



            //_____________________________________________________________________________________________________________________________________________________
            //STEP 4: Seeing the winner, calculating the neighbour value and update the Map
            //----------------------------------------------------------------------------------------------------------------------------------------------------

            //printf("%2d %2d %2d %2d %2d %2d\n", which_net_index, thread_id, learning_rate_Index, datapoint, which_net, thread_position_in_map);

            //Min Dist Pos Array index for each thread to see its winner in that net
            int min_dist_post_net_index = learning_rate_Index * NUM_PARTITIONS * NUM_NETS + feature_space_blockID * NUM_NETS + which_net_index;

            //Capture Winner Index in that net
            int myWinner = min_dist_pos[min_dist_post_net_index];
            // if(feature_space_blockID == 0 && datapoint == 0 && pass == no_Passes_Phase1 -1 && learning_rate_Index == 0){
            //     printf("\nnet : %d, neuron: %d winner :%d, winnerDistnace: %f", which_net_index, thread_id, myWinner, min_dist_array[min_dist_post_net_index]);
            // }
            
            //Calculating the Neighbourhood distance for each neuron from its winner
            int neighbourhood_value = neighbourhood(myWinner, which_net, thread_position_in_map, 0);
            // if(feature_space_blockID == 0 && datapoint == 0 && pass == no_Passes_Phase1 -1 && learning_rate_Index == 0){
            //     printf("\nnet : %d, neuron: %d, neighRate: %d", which_net_index, thread_id, neighbourhood_value);}
            
            int Neurons_current_net = which_net * which_net;
            int update = 0;

             //printf("\nThread: %d Update: %d neighbourhood_value:%d value: %f", thread_id, update, neighbourhood_value, neigh_rate[which_net+myWinner] * (which_net));
            // printf("\n%f", neigh_rate[which_net+myWinner] );
            int neigh_rate_index = learning_rate_Index * NUM_PARTITIONS * NUM_NETS + feature_space_blockID * NUM_NETS + which_net_index;
            //printf("\n neigh rate index : %d thread_id : %d index_value %f",neigh_rate_index, thread_id,(neigh_rate[neigh_rate_index] * (which_net)) );
            if(neighbourhood_value <= (neigh_rate[neigh_rate_index] * (which_net)))
                update = 1;
            else
                update = 0;
            
            // if(feature_space_blockID == 0 && datapoint == 0 && pass == no_Passes_Phase1 -1 && learning_rate_Index == 0){
               // printf("\nnet : %d, neuron: %d winner :%d, Adjusting : %d neighRate: %d", which_net_index, thread_id, myWinner, update, neighbourhood_value);
            // }

            // printf("\nthread : %d neighbourhoodvalue : %dupdate : %d", thread_id, neighbourhood_value, update);
            

            float cumulative_weight = 0;
            float neighbour_adjust = (1 - ((float)(neighbourhood_value)/(float)(Neurons_current_net)));


            //Find the corresponding Map Array Index
            for (int feature = Features_Skip_Partition; feature < Features_Skip_Partition + no_of_features_featurespace; feature++){

                float temp =  map[map_start_index + feature % no_of_features_featurespace];
                float difference_map_input = Input_Data[datapoint * NUM_FEATURES + feature] - map[map_start_index + feature % no_of_features_featurespace];
                
                map[map_start_index + feature % no_of_features_featurespace] = map[map_start_index + feature % no_of_features_featurespace] + (neighbour_adjust * difference_map_input * update * learning_rate_list[learning_rate_Index]);
                 //if(feature_space_blockID == 1 && datapoint == 0 && pass == no_Passes_Phase1 -1 && learning_rate_Index == 0 && feature % no_of_features_featurespace == 0){
                  // printf("\nnet : %d, neuron: %d winner :%d, BeforeMapData : %f AfterMapData: %f neighbour_adjust :%f update :%d diff_map: %f", which_net_index, thread_id, myWinner, temp, map[map_start_index + feature % no_of_features_featurespace], neighbour_adjust,update,difference_map_input  );
                 //}
                cumulative_weight = cumulative_weight + fabs(map[map_start_index + feature % no_of_features_featurespace] - temp);
            }

            culumative_differnce_per_neuron[thread_id] = cumulative_weight;
            //printf("\nculumative_differnce_per_neuron[thread_id]%d",culumative_differnce_per_neuron[thread_id]);
           // printf("%d\n",cumulative_weight);
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    }

}




