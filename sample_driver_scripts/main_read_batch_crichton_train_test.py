#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    12/15/2020                                                                   #
#    Revised: 12/16/2020                                                                   #
#                                                                                          #
#    Tests DataLoader::Get_Next_Elements() function to load data in-line versus            #
#        all at once. Used for large files which are unable to fit in memory.              #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python data_loader_read_batch_test.py"                               #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Python Libraries
import sys
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD

def Main():
    print( "~Begin" )
    file_path           = "../data/train_cs1_closed_discovery_without_aggregators.tsv"
    embedding_path      = "../vectors/test_modified_cs1.embeddings"
    actual_epochs       = 150
    instances_per_batch = 150

    # Create Model With Default Settings Except (network_model = 'rumelhart', per_epoch_saving = True, and skip_out_of_vocabulary_words = True)
    model = LBD( network_model = "simple", model_type = "closed_discovery", activation_function = "softplus",
                 print_debug_log = False, write_log_to_file = False, per_epoch_saving = False,
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False,
                 early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, dropout = 0.0,
                 use_batch_normalization = False, trainable_weights = False, embedding_path = embedding_path )

    ###############################################
    # Generate Token IDs For Data File In Batches #
    ###############################################
    data_loader = model.Get_Data_Loader()
    data_loader.Load_Embeddings( embedding_path )
    data_loader.Generate_Token_IDs()

    ###################
    # Train The Model #
    ###################
    for epoch in range( actual_epochs ):
        # Fetch Elements Until The EOF Is Reached
        while data_loader.Reached_End_Of_File() == False:
            data_elements = data_loader.Get_Next_Batch( file_path, number_of_elements_to_fetch = instances_per_batch )

            # Only Process 'data_elements' If It Contains Data

            # Error Reading Data File
            if data_elements == [-1]:
                print( "Error Reading File \"" + str( file_path ) + "\"" )
                break
            # Process Data Element(s) From File
            elif data_elements:
                model.Fit( data_instances = data_elements, epochs = 1, verbose = 1 )

        # Reset File Position Index
        if data_loader.Reached_End_Of_File():
            print( "EOF Reached / Resetting Position Index" )
            data_loader.Reset_File_Position_Index()

    ######################
    # Evaluate The Model #
    ######################
    # accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )

    # print( "Evaluation Scores - Accuracy : {:.4f}" . format( accuracy  ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()