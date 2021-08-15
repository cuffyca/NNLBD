#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/02/2021                                                                   #
#    Revised: 01/02/2021                                                                   #
#                                                                                          #
#    Generates A Neural Network Used For LBD, Trains Using Data In Format Below.           #
#        Driver Script                                                                     #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python main.py"                                                      #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Modules
import sys
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD


def Main():
    # Create Model With Default Settings Except Those Listed Below
    model = LBD( network_model = "cd2", model_type = "closed_discovery", activation_function = "relu",
                 print_debug_log = False, write_log_to_file = False, per_epoch_saving = False,
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False, loss_function = "sparse_categorical_crossentropy",
                 early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, dropout = 0.0, final_layer_type = "mlp",
                 use_batch_normalization = False, trainable_weights = False, embedding_path = "../vectors/crichton_orig/test_modified_cs1.reduced_embeddings" )

    # model.Get_Data_Loader().Read_Data( "../data/train_cs1_closed_discovery_without_aggregators_original_test" )
    # model.Get_Data_Loader().Load_Embeddings( "../vectors/test_modified_cs1.reduced_embeddings" )
    # model.Get_Data_Loader().Generate_Token_IDs( restrict_output = False )
    # input_1, input_2, input_3, outputs = model.Get_Data_Loader().Vectorize_Model_Inputs( "MESH:C116288", "MESH:D005354", "CHEBI:29108", "0.00036253776435",
    #                                                                         is_crichton_format = True, pad_inputs = False, model_type = model.Get_Model_Type() )
    #
    # print( "Primary  : " + str( input_1 ) )
    # print( "Secondary: " + str( input_2 ) )
    # print( "Tertiary : " + str( input_3 ) )
    # print( "Outputs  : " + str( outputs ) )

    # Train Model Over Data: "data/cui_mini"
    model.Fit( "../data/crichton_orig/train_cs1_closed_discovery_without_aggregators_test", epochs = 150, batch_size = 128, learning_rate = 0.001, verbose = 1,
               margin = 0.45, scale = 0.14 )
    # model.Save_Model( "../test_model" )
    # model.Generate_Model_Metric_Plots( "../test_model" )

    #Evaluate Model Of Entire Data-set
    # accuracy = model.Evaluate_Prediction( "data/cui_mini_eval" )
    #
    # print( ""                    )
    # print( "Evaluation Scores: " )
    # # print( "    F1 Score : {:.4f}" . format( f1_score  ) )
    # # print( "    Precision: {:.4f}" . format( precision ) )
    # # print( "    Recall   : {:.4f}" . format( recall    ) )
    # print( "    Accuracy : {:.4f}" . format( accuracy  ) )
    # # print( "    Loss     : {:.4f}" . format( loss      ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()