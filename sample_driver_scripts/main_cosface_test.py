#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/15/2021                                                                   #
#    Revised: 01/15/2021                                                                   #
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
    # Create Model With Default Settings Except (network_model = 'rumelhart', per_epoch_saving = True, and skip_out_of_vocabulary_words = True)
    model = LBD( print_debug_log = True, write_log_to_file = False, network_model = "cosface", final_layer_type = "arcface",
                 per_epoch_saving = False, use_gpu = True, skip_out_of_vocabulary_words = True, activation_function = 'sigmoid',
                 loss_function = "binary_crossentropy", model_type = "open_discovery", trainable_weights = False,
                 embedding_path = "../vectors/crichton_orig/test_modified_cs1.reduced_embeddings", device_name = "/gpu:0",
                 embedding_modification = "hadamard", optimizer = 'adam', use_csr_format = True, verbose = 2 )

    # Train Model Over Data: "data/cui_mini"
    model.Fit( "../data/crichton_orig/train_cs1_closed_discovery_without_aggregators_mod_test", epochs = 750, verbose = 2,
               learning_rate = 0.005, dropout = 0, margin = 5, scale = 0.35, batch_size = 10,
               shuffle = True )



    # # Evaluate Model Of Entire Data-set
    # accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )
    #
    # #prediction = model.Predict( "c001", "treats", return_raw_values = True )
    # #print( str( prediction ) )
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