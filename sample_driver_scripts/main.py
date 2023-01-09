#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 01/13/2021                                                                   #
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
    model = LBD( print_debug_log = False, write_log_to_file = False, network_model = "hinton", per_epoch_saving = False, use_gpu = True,
                 skip_out_of_vocabulary_words = True, activation_function = 'sigmoid', loss_function = "binary_crossentropy", use_csr_format = True,
                 model_type = "open_discovery", trainable_weights = False, embedding_path = "../vectors/test/vectors_random_cui_mini" )

    # Train Model Over Data: "data/test/cui_mini_open_discovery"
    model.Fit( "../data/test/cui_mini_open_discovery", epochs = 100, verbose = 1, learning_rate = 0.005, batch_size = 32 )
    # model.Save_Model( "../new_test_model" )
    # model.Load_Model( "../new_test_model" )

    # Generate Model Metric Plots
    # model.Generate_Model_Metric_Plots( "./test_model" )

    # Evaluate Model Of Entire Data-set
    accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )

    #prediction = model.Predict( "c001", "treats", return_raw_values = True )
    #print( str( prediction ) )

    print( ""                    )
    print( "Evaluation Scores: " )
    # print( "    F1 Score : {:.4f}" . format( f1_score  ) )
    # print( "    Precision: {:.4f}" . format( precision ) )
    # print( "    Recall   : {:.4f}" . format( recall    ) )
    print( "    Accuracy : {:.4f}" . format( accuracy  ) )
    # print( "    Loss     : {:.4f}" . format( loss      ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()