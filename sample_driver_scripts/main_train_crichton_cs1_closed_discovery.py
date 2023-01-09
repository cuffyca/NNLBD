#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 01/28/2021                                                                   #
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
    model = LBD( network_model = "hinton", model_type = "closed_discovery", loss_function = "binary_crossentropy",
                 print_debug_log = False, write_log_to_file = False, per_epoch_saving = False, activation_function = 'sigmoid',
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False, early_stopping_metric_monitor = "F1_Score",
                 early_stopping_persistence = 3, dropout = 0.5, use_batch_normalization = False, trainable_weights = False,
                 embedding_path = "../vectors/HOC/test_modified_cs1.embeddings" )

    # Train Model Over Data: "data/test/cui_mini"
    model.Fit( "../data/train_cs1_closed_discovery_without_aggregators_mod", epochs = 500,
               batch_size = 128, learning_rate = 0.001, verbose = 1 )
    model.Save_Model( "../hinton_crichton_model" )
    model.Generate_Model_Metric_Plots( "../hinton_crichton_model" )

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