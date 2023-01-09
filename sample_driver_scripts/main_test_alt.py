#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 12/14/2020                                                                   #
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
    model = LBD( network_model = "hinton", model_type = "open_discovery",
                 print_debug_log = False, write_log_to_file = True, per_epoch_saving = False,
                 use_csr_format = True, use_gpu = True, enable_early_stopping = True, verbose = 1,
                 early_stopping_metric_monitor = "F1_Score", early_stopping_persistence = 3, dropout = 0.5,
                 use_batch_normalization = False, trainable_weights = False, embedding_path = "../vectors/word2vec_cuis_1975_2009_noMin",
                 prediction_threshold = 0.5 )

    # Train Model Over Data: "data/test/cui_mini"
    model.Fit( "../data/known_smallTest", epochs = 10, batch_size = 32, verbose = 1 )
    model.Save_Model( "../test_model" )
    model.Generate_Model_Metric_Plots( "../test_model" )

    # Evaluate Model Of Entire Data-set
    accuracy = model.Evaluate_Prediction( "../data/true_smallTest" )

    print( "Evaluation Scores: - Accuracy : {:.4f}" . format( accuracy  ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()