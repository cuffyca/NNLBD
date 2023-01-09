#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    10/06/2020                                                                   #
#    Revised: 12/16/2020                                                                   #
#                                                                                          #
#    Generates A Neural Network Used For LBD, Trains Using Data In Format Below.           #
#        Model Load & Training Resume Driver Script                                        #
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
import os, sys
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD


def Main():
    # Create Model With Default Settings Except
    model = LBD( print_debug_log = True, write_log_to_file = True, model_type = "open_discovery" )

    # Load Previously Trained Model
    model.Load_Model( "../test_model/model" )

    # Continue Refining Model Over Data: "data/test/cui_mini_open_discovery"
    #model.Fit( "data/test/cui_mini_open_discovery", epochs = 30, batch_size = 4, verbose = 1 )

    # Evaluate Model Of Entire Data-set
    accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )

    print( "Evaluation Scores: - Accuracy : {:.4f}" . format( accuracy  ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()