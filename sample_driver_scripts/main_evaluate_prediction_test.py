#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 12/16/2020                                                                   #
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
    model = LBD( write_log_to_file = True, network_model = "hinton", per_epoch_saving = False, skip_out_of_vocabulary_words = True )

    # Train Model Over Data: "data/test/cui_mini"
    model.Fit( "../data/test/cui_mini", epochs = 10, verbose = 1 )
    model.Save_Model( "../test_model" )

    #Evaluate Model Of Entire Data-set
    accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )

    print( "Evaluation Score - Accuracy : {:.4f}" . format( accuracy ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()