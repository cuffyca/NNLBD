#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 12/16/2020                                                                   #
#                                                                                          #
#    Generates A Neural Network Used For LBD.                                              #
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
    # Create Model With Default Settings Except (DebugLog = True and Epochs = 20)
    model = LBD( network_model = "rumelhart", model_type = "open_discovery",
                 print_debug_log = True, per_epoch_saving = False, use_csr_format = True )

    # Train Model Over Data: "data/cui_mini"
    model.Fit( training_file_path = "../data/cui_mini", learning_rate = 0.005, epochs = 10, batch_size = 10, verbose = 1 )

    # Evaluate Model Of Entire Data-set
    loss, accuracy, precision, recall, f1_score = model.Evaluate( "../data/cui_mini_eval" )

    print( "Evaluation Scores: " )
    print( "    F1 Score : {:.4f}" . format( f1_score  ) )
    print( "    Precision: {:.4f}" . format( precision ) )
    print( "    Recall   : {:.4f}" . format( recall    ) )
    print( "    Accuracy : {:.4f}" . format( accuracy  ) )
    print( "    Loss     : {:.4f}" . format( loss      ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()