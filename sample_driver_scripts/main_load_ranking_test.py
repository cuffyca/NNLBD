#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    10/10/2020                                                                   #
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
import sys
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD


def Main():
    # Create Model With Default Settings Except
    model = LBD( write_log_to_file = True )

    # Load Previously Trained Model
    model.Load_Model( "../test_model" )

    # Continue Refining Model Over Data: "data/test/cui_mini"
    #model.Fit( "../data/test/cui_mini", epochs = 30, batch_size = 4, verbose = 1 )

    # Rank Evaluation Data-set Input Instances, Return Top 5 Predictions Per Input Instance
    output_rankings_per_input_instance = model.Evaluate_Ranking( "../data/test/cui_mini", number_of_predictions = 20 )

    print( "Ranked Outputs: " )

    for input_instance in output_rankings_per_input_instance:
        print( input_instance + " => " + str( output_rankings_per_input_instance[ input_instance ] ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()