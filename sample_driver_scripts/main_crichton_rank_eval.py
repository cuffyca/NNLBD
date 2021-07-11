#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/04/2021                                                                   #
#    Revised: 01/04/2021                                                                   #
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
    model = LBD( network_model = "cd2" )

    model.Load_Model( "../test_model/model" )

    eval_data = model.Get_Data_Loader().Read_Data( "../data/test_cs1_closed_discovery_without_aggregators" )

    b_prediction_dictionary = {}
    number_of_ties          = 0
    rank                    = 1
    gold_b_term             = "PR:000002307"
    gold_b_prediction_score = model.Predict( "PR:000001754", "PR:000002307", "MESH:D000236", return_vector = True, return_raw_values = True )

    print( "MESH:C026759 PR:Q84TG3 PR:Q9ZPE4 - Prediction: " + str( model.Predict( "MESH:C026759", "PR:Q84TG3", "PR:Q9ZPE4", return_vector = True, return_raw_values = True ) ) )
    print( "SNP:rs7158754 SNP:rs802047 SNP:rs802049 - Prediction: " + str( model.Predict( "SNP:rs7158754", "SNP:rs802047", "SNP:rs802049", return_vector = True, return_raw_values = True ) ) )

    print( "Performing Inference For Testing Instance Predictions" )

    prediction_dictionary = {}

    for instance in eval_data:
        instance_tokens = instance.split()
        a_term = instance_tokens[0]
        b_term = instance_tokens[1]
        c_term = instance_tokens[2]

        prediction = model.Predict( a_term, b_term, c_term, return_vector = True, return_raw_values = True )

        if isinstance( prediction, list ) == False:
            prediction_dictionary[a_term + "\t" + b_term + "\t" + c_term] = float( prediction )

        if instance_tokens[1] not in b_prediction_dictionary:
            b_prediction_dictionary[b_term] = [prediction]
        else:
            b_prediction_dictionary[b_term].append( prediction )

    for element in prediction_dictionary:
        print( str( element ) + "\t" + str( prediction_dictionary[element] ) )

    print( "Ranking Gold B Term Among All B Terms" )

    # Perform Ranking
    for b_term in b_prediction_dictionary:
        if b_term == gold_b_term: continue
        if b_prediction_dictionary[b_term] > gold_b_prediction_score:
            rank += 1
        elif b_prediction_dictionary[b_term] == gold_b_prediction_score:
            number_of_ties += 1

    print( "Gold B: " + str( gold_b_term ) + " - Rank: " + str( rank ) + " Of " + str( len( eval_data ) ) + " Number Of B Terms" + " - Score: " + str( gold_b_prediction_score ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()