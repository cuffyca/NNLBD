#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/13/2021                                                                   #
#    Revised: 03/18/2021                                                                   #
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
    model = LBD()

    # Load Model
    model.Load_Model( "../trained_models/cs5_crichton_hinton_softplus_model" )

    # Evaluate Model
    # Crichton Experimental A-B-C Links Per Data-set
    # CS1: 'PR:000001754\tPR:000002307\tMESH:D000236\t0.0'
    # CS2: 'PR:000011331\tHOC:42\tPR:000005308\t0.0'
    # CS3: 'PR:000001138\tPR:000003107\tPR:000006736\t0.0'
    # CS4: 'PR:000011170\tCHEBI:26523\tMESH:D010190\t0.0'
    # CS5: 'PR:000006066\tHOC:42\tMESH:D013964\t0.0'
    gold_b_instance = 'PR:000001754\tPR:000002307\tMESH:D000236\t0.0'

    a_term          = gold_b_instance.split( '\t' )[0]
    b_term          = gold_b_instance.split( '\t' )[1]
    c_term          = gold_b_instance.split( '\t' )[2]

    # Model Inference Over A & C Terms (Closed Discovery)
    predictions       = model.Predict( a_term, c_term, return_raw_values = True )[0]

    # Fetch All Unique Terms From DataLoader Dictionary
    unique_token_list = model.Get_Data_Loader().Get_Token_ID_Dictionary().keys()

    print( "\nRank Unique Token Predictions Using Probabilities:\n" )

    prob_dict = {}

    # For Each Prediction From The Model, Store The Prediction Value And Unique Concept Token Within A Dictionary
    for token, prediction in zip( unique_token_list, predictions ):
        # print( str( token ) + "\t:\t" + str( prediction ) )
        prob_dict[token] = prediction

    # Sort Concept And Probability Dictionary In Reverse Order To Rank Concepts.
    prob_dict  = { k: v for k, v in sorted( prob_dict.items(), key = lambda x: x[1], reverse = True ) }

    # Get Index Of Desired Gold B
    gold_b_index = list( prob_dict.keys() ).index( b_term.lower() )
    gold_b_value = prob_dict[b_term.lower()]

    # Get Number Of Ties With Gold B Prediction Value
    gold_b_ties  = list( prob_dict.values() ).count( gold_b_value ) - 1

    print( "Gold B: " + str( b_term ) + " - Rank: " + str( gold_b_index ) + \
           " Of " + str( len( prob_dict ) ) + " - Rank Value: " + str( gold_b_value ) + \
           " - Number Of Ties: " + str( gold_b_ties ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()