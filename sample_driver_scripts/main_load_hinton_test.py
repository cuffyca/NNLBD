#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    04/10/2021                                                                   #
#    Revised: 04/10/2021                                                                   #
#                                                                                          #
#    Loads Hinton Model And Evaluates (Ranks) Given A-B-C Relationship.                    #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python main_load_hinton_test.py"                                     #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Modules
import sys

from numpy.lib.arraysetops import unique
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD


def Main():
    # Create Model With Default Settings Except
    model = LBD( write_log_to_file = False )

    # Load Previously Trained Model
    model.Load_Model( "../cs1_crichton_hinton_sigmoid_model_fs" )
    data_loader = model.Get_Data_Loader()

    # Performs Inference Using A & B Concepts, Returns A Vector Of Raw Predictions Over All Unique C Concepts
    # Closed Discovery A-B-C Link:  "mesh:c044115\tmesh:c021096\tchebi:17719"
    primary_term_id = data_loader.Get_Token_ID( "mesh:c044115", token_type = "primary" )
    secondary_term_id = data_loader.Get_Token_ID( "chebi:17719", token_type = "secondary" )

    if primary_term_id != -1 and secondary_term_id != -1:
        model_predictions = model.Predict( "mesh:c044115", "chebi:17719", return_vector = False, return_raw_values = True )

        # Get The Prediction Value Of The Ground Truth Concept, Given A & B Input Concepts
        desired_id = data_loader.Get_Token_ID( "mesh:c021096", token_type = "secondary" )
        if desired_id != -1: print( "B Concept 'mesh:c021096' Prediction Value: " + str( model_predictions[0][desired_id] ) )

        # Get All Predicted B Concepts Given A & C Concepts
        model_predictions = model.Predict( "mesh:c044115", "chebi:17719", return_vector = False, return_raw_values = False )
        print( "Predicted C-Concepts: " + str( model_predictions ) )


    # Gold B Instance: "pr:000001754\tpr:000002307\tmesh:d000236"
    primary_term_id = data_loader.Get_Token_ID( "pr:000001754", token_type = "primary" )
    secondary_term_id = data_loader.Get_Token_ID( "mesh:d000236", token_type = "secondary" )

    if primary_term_id != -1 and secondary_term_id != -1:
        gold_b_id = data_loader.Get_Token_ID( "pr:000002307", token_type = "secondary" )

        if gold_b_id != -1:
            model_predictions = model.Predict( "pr:000001754", "mesh:d000236", return_vector = False, return_raw_values = True )[0]
            print( "Gold B Concept 'pr:000002307' Prediction Value: " + str( model_predictions[gold_b_id] ) )

        # Perform B Concept Ranking
        unique_b_concepts = None
        prob_dict         = {}

        # For Closed Discovery, Secondary Dictionary Becomes The Output Dictionary (B Is Substituted With C In A-B-C Model)
        unique_b_concepts = list( data_loader.Get_Secondary_ID_Dictionary().keys() )

        if len( unique_b_concepts ) != len( model_predictions ):
            print( "Error: Unique Concept List != Number Of Model Predictions" )
            exit()

        for token, prediction in zip( unique_b_concepts, model_predictions ):
            prob_dict[token] = prediction

        print( str( prob_dict["pr:000002307"] ) )

        # Sort Concept And Probability Dictionary In Reverse Order To Rank Concepts.
        prob_dict  = { k: v for k, v in sorted( prob_dict.items(), key = lambda x: x[1], reverse = True ) }

        # Get Index Of Desired Gold B
        gold_b_rank  = list( prob_dict.keys() ).index( "pr:000002307" )
        gold_b_value = prob_dict["pr:000002307"]

        # Get Number Of Ties With Gold B Prediction Value
        gold_b_ties  = list( prob_dict.values() ).count( gold_b_value ) - 1

        print( "Gold B Rank: " + str( gold_b_rank ) + " - Value: " + str( gold_b_value ) + " - Number Of Ties: " + str( gold_b_ties ) )


    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()