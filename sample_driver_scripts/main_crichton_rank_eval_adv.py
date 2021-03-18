#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/11/2021                                                                   #
#    Revised: 01/12/2021                                                                   #
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
    model = LBD( print_debug_log = False, network_model = "simple" )
    
    # Load Model And Read Data From File
    model.Load_Model( "../test_model/model" )
    eval_data = model.Read_Data( "../data/test_cs1_closed_discovery_without_aggregators" )
    
    # Placeholder Variables (Do Not Modify)
    b_prediction_dictionary = {}
    prediction_dictionary   = {}
    number_of_ties          = 0
    rank                    = 1
    gold_b_term             = "PR:000002307"
    gold_b_instance         = 'PR:000001754\tPR:000002307\tMESH:D000236\t0.0'
    
    # Vectorize Gold B Term And Entire Evaluation Data-set
    gold_b_input_1, gold_b_input_2, gold_b_input_3, _ = model.Vectorize_Model_Data( data_list = [gold_b_instance], model_type = "closed_discovery",
                                                                                    keep_in_memory = False, use_csr_format = True, is_crichton_format = True )
    eval_input_1, eval_input_2, eval_input_3, _       = model.Vectorize_Model_Data( model_type = "closed_discovery", use_csr_format = True, is_crichton_format = True )
    
    # Checks
    if gold_b_input_1 is None or gold_b_input_2 is None or gold_b_input_3 is None:
        print( "Error Occurred During Data Vectorization (Gold B)" )
        exit()
    if eval_input_1 is None or eval_input_2 is None or eval_input_3 is None:
        print( "Error Occurred During Data Vectorization (Evaluation Data)" )
        exit()
    
    gold_b_prediction_score = model.Predict( primary_input_matrix = gold_b_input_1, secondary_input_matrix = gold_b_input_2,
                                             tertiary_input_matrix = gold_b_input_3, return_vector = True, return_raw_values = True )
    
    # Test Predictions: First One Is Supposed To Predict '0' And The Second Is Supposed To Predict '1'. (Pulled From Training Data.)
    print( "MESH:C026759 PR:Q84TG3 PR:Q9ZPE4 - Prediction: " + str( model.Predict( "MESH:C026759", "PR:Q84TG3", "PR:Q9ZPE4", return_vector = True, return_raw_values = True ) ) )
    print( "SNP:rs7158754 SNP:rs802047 SNP:rs802049 - Prediction: " + str( model.Predict( "SNP:rs7158754", "SNP:rs802047", "SNP:rs802049", return_vector = True, return_raw_values = True ) ) )
    
    print( "Performing Inference For Testing Instance Predictions" )
    
    # Perform Prediction Over The Entire Evaluation Data-set (Model Inference)
    predictions = model.Predict( primary_input_matrix = eval_input_1, secondary_input_matrix = eval_input_2,
                                 tertiary_input_matrix = eval_input_3, return_vector = True, return_raw_values = True )
    
    if isinstance( predictions, list ) and len( predictions ) == 0:
        print( "Error Occurred During Model Inference" )
        exit()
    
    for instance, instance_prediction in zip( eval_data, predictions ):
        instance_tokens = instance.split()
        a_term = instance_tokens[0]
        b_term = instance_tokens[1]
        c_term = instance_tokens[2]
        
        prediction_dictionary[a_term + "\t" + b_term + "\t" + c_term] = instance_prediction
        
        if instance_tokens[1] not in b_prediction_dictionary:
            b_prediction_dictionary[b_term] = [instance_prediction]
        else:
            b_prediction_dictionary[b_term].append( instance_prediction )
    
    for element in prediction_dictionary:
        print( str( element ) + "\t" + str( prediction_dictionary[element] ) )
    
    print( "\nRanking Gold B Term Among All B Terms" )
    
    # Perform Ranking
    for b_term in b_prediction_dictionary:
        if b_term == gold_b_term: continue
        
        if b_prediction_dictionary[b_term] > gold_b_prediction_score:
            rank += 1
        elif b_prediction_dictionary[b_term] == gold_b_prediction_score:
            number_of_ties += 1
    
    print( "Gold B: " + str( gold_b_term ) + " - Rank: " + str( rank ) + " Of " + str( len( eval_data ) ) +
           " Number Of B Terms" + " - Score: " + str( gold_b_prediction_score ) +
           " - Number Of Ties: " + str( number_of_ties ) )
    
    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()