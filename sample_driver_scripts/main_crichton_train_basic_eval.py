#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/09/2021                                                                   #
#    Revised: 01/09/2021                                                                   #
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
    model = LBD( network_model = "simple", model_type = "closed_discovery", activation_function = "softplus",
                 print_debug_log = False, write_log_to_file = False, per_epoch_saving = False,
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False,
                 early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, dropout = 0.0,
                 use_batch_normalization = False, trainable_weights = False, embedding_path = "../vectors/test_modified_cs1.embeddings" )

    # Training/Evaluation Variables
    model_iterations        = 800
    ranking_per_epoch       = []
    ranking_per_epoch_value = []
    best_number_of_ties     = 0
    best_ranking            = sys.maxsize
    eval_data               = model.Get_Data_Loader().Read_Data( "../data/test_cs1_closed_discovery_without_aggregators.tsv", keep_in_memory = False )

    for iteration in range( model_iterations ):
        # Train Model Over Data: "../data/train_cs1_closed_discovery_without_aggregators_original"
        model.Fit( "../data/train_cs1_closed_discovery_without_aggregators.tsv", epochs = 1, batch_size = 128, learning_rate = 0.001, verbose = 1 )

        # Ranking/Evaluation Variables
        b_prediction_dictionary = {}
        rank                    = 1
        number_of_ties          = 0
        gold_b_term             = "PR:000002307"
        gold_b_prediction_score = model.Predict( "PR:000001754", "PR:000002307", "MESH:D000236", return_vector = True, return_raw_values = True )

        print( "Performing Inference For Testing Instance Predictions" )

        # Perform Model Evaluation (Ranking Of Gold B Term)
        for instance in eval_data:
            instance_tokens = instance.split()
            a_term = instance_tokens[0]
            b_term = instance_tokens[1]
            c_term = instance_tokens[2]

            # Perform Model Inference Using Evaluation Instance From Evaluation Data
            prediction = model.Predict( a_term, b_term, c_term, return_vector = True, return_raw_values = True )

            if instance_tokens[1] not in b_prediction_dictionary:
                b_prediction_dictionary[b_term] = [prediction]
            else:
                b_prediction_dictionary[b_term].append( prediction )

        # Ranking Gold B Term Among All B Terms
        for b_term in b_prediction_dictionary:
            if b_term == gold_b_term: continue
            if b_prediction_dictionary[b_term] > gold_b_prediction_score:
                rank += 1
            elif b_prediction_dictionary[b_term] == gold_b_prediction_score:
                number_of_ties += 1

        ranking_per_epoch.append( rank )
        ranking_per_epoch_value.append( gold_b_prediction_score )

        if rank < best_ranking:
            best_ranking        = rank
            best_number_of_ties = number_of_ties
            model.Save_Model( "../test_model" )

        print( "Epoch : " + str( iteration ) + " - Gold B: " + str( gold_b_term ) + " - Rank: " + str( rank ) +
               " Of " + str( len( eval_data ) ) + " Number Of B Terms" + " - Score: " + str( gold_b_prediction_score ) )

    # Print Ranking Information Per Epoch
    for epoch in range( len( ranking_per_epoch ) ):
        print( "Epoch: " + str( epoch ) + " - Rank: " + str( ranking_per_epoch[epoch] ) +
               " - Value: " + str( ranking_per_epoch_value[epoch] ) + " - Number Of Ties: " + str( number_of_ties) )

    print( "\nBest Rank: " + str( best_ranking ) )
    print( "Number Of Ties With Best Rank: " + str( best_number_of_ties ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()