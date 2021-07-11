#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    01/09/2021                                                                   #
#    Revised: 01/28/2021                                                                   #
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
import re, sys
import matplotlib.pyplot as plt
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD import LBD


def Main():
    # User Settings
    training_data_file_path   = "../data/train_cs1_closed_discovery_without_aggregators"
    evaluation_data_file_path = "../data/test_cs1_closed_discovery_without_aggregators"
    embedding_path            = "../vectors/test_modified_cs1.embeddings"
    # embedding_path            = "../data/test_modified_cs5.embeddings"
    model_save_path           = "../cs1_softplus_model"

    # Training/Evaluation Settings
    model_iterations          = 300
    learning_rate             = 0.001
    batch_size                = 128
    activation_function       = "softplus"
    embedding_modification    = "hadamard"
    gold_b_term               = "PR:000002307"
    gold_b_instance           = 'PR:000001754\tPR:000002307\tMESH:D000236\t0.0'

    # Crichton Experimental A-B-C Links Per Data-set
    # CS1: 'PR:000001754\tPR:000002307\tMESH:D000236\t0.0'
    # CS2: 'PR:000011331\tHOC:42\tPR:000005308\t0.0'
    # CS3: 'PR:000001138\tPR:000003107\tPR:000006736\t0.0'
    # CS4: 'PR:000011170\tCHEBI:26523\tMESH:D010190\t0.0'
    # CS5: 'PR:000006066\tHOC:42\tMESH:D013964\t0.0'

    # Create Model With Default Settings Except Those Listed Below
    model = LBD( network_model = "cd2", model_type = "closed_discovery", activation_function = activation_function,
                 print_debug_log = False, write_log_to_file = False, per_epoch_saving = False,
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False, loss_function = "binary_crossentropy",
                 early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, dropout = 0.0,
                 use_batch_normalization = False, trainable_weights = False, embedding_path = embedding_path,
                 embedding_modification = embedding_modification )

    # Training/Evaluation Variables (Do Not Modify)
    ranking_per_epoch        = []
    ranking_per_epoch_value  = []
    number_of_ties_per_epoch = []
    loss_per_epoch           = []
    accuracy_per_epoch       = []
    precision_per_epoch      = []
    recall_per_epoch         = []
    f1_score_per_epoch       = []
    number_of_ties           = 0
    best_number_of_ties      = 0
    best_ranking             = sys.maxsize
    eval_data                = model.Get_Data_Loader().Read_Data( evaluation_data_file_path, keep_in_memory = False )

    print( "Preparing Evaluation Data" )

    model.Get_Data_Loader().Load_Embeddings( embedding_path )
    model.Get_Data_Loader().Generate_Token_IDs()

    # Vectorize Gold B Term And Entire Evaluation Data-set
    gold_b_input_1, gold_b_input_2, gold_b_input_3, _ = model.Encode_Model_Data( data_list = [gold_b_instance], model_type = "closed_discovery",
                                                                                 use_csr_format = True, is_crichton_format = True, keep_in_memory = False )
    eval_input_1, eval_input_2, eval_input_3, _       = model.Encode_Model_Data( data_list = eval_data, model_type = "closed_discovery",
                                                                                 use_csr_format = True, is_crichton_format = True, keep_in_memory = False )

    # Checks
    if gold_b_input_1 is None or gold_b_input_2 is None or gold_b_input_3 is None:
        print( "Error Occurred During Data Vectorization (Gold B)" )
        exit()
    if eval_input_1 is None or eval_input_2 is None or eval_input_3 is None:
        print( "Error Occurred During Data Vectorization (Evaluation Data)" )
        exit()

    model.Get_Data_Loader().Clear_Data()

    # Create Directory
    model.utils.Create_Path( model_save_path )

    print( "Beginning Model Data Preparation/Model Training" )

    for iteration in range( model_iterations ):
        # Train Model Over Data: "../data/train_cs1_closed_discovery_without_aggregators"
        model.Fit( training_data_file_path, epochs = 1, batch_size = batch_size, learning_rate = learning_rate, verbose = 1 )

        history = model.Get_Model().model_history.history
        loss_per_epoch.append( history['loss'][-1] )
        accuracy_per_epoch.append( history['accuracy'][-1] )
        precision_per_epoch.append( history['Precision'][-1] )
        recall_per_epoch.append( history['Recall'][-1] )
        f1_score_per_epoch.append( history['F1_Score'][-1] )

        # Ranking/Evaluation Variables
        b_prediction_dictionary = {}
        rank                    = 1
        number_of_ties          = 0

        # Get Prediction For Gold B Term
        gold_b_prediction_score = model.Predict( primary_input_matrix = gold_b_input_1, secondary_input_matrix = gold_b_input_2,
                                                 tertiary_input_matrix = gold_b_input_3, return_vector = True, return_raw_values = True )

        # Perform Prediction Over The Entire Evaluation Data-set (Model Inference)
        predictions = model.Predict( primary_input_matrix = eval_input_1, secondary_input_matrix = eval_input_2,
                                     tertiary_input_matrix = eval_input_3, return_vector = True, return_raw_values = True )

        print( "Performing Inference For Testing Instance Predictions" )

        # Perform Model Evaluation (Ranking Of Gold B Term)
        if isinstance( predictions, list ) and len( predictions ) == 0:
            print( "Error Occurred During Model Inference" )
            continue

        for instance, instance_prediction in zip( eval_data, predictions ):
            instance_tokens = instance.split()
            a_term = instance_tokens[0]
            b_term = instance_tokens[1]
            c_term = instance_tokens[2]

            if b_term not in b_prediction_dictionary:
                b_prediction_dictionary[b_term] = [instance_prediction]
            else:
                b_prediction_dictionary[b_term].append( instance_prediction )

        # Ranking Gold B Term Among All B Terms
        for b_term in b_prediction_dictionary:
            if b_term == gold_b_term: continue
            if b_prediction_dictionary[b_term] > gold_b_prediction_score:
                rank += 1
            elif b_prediction_dictionary[b_term] == gold_b_prediction_score:
                number_of_ties += 1

        number_of_ties_per_epoch.append( number_of_ties )

        ranking_per_epoch.append( rank )
        ranking_per_epoch_value.append( gold_b_prediction_score )

        # Keep Track Of The Best Rank
        if rank < best_ranking:
            best_ranking        = rank
            best_number_of_ties = number_of_ties
            # model.Save_Model( model_save_path )
            # model.Generate_Model_Metric_Plots( model_save_path )  # Useless Since Model Outputs Metics Every 1 Epoch, There's Nothing To Compare / See Metric Code Below

        print( "Epoch : " + str( iteration ) + " - Gold B: " + str( gold_b_term ) + " - Rank: " + str( rank ) +
               " Of " + str( len( eval_data ) ) + " Number Of B Terms" + " - Score: " + str( gold_b_prediction_score ) +
               " - Number Of Ties: " + str( number_of_ties ) )

    # Print Ranking Information Per Epoch
    print( "" )

    for epoch in range( len( ranking_per_epoch ) ):
        print( "Epoch: " + str( epoch ) + " - Rank: " + str( ranking_per_epoch[epoch] ) +
               " - Value: " + str( ranking_per_epoch_value[epoch] ) + " - Number Of Ties: " + str( number_of_ties_per_epoch[epoch] ) )

    print( "\nGenerating Model Metric Charts" )
    if not re.search( r"\/$", model_save_path ): model_save_path += "/"

    plt.plot( range( len( ranking_per_epoch ) ), ranking_per_epoch )
    plt.title( "Training: Rank vs Epoch" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Rank" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_rank.png" )
    plt.clf()

    plt.plot( range( len( number_of_ties_per_epoch ) ), number_of_ties_per_epoch )
    plt.title( "Training: Ties vs Epoch" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Ties" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_ties.png" )
    plt.clf()

    plt.plot( range( len( loss_per_epoch ) ), loss_per_epoch )
    plt.title( "Training: Loss vs Epoch" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Loss" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_loss.png" )
    plt.clf()

    plt.plot( range( len( accuracy_per_epoch ) ), accuracy_per_epoch )
    plt.title( "Training: Epoch vs Accuracy" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Accuracy" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_accuracy.png" )
    plt.clf()

    plt.plot( range( len( precision_per_epoch ) ), precision_per_epoch )
    plt.title( "Training: Epoch vs Precision" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Precision" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_precision.png" )
    plt.clf()

    plt.plot( range( len( recall_per_epoch ) ), recall_per_epoch )
    plt.title( "Training: Epoch vs Recall" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "Recall" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_recall.png" )
    plt.clf()

    plt.plot( range( len( f1_score_per_epoch ) ), f1_score_per_epoch )
    plt.title( "Training: Epoch vs F1-Score" )
    plt.xlabel( "Epoch" )
    plt.ylabel( "F1-Score" )
    plt.savefig( str( model_save_path ) + "training_epoch_vs_f1.png" )
    plt.clf()

    print( "\nBest Rank: " + str( best_ranking ) )
    print( "Number Of Ties With Best Rank: " + str( best_number_of_ties ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()