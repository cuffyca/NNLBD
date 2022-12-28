#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    03/30/2022                                                                   #
#    Revised: 06/10/2022                                                                   #
#                                                                                          #
#    Loads Trained Model For And Performs Manual Evaluation Against Training Data.         #
#        Used To Determine How Well The Model Generalized The Data For Extracting          #
#        Implicit Relations.                                                               #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python load_model_eval_test.py"                                      #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import os, re, sys
import numpy as np
import random

sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD      import LBD
from NNLBD.Misc import Utils


############################################################################################
#                                                                                          #
#    LBD Support Function(s)                                                               #
#                                                                                          #
############################################################################################

# Fetch Unique A & C Concepts Associated With A Desired B-Concept (Given A Set Of A-B-C Relations)
#   i.e. A-C Relation Given A Linking B-Concept From An A-B-C Relation
def Get_All_A_C_Concepts_Associated_With_B_Concept( data_list, desired_b_concept ):
    associated_a_c_relations = []

    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )

        if b_concept == desired_b_concept and a_concept + "\t" + c_concept not in associated_a_c_relations:
            associated_a_c_relations.append( a_concept + "\t" + c_concept )

    return associated_a_c_relations

# Fetch Unique B Concepts Associate With An A & C Concept (Given A Set Of A-B-C Relations)
#   i.e. Linking B-Concept Given An A-B-C Relation
def Get_All_B_Concepts_Associated_With_A_C_Relation( data_list, desired_a_concept, desired_c_concept ):
    unique_b_concepts = []

    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )

        if a_concept == desired_a_concept and c_concept == desired_c_concept and b_concept not in unique_b_concepts:
            unique_b_concepts.append( b_concept )

    return unique_b_concepts

# Fetch Unique A Or C Concepts Associate With A B-Concept (Given A Set Of A-B-C Relations)
#   i.e. A Given An A-B Relation Or C Given An B-C Relation
def Get_All_A_Or_C_Concepts_Associated_With_B_Concept( data_list, desired_b_concept ):
    unique_a_concepts, unique_c_concepts = []

    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )

        if b_concept == desired_b_concept and a_concept not in unique_a_concepts: unique_a_concepts.append( a_concept )
        if b_concept == desired_b_concept and c_concept not in unique_c_concepts: unique_c_concepts.append( c_concept )

    return unique_a_concepts, unique_c_concepts

# Fetch Unique B Concepts Associate With An A Or C-Concept (Given A Set Of A-B-C Relations)
#   i.e. B Given An A-B Relation Or B Given An B-C Relation
def Get_All_B_Concepts_Associated_With_A_Or_C_Concept( data_list, desired_a_concept, desired_c_concept ):
    unique_b_concepts = []

    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )

        if a_concept == desired_a_concept and b_concept not in unique_b_concepts: unique_b_concepts.append( b_concept )
        if c_concept == desired_c_concept and b_concept not in unique_b_concepts: unique_b_concepts.append( b_concept )

    return unique_b_concepts

# Fetch Linking B Concepts Given An A And C-Concept (Given A Set Of A-B-C Relations)
#   i.e. B Within An A-B Relation That Also Exists Within A B-C Relation
def Get_All_Linking_B_Concepts( data_list, desired_a_concept, desired_c_concept ):
    a_b_relations, c_b_relations = {}, {}

    desired_a_concept = desired_a_concept.lower()
    desired_c_concept = desired_c_concept.lower()

    # Aggregate A-B Relations And C-B Relations
    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )
        a_concept = a_concept.lower()
        b_concept = b_concept.lower()
        c_concept = c_concept.lower()

        if a_concept == desired_a_concept:
            if   a_concept not in a_b_relations:            a_b_relations[a_concept] = [b_concept]
            elif b_concept not in a_b_relations[a_concept]: a_b_relations[a_concept].append( b_concept )

        if c_concept == desired_c_concept:
            if   c_concept not in c_b_relations:            c_b_relations[c_concept] = [b_concept]
            elif b_concept not in c_b_relations[c_concept]: c_b_relations[c_concept].append( b_concept )

    # Determine If Similar B-Concepts Exists Between Both Sets Of Relations: A-B and C-B
    #   Return Unique List Of Linking B-Concepts, Linking A-Concept To C-Concept
    if len( a_b_relations ) == 0 or len( c_b_relations ) == 0:
        return []

    return list( set( [ b_concept for b_concept in a_b_relations[desired_a_concept] if b_concept in c_b_relations[desired_c_concept] ] ) )

# Performs Manual Closed Discovery Evaluation. Ranks A Gold B Given A Gold A-B-C Relation Among All Unique B Concepts
def Perform_Closed_Discovery_Evaluation_Ranking( model, gold_a_b_c_relation ):
    if not model:
        print( "Error: Model Not Defined" )
        return

    gold_a_concept, gold_b_concept, gold_c_concept = gold_a_b_c_relation.split( '\t' )
    model_predictions = model.Predict( primary_input = gold_a_concept, secondary_input = gold_c_concept, return_raw_values = True )

    if model_predictions.ndim == 2: model_predictions = model_predictions[0]

    # Compute Ranking Against Evaluation Concepts (Unique B Concepts) Or Complete List Of Unique Concepts
    if model.Get_Data_Loader().Get_Restrict_Output():
        # Unique B Concepts
        unique_b_concept_list = list( model.Get_Data_Loader().Get_Secondary_ID_Dictionary().keys() )
    else:
        # Complete List Of Unique Concepts
        unique_b_concept_list = list( model.Get_Data_Loader().Get_Token_ID_Dictionary().keys() )

    if len( unique_b_concept_list ) != len( model_predictions ):
        print( "Error: Unique B Concept List != Number Of Model Predictions / Unable To Perform Evaluation" )
        return

    probability_dict = {}

    # For Each Prediction From The Model, Store The Prediction Value And Unique Concept Token Within A Dictionary
    for token, prediction in zip( unique_b_concept_list, model_predictions ): probability_dict[token] = prediction

    # Sort Concept And Probability Dictionary In Reverse Order To Rank Concepts
    probability_dict = { k: v for k, v in sorted( probability_dict.items(), key = lambda x: x[1], reverse = True ) }

    # Get Index Of Desired Gold B
    gold_b_rank  = list( probability_dict.keys() ).index( gold_b_concept.lower() ) + 1
    gold_b_value = probability_dict[gold_b_concept.lower()]

    # Get Number Of Ties With Gold B Prediction Value
    gold_b_ties  = list( probability_dict.values() ).count( gold_b_value ) - 1

    print( "Gold B Rank : " + str( gold_b_rank  ) )
    print( "Gold B Ties : " + str( gold_b_ties  ) )
    print( "Gold B Value: " + str( gold_b_value ) )

# This Doesn't Actually Perform Any Evaluation Yet
def Manually_Evaluate_Data( model, data_list, a_b_c_relation ):
    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )
        predictions   = model.Predict( primary_input = a_concept, secondary_input = c_concept, return_raw_values = True  )

        if model.Get_Data_Loader().Get_Restrict_Output():
            b_concept_idx = model.Get_Data_Loader().Get_Token_ID( token = b_concept, token_type = "secondary"  )
        else:
            b_concept_idx = model.Get_Data_Loader().Get_Token_ID( token = b_concept )

        b_concept_prediction_value = predictions[0][b_concept_idx]

def Evaluate_Model_Predictions( model, encoded_primary_inputs, encoded_secondary_inputs, encoded_outputs, threshold_value = 0.5 ):
    batch_size        = 128
    number_of_batches = int( ( encoded_primary_inputs.shape[0] / batch_size ) + 1 )

    precision_per_batch, recall_per_batch, f1_score_per_batch = [], [], []

    for i in range( number_of_batches ):
        start_index, end_index = i * batch_size, ( i + 1 ) * batch_size

        # Assumes Data Is Encoded In 'csr_matrix' Format
        temp_encoded_primary_inputs   = encoded_primary_inputs[start_index:end_index,].todense()
        temp_encoded_secondary_inputs = encoded_secondary_inputs[start_index:end_index,].todense()
        temp_true_labels              = encoded_outputs[start_index:end_index,].todense()

        predictions = model.Predict( encoded_primary_input = temp_encoded_primary_inputs,
                                     encoded_secondary_input = temp_encoded_secondary_inputs,
                                     return_vector = True, return_raw_values = True )

        predictions = Threshold_Predictions( predictions = predictions, threshold_value = threshold_value )

        predictions      = np.asarray( predictions )
        temp_true_labels = np.asarray( temp_true_labels )

        precision   = Compute_Precision( predictions = predictions, true_labels = temp_true_labels )
        recall      = Compute_Recall( predictions = predictions, true_labels = temp_true_labels )
        f1_score    = Compute_F1_Score( predictions = predictions, true_labels = temp_true_labels )

        precision_per_batch.append( precision )
        recall_per_batch.append( recall )
        f1_score_per_batch.append( f1_score )

    precision_per_batch = np.asarray( precision_per_batch )
    recall_per_batch    = np.asarray( recall_per_batch    )
    f1_score_per_batch  = np.asarray( f1_score_per_batch  )

    print( "Precision: " + str( np.sum( precision_per_batch ) / precision_per_batch.shape[0] ) )
    print( "Recall   : " + str( np.sum( recall_per_batch    ) / recall_per_batch.shape[0]    ) )
    print( "F1-Score : " + str( np.sum( f1_score_per_batch  ) / f1_score_per_batch.shape[0]  ) )

# Generates Crichton Formatted Evaluation Data For Closed Discovery
#   Generates Random False A-B-C Gold Relation Given Data And Associated Evaluation Data
#   Tries To Find Linking Term Between A and C Concepts
def Generate_False_Closed_Discovery_Evaluation_Data( data_list, skip_relation_list = [] ):
    false_a_b_c_relation, false_evaluation_data             = None, None
    unique_a_concepts, unique_b_concepts, unique_c_concepts = [], [], []
    unique_a_b_relations, unique_b_c_relations              = [], []
    false_a_b_relations, false_b_c_relations                = [], []
    false_a_b_c_relations                                   = []

    # Build Unique Concept Lists
    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )
        a_b_relation, b_c_relation      = a_concept + "\t" + b_concept, b_concept + "\t" + c_concept

        if a_concept not in unique_a_concepts: unique_a_concepts.append( a_concept )
        if b_concept not in unique_b_concepts: unique_b_concepts.append( b_concept )
        if c_concept not in unique_c_concepts: unique_c_concepts.append( c_concept )

        if a_b_relation not in unique_a_b_relations: unique_a_b_relations.append( a_b_relation )
        if b_c_relation not in unique_b_c_relations: unique_b_c_relations.append( b_c_relation )

    # Build False A-B Relation / Check For Non-Existing A-B Relation In Data-set
    for a_concept in unique_a_concepts:
        for b_concept in unique_b_concepts:
            possible_a_b_relation = a_concept + "\t" + b_concept
            possible_b_a_relation = b_concept + "\t" + a_concept

            if possible_a_b_relation in unique_a_b_relations: continue
            if possible_b_a_relation in unique_a_b_relations: continue

            false_a_b_relations.append( possible_a_b_relation )

    # Build False B-C Relation / Check For Non-Existing B-C Relation In Data-set
    for b_concept in unique_b_concepts:
        for c_concept in unique_c_concepts:
            possible_b_c_relation = b_concept + "\t" + c_concept
            possible_c_b_relation = c_concept + "\t" + b_concept

            if possible_b_c_relation in unique_b_c_relations: continue
            if possible_c_b_relation in unique_b_c_relations: continue

            false_b_c_relations.append( possible_b_c_relation )

    # Generate New False A-B-C Link
    for a_b_relation in false_a_b_relations:
        a_concept, b_concept = a_b_relation.split( '\t' )

        for b_c_relation in false_b_c_relations:
            if b_concept in b_c_relation: false_a_b_c_relations.append( a_concept + "\t" + b_c_relation )

    # Select A Random False A-B-C Relation
    while True:
        random_index         = random.randint( 0, len( false_a_b_c_relations ) )
        false_a_b_c_relation = false_a_b_c_relations[random_index]
        if false_a_b_c_relation not in skip_relation_list: break

    # Now We Have A False A-B-C Link, Let's Generate The Crichton Formatted Evaluation Data
    false_a_concept, _, false_c_concept = false_a_b_c_relation.split( '\t' )

    false_evaluation_data = [false_a_concept + "\t" + b_concept + "\t" + false_c_concept for b_concept in unique_b_concepts]

    return false_a_b_c_relation, false_evaluation_data

# Generates Crichton Formatted Evaluation Data For Closed Discovery
#   Generates Random False A-B-C Gold Relation Given Data And Associated Evaluation Data
#   Generates Completely Random A-B-C Relation / No Linking B Term Between A And C Concepts
def Generate_Random_False_Closed_Discovery_Evaluation_Data( data_list, skip_relation_list = [] ):
    false_a_b_c_relation, false_evaluation_data             = None, None
    unique_a_concepts, unique_b_concepts, unique_c_concepts = [], [], []
    false_a_b_c_relations                                   = []

    # Build Unique Concept Lists
    for a_b_c_relation in data_list:
        a_concept, b_concept, c_concept = a_b_c_relation.split( '\t' )

        if a_concept not in unique_a_concepts: unique_a_concepts.append( a_concept )
        if b_concept not in unique_b_concepts: unique_b_concepts.append( b_concept )
        if c_concept not in unique_c_concepts: unique_c_concepts.append( c_concept )

    # Select A Random A, B & C Concepts To Form Random A-B-C Relation
    while True:
        a_concept = unique_a_concepts[random.randint( 0, len( unique_a_concepts ) )]
        b_concept = unique_b_concepts[random.randint( 0, len( unique_b_concepts ) )]
        c_concept = unique_c_concepts[random.randint( 0, len( unique_c_concepts ) )]

        if a_concept == b_concept or b_concept == c_concept or a_concept == c_concept: continue

        false_a_b_c_relation = a_concept + "\t" + b_concept + "\t" + c_concept
        if false_a_b_c_relation not in skip_relation_list: break

    # Now We Have A False A-B-C Link, Let's Generate The Crichton Formatted Evaluation Data
    random_a_concept, _, random_c_concept = false_a_b_c_relation.split( '\t' )

    false_evaluation_data = [random_a_concept + "\t" + b_concept + "\t" + random_c_concept for b_concept in unique_b_concepts]

    return false_a_b_c_relation, false_evaluation_data


############################################################################################
#                                                                                          #
#    Model Metric Function(s)                                                              #
#                                                                                          #
############################################################################################

def Threshold_Predictions( predictions, threshold_value = 0.5 ):
    temp_predictions = predictions.copy()

    if temp_predictions.ndim == 1:
        for i in range( temp_predictions.shape[0] ):
            temp_predictions[i] = 1 if temp_predictions[i] > threshold_value else 0
    elif temp_predictions.ndim == 2:
        for i in range( temp_predictions.shape[0] ):
            for j in range( temp_predictions.shape[1] ):
                temp_predictions[i][j] = 1 if temp_predictions[i][j] > threshold_value else 0

    return temp_predictions

def Compute_Precision( predictions, true_labels ):
    if predictions.ndim == 3: predictions = predictions[0]
    if true_labels.ndim == 3: true_labels = true_labels[0]

    if not isinstance( predictions, np.ndarray ): predictions = np.asarray( predictions )
    if not isinstance( true_labels, np.ndarray ): true_labels = np.asarray( true_labels )

    true_positive_count = np.sum( predictions * true_labels )
    true_positive_false_positive_count = np.sum( predictions )

    # Compute Precision
    return true_positive_count / ( true_positive_false_positive_count + np.finfo(float).eps )

def Compute_Recall( predictions, true_labels ):
    if predictions.ndim == 3: predictions = predictions[0]
    if true_labels.ndim == 3: true_labels = true_labels[0]

    if not isinstance( predictions, np.ndarray ): predictions = np.asarray( predictions )
    if not isinstance( true_labels, np.ndarray ): true_labels = np.asarray( true_labels )

    true_positive_count = np.sum( predictions * true_labels )
    true_positive_false_negative_count = np.sum( true_labels )

    # Compute Recall
    return true_positive_count / ( true_positive_false_negative_count + np.finfo(float).eps )

def Compute_F1_Score( predictions, true_labels ):
    if predictions.ndim == 3: predictions = predictions[0]
    if true_labels.ndim == 3: true_labels = true_labels[0]

    if not isinstance( predictions, np.ndarray ): predictions = np.asarray( predictions )
    if not isinstance( true_labels, np.ndarray ): true_labels = np.asarray( true_labels )

    precision = Compute_Precision( predictions = predictions, true_labels = true_labels )
    recall    = Compute_Recall( predictions = predictions, true_labels = true_labels )

    # Compute F1-Score
    return 2 * ( ( precision * recall ) / ( precision + recall + np.finfo(float).eps ) )


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

def Main():
    training_data_set_path = "../data/train_cs1_closed_discovery_without_aggregators_mod_nns"
    trained_model_path     = "../saved_models/test_avg_fs_ro_lr0.0001/cs1_crichton_hinton_softplus_model_1_best_model"
    false_data_save_path   = "./"
    test_a_concept         = "mesh:c044115"
    test_b_concept         = ""
    test_c_concept         = "chebi:17719"
    gold_a_b_c_relation    = "PR:000011331\tHOC:42\tPR:000005308"

    # Check(s)
    if not Utils().Check_If_Path_Exists( path = trained_model_path ):
        print( "Error: Specified Model Path Does Not Exist" )
        return

    # Load Trained Model
    model = LBD()
    model.Load_Model( model_path = trained_model_path )

    # Report Model Loading Error
    if not model.Is_Model_Loaded():
        print( "Error: Model Is Not Able To Be Loaded" )
        return

    # Test Prediction
    predictions = model.Predict( primary_input = test_a_concept, secondary_input = test_c_concept,
                                 return_vector = True, return_raw_values = True )

    if predictions.ndim == 2: predictions = predictions[0]

    # Threshold_Predictions( predictions = predictions, threshold_value = 0.002 )

    # Read Training Data
    data_list   = model.Read_Data( file_path = training_data_set_path, keep_in_memory = False )

    # Manually Evaluate Training Data Predictions - (This Doesn't Actually Perform Any Evaluation Yet)
    # Manually_Evaluate_Data( model = model, data_list = data_list, a_b_c_relation = gold_a_b_c_relation )

    # Evaluate Closed Discovery Gold A-B-C Relation
    # Perform_Closed_Discovery_Evaluation_Ranking( model = model, gold_a_b_c_relation = gold_a_b_c_relation )

    # Get All B Concepts Associated With Gold A & C - This Should Return Back An Empty List Since We're Doing Time-Slicing
    #   Data List Is Composed Of A-B-C Relations Prior To The Explicit Gold A-B-C Discovery
    gold_a_concept, gold_b_concept, gold_c_concept = gold_a_b_c_relation.split( '\t' )
    # associated_b_concepts = Get_All_B_Concepts_Associated_With_A_C_Relation( data_list = data_list,
    #                                                                          desired_a_concept = gold_a_concept,
    #                                                                          desired_c_concept = gold_c_concept )

    # associated_b_concepts = Get_All_B_Concepts_Associated_With_A_Or_C_Concept( data_list = data_list,
    #                                                                            desired_a_concept = gold_a_concept,
    #                                                                            desired_c_concept = gold_c_concept )

    linking_b_concepts = Get_All_Linking_B_Concepts( data_list = data_list,
                                                     desired_a_concept = gold_a_concept,
                                                     desired_c_concept = gold_c_concept )

    # Generate 10 Random/False A-B-C Relations And Related Evaluation Data
    false_gold_a_b_c_relation_list, false_evaluation_data_list, skip_relation_list = [], [], [gold_a_b_c_relation]

    for _ in range( 10 ):
        false_gold_a_b_c_relation, false_evaluation_data = Generate_Random_False_Closed_Discovery_Evaluation_Data( data_list = data_list,
                                                                                                                   skip_relation_list = skip_relation_list )
        # Add Generated False A-B-C Relation To Skip List (Prevent Duplicate Relation Creation)
        skip_relation_list.append( false_gold_a_b_c_relation )

        # Add Generated Data To Lists
        temp_a, temp_b, temp_c = false_gold_a_b_c_relation.split( '\t' )
        false_gold_a_b_c_relation = temp_a + "\\t" + temp_b + "\\t" + temp_c
        false_gold_a_b_c_relation_list.append( false_gold_a_b_c_relation )
        false_evaluation_data_list.append( false_evaluation_data )

    # Save False/Random Gold A-B-C Relations And Evaluation Data
    print( "Generating False Gold A-B-C Instance And Evaluation Data" )

    temp_eval_file_name_dict = {}

    for i, ( false_gold_a_b_c_relation, data ) in enumerate( zip( false_gold_a_b_c_relation_list, false_evaluation_data_list ) ):
        save_name    = "cs1_false_eval_data_" + str( i )
        data_to_save = "\n".join( data )
        Utils().Write_Data_To_File( file_path = false_data_save_path + "/" + str( save_name ), data = data_to_save )
        temp_eval_file_name_dict[save_name] = false_gold_a_b_c_relation

    temp_data = [false_eval_file_name + " => " + false_gold_a_b_c_relation for false_eval_file_name, false_gold_a_b_c_relation in temp_eval_file_name_dict.items()]
    temp_data = "\n".join( temp_data )
    Utils().Write_Data_To_File( file_path = false_data_save_path + "/cs1_false_a_b_c_gold_relations.txt", data = temp_data )

    print( "Complete - Evaluation Data Saved" )

    # -------------------------------------------------------------------------- #
    #  We Know The Model Ranks The Gold A-B-C Within The Top 10 Results.         #
    #  Let's Investigate If The Model As Actually Generalized The Training Data. #
    # -------------------------------------------------------------------------- #

    # Encode Our Training Data
    # encoded_primary_inputs, encoded_secondary_inputs, _, encoded_outputs = model.Encode_Model_Data( data_list = data_list,
    #                                                                                                 use_csr_format = True,
    #                                                                                                 keep_in_memory = False )

    # Evaluate_Model_Predictions( model = model, encoded_primary_inputs = encoded_primary_inputs,
    #                             encoded_secondary_inputs = encoded_secondary_inputs,
    #                             encoded_outputs = encoded_outputs, threshold_value = 0.5 )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()