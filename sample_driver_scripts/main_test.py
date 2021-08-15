#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    08/05/2020                                                                   #
#    Revised: 01/14/2020                                                                   #
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
    model = LBD( network_model = "hinton", model_type = "open_discovery", bilstm_merge_mode = "concat",
                 print_debug_log = True, write_log_to_file = True, per_epoch_saving = False, activation_function = "softplus",
                 use_csr_format = True, use_gpu = True, enable_early_stopping = False, final_layer_type = "dense",
                 early_stopping_metric_monitor = "F1_Score", early_stopping_persistence = 3, dropout = 0.5,
                 use_batch_normalization = False, trainable_weights = True, embedding_path = "../vectors/vectors_random_cui_mini",
                 restrict_output = True )

    # Train Model Over Data: "data/cui_mini"
    model.Fit( "../data/cui_mini", learning_rate = 0.001, epochs = 40, batch_size = 32, verbose = 1 )
    # model.Save_Model( "../test_model" )
    # model.Generate_Model_Metric_Plots()

    # Predict Outputs Given Input Instance (Primary & Secondary Input Instances)
    cui_input       = "C001"
    predicate_input = "TREATS"
    cui_outputs     = "C002 C004 C009"

    print( ""                                     )
    print( "Prediction Inputs (Text):"            )
    print( "         (CUI) ->", cui_input         )
    print( "   (Predicate) ->", predicate_input   )
    print( "True Outputs (Text):"                 )
    print( "         (CUI) ->", cui_outputs, "\n" )

    model_prediction = model.Predict( cui_input, predicate_input, return_vector = False, return_raw_values = False )
    print( "Model Prediction:", str( model_prediction ) )
    print( "Using Embeddings:", str( model.Is_Embeddings_Loaded() ) )

    if model.Get_Network_Model() == "bilstm" or model.Is_Embeddings_Loaded():
        cui_input, predicate_input, _, cui_outputs = model.Encode_Model_Instance( cui_input, predicate_input, "", cui_outputs,
                                                                                  pad_inputs = False, pad_output = True,
                                                                                  model_type = model.Get_Model_Type() )
    elif model.Get_Network_Model() == "rumelhart" or model.Get_Network_Model() == "hinton":
        cui_input, predicate_input, _, cui_outputs = model.Encode_Model_Instance( cui_input, predicate_input, "", cui_outputs,
                                                                                  pad_inputs = False, pad_output = True,
                                                                                  model_type = model.Get_Model_Type() )

    print( "Prediction Inputs (Vectorized):"      )
    print( "         (CUI) ->", cui_input         )
    print( "   (Predicate) ->", predicate_input   )
    print( "True Outputs (Vectorized):"           )
    print( "         (CUI) ->", cui_outputs, "\n" )

    # Print Open Discovery Prediction Vectors
    if model.Get_Model_Type() == "open_discovery":
        model_prediction = model.Predict_Vector( cui_input, predicate_input, return_vector = True, return_raw_values = False )
        print( "Model Prediction Vector:", model_prediction )
    # Print Closed Discovery Prediction Vectors
    else:
        for index in range( len( cui_input ) ):
            model_prediction = model.Predict_Vector( cui_input[index], predicate_input[index], return_vector = True, return_raw_values = False )
            print( "Model Prediction Vector:", model_prediction )


    # Evaluate Model Of Entire Data-set
    accuracy = model.Evaluate_Prediction( "../data/cui_mini_eval" )

    print( "Evaluation Scores: " )
    # print( "    F1 Score : {:.4f}" . format( f1_score  ) )
    # print( "    Precision: {:.4f}" . format( precision ) )
    # print( "    Recall   : {:.4f}" . format( recall    ) )
    print( "    Accuracy : {:.4f}" . format( accuracy  ) )
    # print( "    Loss     : {:.4f}" . format( loss      ) )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()