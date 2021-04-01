#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    02/14/2021                                                                   #
#    Revised: 03/31/2021                                                                   #
#                                                                                          #
#    Reads JSON experiment configuration data and runs LBD class using JSON data.          #
#        Driver Script                                                                     #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python LBDDriver.py experiment_parameter_file.json"                  #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import json, os, re, sys, time
import matplotlib.pyplot as plt
import subprocess as sp
from shutil import copy2

sys.path.insert( 0, "../" )

# Fixes Recursion Limit Issue
#   Default = 10 ** 4
# sys.setrecursionlimit( 10**6 )

# Custom Modules
from NNLBD import LBD, Utils


############################################################################################
#                                                                                          #
#    NNLBD JSON Driver Class                                                               #
#                                                                                          #
############################################################################################

class NNLBD_Driver:
    def __init__( self ):
        # Global Parameters
        self.number_of_iterations        = 1
        self.json_data                   = None
        self.global_device_name          = "/gpu:0"

        # Check For Available GPU Parmeters (Not Model Related)
        self.enable_gpu_polling          = False
        self.available_device_name       = ""
        self.acceptable_available_memory = 4096

        # Private Variables (Do Not Modify)
        self.json_file_path              = ""


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Reads JSON Data From File
    """
    def Read_JSON_Data( self, json_file_path ):
        json_data = {}

        # Read JSON Data From File
        if Utils().Check_If_File_Exists( json_file_path ):
            self.json_file_path = json_file_path

            with open( json_file_path ) as json_file:
                json_data = json.load( json_file )

            self.json_data = json_data

        else:
            print( "Error: Specified JSON File Does Not Exist" )

        return json_data

    """
        Extracts Global Settings From JSON Dictionary Data
    """
    def Extract_Global_Settings( self, data_dict = None ):
        # Check
        if data_dict is None and "global_settings" in self.json_data:
            data_dict = self.json_data["global_settings"][0]

        if "device_name"                 in data_dict: self.global_device_name          = str( data_dict["device_name"] )
        if "enable_gpu_polling"          in data_dict: self.enable_gpu_polling          = True if data_dict["enable_gpu_polling"] == "True" else False
        if "number_of_iterations"        in data_dict: self.number_of_iterations        = int( data_dict["number_of_iterations"] )
        if "acceptable_available_memory" in data_dict: self.acceptable_available_memory = int( data_dict["acceptable_available_memory"] )

    """
        Print All JSON Data Read From File
    """
    def Print_All_JSON_Data( self, json_data = None ):
        # Check
        if json_data is None:
            json_data = self.json_data

        for experiment_name in json_data:
            print( str( experiment_name ) )

            # Get 1st Dictionary (Should Only Be One Anyway)
            experiment_variables = json_data[experiment_name][0]

            self.Print_JSON_Dictionary_Data( experiment_variables )

    """
        Prints JSON Dictionary Data
    """
    def Print_JSON_Dictionary_Data( self, data_dict ):
        for variable_name in data_dict:
            print( str( variable_name ) + " : " + str( data_dict[variable_name] ) )

    """
        Saves Data Plots/Graphs
    """
    def Generate_Plot( self, data_list = [], title = "Title", x_label = "X_Label",
                       y_label = "Y_Label", file_name = "file_name.png", save_path = "./" ):
        plt.plot( range( len( data_list ) ), data_list )
        plt.title( str( title ) )
        plt.xlabel( str( x_label ) )
        plt.ylabel( str( y_label ) )
        plt.savefig( str( save_path ) + str( file_name ) )
        plt.clf()

    """
        Run Extracted Experiment Setups From JSON Data
    """
    def Run_Experiments( self, json_dict = None ):
        # Check
        if json_dict is None: json_dict = self.json_data

        # LBD Experiment Variables
        print_debug_log, write_log_to_file, optimizer, activation_function, loss_function           = False, False, "adam", "sigmoid", "binary_crossentropy"
        network_model, model_type, use_gpu, device_name, trainable_weights, final_layer_type        = "rumelhart", "open_discovery", True, self.global_device_name, False, "dense"
        bilstm_merge_mode, bilstm_dimension_size, learning_rate, epochs, momentum, dropout          = "concat", 64, 0.005, 30, 0.05, 0.5
        batch_size, prediction_threshold, shuffle, embedding_path, use_csr_format, per_epoch_saving = 32, 0.5, True, "", True, False
        margin, scale, verbose, train_data_path, enable_tensorboard_logs, enable_early_stopping     = 30.0, 0.35, "", False, False, False
        early_stopping_metric_monitor, early_stopping_persistence, use_batch_normalization          = "loss", 3, False
        embedding_modification, skip_out_of_vocabulary_words, eval_data_path, checkpoint_directory  = "concatenate", True, "", "ckpt_models"
        model_save_path, model_load_path, set_per_iteration_model_path, learning_rate_decay         = "", "", False, 0.004
        feature_scale_value                                                                         = 1.0

        # Model Variables
        run_eval_number_epoch = 1
        gold_b_instance       = None

        # Run Experiments
        for iter in range( 1, self.number_of_iterations + 1 ):
            for run_id in json_dict:
                # Skip Global Variable Dictionary
                if re.search( r'[Gg]lobal_[Ss]ettings', run_id ): continue

                print( "\nBuilding LBD Experiment Run ID: " + str( run_id ) + "\n" )

                # Extract Experiment JSON Data Dictionary
                run_dict = json_dict[run_id][0]

                # Extract LBD Variable Data From JSON Run Dictionary Data
                if "print_debug_log"               in run_dict: print_debug_log               = True if run_dict["print_debug_log"]              == "True" else False
                if "write_log_to_file"             in run_dict: write_log_to_file             = True if run_dict["write_log_to_file"]            == "True" else False
                if "per_epoch_saving"              in run_dict: per_epoch_saving              = True if run_dict["per_epoch_saving"]             == "True" else False
                if "use_gpu"                       in run_dict: use_gpu                       = True if run_dict["use_gpu"]                      == "True" else False
                if "skip_out_of_vocabulary_words"  in run_dict: skip_out_of_vocabulary_words  = True if run_dict["skip_out_of_vocabulary_words"] == "True" else False
                if "use_csr_format"                in run_dict: use_csr_format                = True if run_dict["use_csr_format"]               == "True" else False
                if "trainable_weights"             in run_dict: trainable_weights             = True if run_dict["trainable_weights"]            == "True" else False
                if "shuffle"                       in run_dict: shuffle                       = True if run_dict["shuffle"]                      == "True" else False
                if "enable_tensorboard_logs"       in run_dict: enable_tensorboard_logs       = True if run_dict["enable_tensorboard_logs"]      == "True" else False
                if "enable_early_stopping"         in run_dict: enable_early_stopping         = True if run_dict["enable_early_stopping"]        == "True" else False
                if "use_batch_normalization"       in run_dict: use_batch_normalization       = True if run_dict["use_batch_normalization"]      == "True" else False
                if "set_per_iteration_model_path"  in run_dict: set_per_iteration_model_path  = True if run_dict["set_per_iteration_model_path"] == "True" else False
                if "network_model"                 in run_dict: network_model                 = run_dict["network_model"]
                if "model_type"                    in run_dict: model_type                    = run_dict["model_type"]
                if "activation_function"           in run_dict: activation_function           = run_dict["activation_function"]
                if "loss_function"                 in run_dict: loss_function                 = run_dict["loss_function"]
                if "embedding_path"                in run_dict: embedding_path                = run_dict["embedding_path"]
                if "train_data_path"               in run_dict: train_data_path               = run_dict["train_data_path"]
                if "eval_data_path"                in run_dict: eval_data_path                = run_dict["eval_data_path"]
                if "model_save_path"               in run_dict: model_save_path               = run_dict["model_save_path"]
                if "model_load_path"               in run_dict: model_load_path               = run_dict["model_load_path"]
                if "checkpoint_directory"          in run_dict: checkpoint_directory          = run_dict["checkpoint_directory"]
                if "epochs"                        in run_dict: epochs                        = int( run_dict["epochs"] )
                if "verbose"                       in run_dict: verbose                       = int( run_dict["verbose"] )
                if "learning_rate"                 in run_dict: learning_rate                 = float( run_dict["learning_rate"] )
                if "learning_rate_decay"           in run_dict: learning_rate_decay           = float( run_dict["learning_rate_decay"] )
                if "feature_scale_value"           in run_dict: feature_scale_value           = float( run_dict["feature_scale_value"] )
                if "batch_size"                    in run_dict: batch_size                    = int( run_dict["batch_size"] )
                if "optimizer"                     in run_dict: optimizer                     = run_dict["optimizer"]
                if "device_name"                   in run_dict: device_name                   = run_dict["device_name"]
                if "final_layer_type"              in run_dict: final_layer_type              = run_dict["final_layer_type"]
                if "bilstm_merge_mode"             in run_dict: bilstm_merge_mode             = run_dict["bilstm_merge_mode"]
                if "bilstm_dimension_size"         in run_dict: bilstm_dimension_size         = int( run_dict["bilstm_dimension_size"] )
                if "dropout"                       in run_dict: dropout                       = float( run_dict["dropout"] )
                if "momentum"                      in run_dict: momentum                      = float( run_dict["momentum"] )
                if "early_stopping_metric_monitor" in run_dict: early_stopping_metric_monitor = run_dict["early_stopping_metric_monitor"]
                if "early_stopping_persistence"    in run_dict: early_stopping_persistence    = int( run_dict["early_stopping_persistence"] )
                if "prediction_threshold"          in run_dict: prediction_threshold          = float( run_dict["prediction_threshold"] )
                if "margin"                        in run_dict: margin                        = float( run_dict["margin"] )
                if "scale"                         in run_dict: scale                         = float( run_dict["scale"] )
                if "embedding_modification"        in run_dict: embedding_modification        = run_dict["embedding_modification"]
                if "run_eval_number_epoch"         in run_dict: run_eval_number_epoch         = run_dict["run_eval_number_epoch"]
                if "gold_b_instance"               in run_dict: gold_b_instance               = run_dict["gold_b_instance"]


                # Wait For Next Available GPU
                if self.enable_gpu_polling and self.available_device_name == "":
                    print( "*** Waiting For The Next Available GPU ***" )
                    self.available_device_name = self.Get_Next_Available_CUDA_GPU( self.acceptable_available_memory )

                    if re.search( r'[Gg][Pp][Uu]', self.available_device_name ):
                        print( "*** Using Available GPU Selected: '" + str( self.available_device_name ) + "' ***\n" )
                    else:
                        print( "*** Warning: Unable To Allocate A GPU / Using CPU ***" )

                    # Check CUDA GPU Device Override Setting
                    if self.available_device_name != "": device_name = self.available_device_name

                # Create LBD Class
                model = LBD( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, network_model = network_model, model_type = model_type, dropout = dropout,
                             optimizer = optimizer, activation_function = activation_function, loss_function = loss_function, checkpoint_directory = checkpoint_directory, shuffle = shuffle,
                             use_gpu = use_gpu, device_name = device_name, trainable_weights = trainable_weights,  final_layer_type = final_layer_type, bilstm_merge_mode = bilstm_merge_mode,
                             bilstm_dimension_size = bilstm_dimension_size, learning_rate = learning_rate, epochs = epochs, momentum = momentum, batch_size = batch_size,  verbose = verbose,
                             prediction_threshold = prediction_threshold, embedding_path = embedding_path, use_csr_format = use_csr_format, per_epoch_saving = per_epoch_saving,
                             margin = margin, scale = scale, enable_early_stopping = enable_early_stopping,  early_stopping_metric_monitor = early_stopping_metric_monitor,
                             use_batch_normalization = use_batch_normalization, embedding_modification = embedding_modification,  enable_tensorboard_logs = enable_tensorboard_logs,
                             early_stopping_persistence = early_stopping_persistence, skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, learning_rate_decay = learning_rate_decay,
                             feature_scale_value = feature_scale_value )

                ######################################################
                # Determine What Type Of Experiment Will Be Executed #
                ######################################################

                # Adjust Model Save Path To Differentiate Each Model For Each Iteration
                if set_per_iteration_model_path and model_save_path != "": model_save_path = model_save_path + "_" + str( iter )

                # Train Model
                if re.match( r"^[Tt]rain_\d+", run_id ):
                    model.Fit( training_file_path = train_data_path )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )
                        model.Generate_Model_Metric_Plots( model_save_path )

                # Evaluate Model
                elif re.match( r"^[Ee]val_\d+", run_id ):
                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        model.Evaluate( training_file_path = train_data_path )

                # Evaluate Model For Prediction
                elif re.match( r"^[Ee]val_[Pp]rediction_\d+", run_id ):
                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        model.Evaluate_Prediction( training_file_path = train_data_path )

                # Evaluate Model For Ranking
                elif re.match( r"^[Ee]val_[Rr]anking_\d+", run_id ):
                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        model.Evaluate_Prediction( training_file_path = train_data_path )

                # Train Model And Evaluate
                elif re.match( r"^[Tt]rain_[Aa]nd_[Ee]val_\d+", run_id ):
                    model.Fit( training_file_path = train_data_path )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )
                        model.Generate_Model_Metric_Plots( model_save_path )
                        model.Evaluate( test_file_path = eval_data_path )

                # Train Model And Evaluate Prediction
                elif re.match( r"^[Tt]rain_[Aa]nd_[Ee]val_[Pp]rediction_\d+", run_id ):
                    model.Fit( training_file_path = train_data_path )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )
                        model.Generate_Model_Metric_Plots( model_save_path )
                        model.Evaluate_Prediction( test_file_path = eval_data_path )

                # Train Model And Evaluate Ranking
                elif re.match( r"^[Tt]rain_[Aa]nd_[Ee]val_[Rr]anking_\d+", run_id ):
                    model.Fit( training_file_path = train_data_path )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )
                        model.Generate_Model_Metric_Plots( model_save_path )
                        model.Evaluate_Ranking( test_file_path = eval_data_path )

                # Refine Existing Model
                elif re.match( r"^[Rr]efine_\d+", run_id ):
                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        model.Fit( training_file_path = train_data_path )

                        if model_save_path != "":
                            model.Save_Model( model_save_path )
                            model.Generate_Model_Metric_Plots( model_save_path )

                # Run Crichton Closed Discovery Train And Eval - (Simple Model)
                elif re.match( r"^[Cc]richton_[Cc]losed_[Dd]iscovery_[Tt]rain_[Aa]nd_[Ee]val_\d+", run_id ):
                    if gold_b_instance is None:
                        print( " Error: No Gold B Instance Defined" )
                        continue
                    if model_type != "closed_discovery":
                        print( " Error: Only Closed Discovery Supported" )
                        continue

                    self.Crichton_Closed_Discovery_Train_And_Eval( model = model, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, verbose = verbose,
                                                                   run_eval_number_epoch = run_eval_number_epoch, train_data_path = train_data_path, eval_data_path = eval_data_path,
                                                                   model_save_path = model_save_path, embedding_path = embedding_path, gold_b_instance = gold_b_instance )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )

                # Run Crichton Closed Discovery Refine And Eval - (Simple Model Refinement)
                elif re.match( r"^[Cc]richton_[Cc]losed_[Dd]iscovery_[Rr]efine_[Aa]nd_[Ee]val_\d+", run_id ):
                    if gold_b_instance is None:
                        print( " Error: No Gold B Instance Defined" )
                        continue
                    if model_type != "closed_discovery":
                        print( " Error: Only Closed Discovery Supported" )
                        continue

                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        self.Crichton_Closed_Discovery_Train_And_Eval( model = model, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate,
                                                                       verbose = verbose, run_eval_number_epoch = run_eval_number_epoch, train_data_path = train_data_path,
                                                                       eval_data_path = eval_data_path, model_save_path = model_save_path, embedding_path = embedding_path,
                                                                       gold_b_instance = gold_b_instance )

                        if model_save_path != "":
                            model.Save_Model( model_save_path )

                    else:
                        print( "Error: Unable To Load Model To Refine / No Model Path Specified" )

                # Run Closed Discovery Train And Eval - (All Models Except 'Simple' Model)
                elif re.match( r"^[Cc]losed_[Dd]iscovery_[Tt]rain_[Aa]nd_[Ee]val_\d+", run_id ):
                    if gold_b_instance is None:
                        print( " Error: No Gold B Instance Defined" )
                        continue
                    if model_type != "closed_discovery":
                        print( " Error: Only Closed Discovery Supported" )
                        continue

                    self.Closed_Discovery_Train_And_Eval( model = model, epochs = epochs, batch_size = batch_size,
                                                          learning_rate = learning_rate, verbose = verbose,
                                                          run_eval_number_epoch = run_eval_number_epoch,
                                                          train_data_path = train_data_path, model_save_path = model_save_path,
                                                          embedding_path = embedding_path, gold_b_instance = gold_b_instance )

                    if model_save_path != "":
                        model.Save_Model( model_save_path )

                # Run Closed Discovery Refine And Eval - (All Models Except 'Simple' Model)
                elif re.match( r"^[Cc]losed_[Dd]iscovery_[Rr]efine_[Aa]nd_[Ee]val_\d+", run_id ):
                    if gold_b_instance is None:
                        print( " Error: No Gold B Instance Defined" )
                        continue
                    if model_type != "closed_discovery":
                        print( " Error: Only Closed Discovery Supported" )
                        continue

                    if model_load_path != "":
                        if not model.Load_Model( model_load_path ): continue

                        # Update Model Parameters
                        model.Update_Model_Parameters( print_debug_log = print_debug_log, epochs = epochs, per_epoch_saving = per_epoch_saving,
                                                       batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle,
                                                       embedding_path = embedding_path, verbose = verbose )

                        self.Closed_Discovery_Train_And_Eval( model = model, epochs = epochs, batch_size = batch_size,
                                                              learning_rate = learning_rate, verbose = verbose,
                                                              run_eval_number_epoch = run_eval_number_epoch,
                                                              train_data_path = train_data_path, model_save_path = model_save_path,
                                                              embedding_path = embedding_path, gold_b_instance = gold_b_instance )

                        if model_save_path != "":
                            model.Save_Model( model_save_path )

                # Copy JSON File To Model Save Path
                if model_save_path != "":
                    copy2( self.json_file_path, model_save_path )

                # Clean-Up
                model = None

    ############################################################################################
    #                                                                                          #
    #    GPU Polling Function                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Waits For CUDA GPU To Become Available
          - Only Polls For A Single GPU Up To 2 Weeks.
          - For Multi-GPU Polling Use 'BaseModel::Initialize_GPU()' Function.
    """
    def Get_Next_Available_CUDA_GPU( self, acceptable_available_memory, polling_counter_limit = 1209600 ):
        polling_timer_exceeded = False
        available_device_id    = ""
        polling_counter        = 0
        COMMAND                = "nvidia-smi --query-gpu=memory.free --format=csv"

        while available_device_id == "":
            try:
                _output_to_list    = lambda x: x.decode( 'ascii' ).split( '\n' )[:-1]
                memory_free_info   = _output_to_list( sp.check_output( COMMAND.split() ) )[1:]
                memory_free_values = [int( x.split()[0] ) for i, x in enumerate( memory_free_info )]
                available_gpus     = [i for i, x in enumerate( memory_free_values ) if x > acceptable_available_memory]

                # Choose First Available GPU
                if len( available_gpus ) > 0: available_device_id = "/gpu:" + str( available_gpus[0] )

                # Wait For One Second And Then Check Again
                time.sleep( 1 )

                # Increment Polling Counter
                polling_counter += 1

                if polling_counter >= polling_counter_limit:
                    polling_timer_exceeded = True
                    break

            except Exception as e:
                print( "LBDDriver::Get_Next_Available_CUDA_GPU() - Warning: 'nvidia-smi' Not Detected In Path / Unable To Detect Available GPUs." )
                print( "                                         - Setting Device Name To CPU" )
                print( "                                         - " + str( e ) )
                available_device_id = "/cpu:0"
                break

        if polling_timer_exceeded:
            print( "LBDDriver::Get_Next_Available_CUDA_GPU() - Error: Unable To Secure Available GPU Within A 2 Week Period / Terminating Program" )

        return available_device_id


    ############################################################################################
    #                                                                                          #
    #    Training And Evaluation (Ranking) Functions                                           #
    #                                                                                          #
    ############################################################################################

    def Crichton_Closed_Discovery_Train_And_Eval( self, model, epochs, batch_size, learning_rate, verbose, run_eval_number_epoch,
                                                  train_data_path, eval_data_path, model_save_path, embedding_path, gold_b_instance ):
        # Check(s)
        if Utils().Check_If_File_Exists( train_data_path ) == False:
            print( "Error: Specified Training File Does Not Exist" )
            return
        if Utils().Check_If_File_Exists( eval_data_path ) == False:
            print( "Error: Specified Evaluation File Does Not Exist" )
            return
        if embedding_path != "" and Utils().Check_If_File_Exists( embedding_path ) == False:
            print( "Error: Specified Embedding File Does Not Exist" )
            return

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
        gold_b_term              = gold_b_instance.split()[1]
        best_ranking             = sys.maxsize
        eval_data                = model.Get_Data_Loader().Read_Data( eval_data_path, keep_in_memory = False )

        # Check
        if eval_data == [] or len( eval_data ) == 0:
            print( "Error Loading Evaluation Data" )
            return

        print( "Preparing Evaluation Data" )

        model.Get_Data_Loader().Load_Embeddings( embedding_path )
        model.Get_Data_Loader().Generate_Token_IDs()

        # Check
        if model.Get_Data_Loader().Get_Number_Of_Embeddings() == 0:
            print( "Error Loading Embeddings Or No Embeddings Specified" )
            return

        # Vectorize Gold B Term And Entire Evaluation Data-set
        gold_b_input_1, gold_b_input_2, gold_b_input_3, _ = model.Vectorize_Model_Data( data_list = [gold_b_instance], model_type = "closed_discovery",
                                                                                        use_csr_format = True, is_crichton_format = True, keep_in_memory = False )
        eval_input_1, eval_input_2, eval_input_3, _       = model.Vectorize_Model_Data( data_list = eval_data, model_type = "closed_discovery",
                                                                                        use_csr_format = True, is_crichton_format = True, keep_in_memory = False )

        # Checks
        if gold_b_input_1 is None or gold_b_input_2 is None or gold_b_input_3 is None:
            print( "Error Occurred During Data Vectorization (Gold B)" )
            return
        if eval_input_1 is None or eval_input_2 is None or eval_input_3 is None:
            print( "Error Occurred During Data Vectorization (Evaluation Data)" )
            return

        model.Get_Data_Loader().Clear_Data()

        # Create Directory
        model.utils.Create_Path( model_save_path )

        print( "Beginning Model Data Preparation/Model Training" )

        # Set Correct Number Of Epochs Versus Number Of Epochs To Run Before Evaluation Is Performed
        epochs = epochs // run_eval_number_epoch

        for iteration in range( epochs ):
            # Train Model Over Data: "../data/train_cs1_closed_discovery_without_aggregators"
            model.Fit( train_data_path, epochs = run_eval_number_epoch, batch_size = batch_size, learning_rate = learning_rate, verbose = verbose )

            # Check
            if model.Get_Model() == None or model.Get_Model().model_history == None:
                print( "Error Model Contains No History / Model Failed To Train" )
                return

            history = model.Get_Model().model_history.history
            accuracy_score = history['accuracy'][-1] if 'accuracy' in history else history['acc'][-1]

            loss_per_epoch.append( history['loss'][-1] )
            accuracy_per_epoch.append( accuracy_score )
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
                a_term = instance_tokens[0] # Not Used
                b_term = instance_tokens[1]
                c_term = instance_tokens[2] # Not Used

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

        self.Generate_Plot( data_list = ranking_per_epoch, title = "Evaluation: Rank vs Epoch", x_label = "Epoch",
                            y_label = "Rank", file_name = "evaluation_rank_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = number_of_ties_per_epoch, title = "Evaluation: Ties vs Epoch", x_label = "Epoch",
                            y_label = "Ties", file_name = "evaluation_ties_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = loss_per_epoch, title = "Training: Loss vs Epoch", x_label = "Epoch",
                            y_label = "Loss", file_name = "training_loss_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = accuracy_per_epoch, title = "Training: Accuracy vs Epoch", x_label = "Epoch",
                            y_label = "Accuracy", file_name = "training_accuracy_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = precision_per_epoch, title = "Training: Precision vs Epoch", x_label = "Epoch",
                            y_label = "Precision", file_name = "training_precision_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = recall_per_epoch, title = "Training: Recall vs Epoch", x_label = "Epoch",
                            y_label = "Recall", file_name = "training_recall_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = f1_score_per_epoch, title = "Training: F1-Score vs Epoch", x_label = "Epoch",
                            y_label = "F1-Score", file_name = "training_f1_vs_epoch.png", save_path = model_save_path )

        print( "\nBest Rank: " + str( best_ranking ) )
        print( "Number Of Ties With Best Rank: " + str( best_number_of_ties ) + "\n" )

    def Closed_Discovery_Train_And_Eval( self, model, epochs, batch_size, learning_rate, verbose, run_eval_number_epoch,
                                         train_data_path, model_save_path, embedding_path, gold_b_instance ):
        # Check(s)
        if Utils().Check_If_File_Exists( train_data_path ) == False:
            print( "Error: Specified Training File Does Not Exist" )
            return
        if embedding_path != "" and Utils().Check_If_File_Exists( embedding_path ) == False:
            print( "Error: Specified Embedding File Does Not Exist" )
            return

        # Training/Evaluation Variables (Do Not Modify)
        ranking_per_epoch        = []
        ranking_per_epoch_value  = []
        number_of_ties_per_epoch = []
        loss_per_epoch           = []
        accuracy_per_epoch       = []
        precision_per_epoch      = []
        recall_per_epoch         = []
        f1_score_per_epoch       = []
        best_number_of_ties      = 0
        a_term                   = gold_b_instance.split( '\t' )[0]
        gold_b_term              = gold_b_instance.split( '\t' )[1]
        c_term                   = gold_b_instance.split( '\t' )[2]
        best_ranking             = sys.maxsize

        # Create Directory
        model.utils.Create_Path( model_save_path )

        print( "Beginning Model Data Preparation/Model Training" )

        # Set Correct Number Of Epochs Versus Number Of Epochs To Run Before Evaluation Is Performed
        epochs = epochs // run_eval_number_epoch

        for iteration in range( epochs ):
            # Train Model Over Data: "../data/train_cs1_closed_discovery_without_aggregators"
            model.Fit( train_data_path, epochs = run_eval_number_epoch, batch_size = batch_size, learning_rate = learning_rate, verbose = verbose )

            # Check
            if model.Get_Model() == None or model.Get_Model().model_history == None:
                print( "Error Model Contains No History / Model Failed To Train" )
                return

            history = model.Get_Model().model_history.history
            accuracy_score = history['accuracy'][-1] if 'accuracy' in history else history['acc'][-1]

            loss_per_epoch.append( history['loss'][-1] )
            accuracy_per_epoch.append( accuracy_score )
            precision_per_epoch.append( history['Precision'][-1] )
            recall_per_epoch.append( history['Recall'][-1] )
            f1_score_per_epoch.append( history['F1_Score'][-1] )

            # Perform Prediction Over The Entire Evaluation Data-set (Model Inference)
            predictions = model.Predict( a_term, c_term, return_raw_values = True )[0]

            # Fetch All Unique Terms From DataLoader Dictionary
            unique_token_list = model.Get_Data_Loader().Get_Token_ID_Dictionary().keys()

            ################################################################
            # Rank Unique Token Predictions Using Their Probability Values #
            ################################################################

            prob_dict = {}

            # For Each Prediction From The Model, Store The Prediction Value And Unique Concept Token Within A Dictionary
            for token, prediction in zip( unique_token_list, predictions ):
                # print( str( token ) + "\t:\t" + str( prediction ) )
                prob_dict[token] = prediction

            # Sort Concept And Probability Dictionary In Reverse Order To Rank Concepts.
            prob_dict  = { k: v for k, v in sorted( prob_dict.items(), key = lambda x: x[1], reverse = True ) }

            # Get Index Of Desired Gold B
            gold_b_rank  = list( prob_dict.keys() ).index( gold_b_term.lower() )
            gold_b_value = prob_dict[gold_b_term.lower()]

            # Get Number Of Ties With Gold B Prediction Value
            gold_b_ties  = list( prob_dict.values() ).count( gold_b_value ) - 1

            ranking_per_epoch.append( gold_b_rank )
            ranking_per_epoch_value.append( gold_b_value )
            number_of_ties_per_epoch.append( gold_b_ties )

            # Keep Track Of The Best Rank
            if gold_b_rank < best_ranking:
                best_ranking        = gold_b_rank
                best_number_of_ties = gold_b_ties

            print( "Epoch : " + str( iteration ) + " - Gold B: " + str( gold_b_term ) + " - Rank: " + str( gold_b_rank ) + \
                   " Of " + str( len( prob_dict ) ) + " Number Of B Terms" + " - Score: " + str( gold_b_value ) + \
                   " - Number Of Ties: " + str( gold_b_ties ) )

        # Print Ranking Information Per Epoch
        print( "" )

        for epoch in range( len( ranking_per_epoch ) ):
            print( "Epoch: " + str( epoch ) + " - Rank: " + str( ranking_per_epoch[epoch] ) +
                " - Value: " + str( ranking_per_epoch_value[epoch] ) + " - Number Of Ties: " + str( number_of_ties_per_epoch[epoch] ) )

        print( "\nGenerating Model Metric Charts" )
        if not re.search( r"\/$", model_save_path ): model_save_path += "/"

        self.Generate_Plot( data_list = ranking_per_epoch, title = "Evaluation: Rank vs Epoch", x_label = "Epoch",
                            y_label = "Rank", file_name = "evaluation_rank_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = number_of_ties_per_epoch, title = "Evaluation: Ties vs Epoch", x_label = "Epoch",
                            y_label = "Ties", file_name = "evaluation_ties_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = loss_per_epoch, title = "Training: Loss vs Epoch", x_label = "Epoch",
                            y_label = "Loss", file_name = "training_loss_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = accuracy_per_epoch, title = "Training: Accuracy vs Epoch", x_label = "Epoch",
                            y_label = "Accuracy", file_name = "training_accuracy_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = precision_per_epoch, title = "Training: Precision vs Epoch", x_label = "Epoch",
                            y_label = "Precision", file_name = "training_precision_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = recall_per_epoch, title = "Training: Recall vs Epoch", x_label = "Epoch",
                            y_label = "Recall", file_name = "training_recall_vs_epoch.png", save_path = model_save_path )

        self.Generate_Plot( data_list = f1_score_per_epoch, title = "Training: F1-Score vs Epoch", x_label = "Epoch",
                            y_label = "F1-Score", file_name = "training_f1_vs_epoch.png", save_path = model_save_path )

        print( "\nBest Rank: " + str( best_ranking ) )
        print( "Number Of Ties With Best Rank: " + str( best_number_of_ties ) + "\n" )


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

def Main():
    # Check(s)
    is_file_criteria_met = all([ os.path.exists( sys.argv[1] ), os.path.isdir( sys.argv[1] ) == False ])

    if len( sys.argv ) < 2:
        print( "Error: No JSON File Specified" )
        print( "    Example: 'python LBDDriver.py paramter_file.json'" )
        exit()
    elif not is_file_criteria_met:
        print( "Error: Specified File Does Not Exist" )
        print( "    File:", sys.argv[1] )
        exit()

    # Create LBD Driver Class Object
    driver = NNLBD_Driver()

    # Open JSON File (Command-line/Terminal Argument)
    driver.Read_JSON_Data( sys.argv[1] )
    driver.Extract_Global_Settings()

    # Get All Experiments In JSON Data File
    #  There Can Be More Than One
    #driver.Print_All_JSON_Data()

    # Run Experiments
    driver.Run_Experiments()

    # Clean-Up
    driver = None

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()