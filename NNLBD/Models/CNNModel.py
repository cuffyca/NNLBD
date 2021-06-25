#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    12/08/2020                                                                   #
#    Revised: 06/02/2021                                                                   #
#                                                                                          #
#    Generates A Neural Network Used For LBD, Trains Using Data In Format Below.           #
#                                                                                          #
#    Expected Format:                                                                      #
#         C001	TREATS	C002	C004	C009                                               #
#         C001	ISA	    C003                                                               #
#         C002	TREATS	C005	C006	C010                                               #
#         ...                                                                              #
#         C004	ISA	    C010                                                               #
#         C005	TREATS	C003	C004	C006                                               #
#         C005	AFFECTS	C001	C009	C010                                               #
#                                                                                          #
#    How To Run:                                                                           #
#        See 'main.py' For Running Example. Execute Via:                                   #
#                    "python main.py"                                                      #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Suppress Warnings/FutureWarnings
import warnings
warnings.filterwarnings( 'ignore' )
#warnings.simplefilter( action = 'ignore', category = Warning )
#warnings.simplefilter( action = 'ignore', category = FutureWarning )   # Also Works For Future Warnings

# Standard Modules
import os, re

# Suppress Tensorflow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes Tensorflow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow as tf
#tf.logging.set_verbosity( tf.logging.ERROR )                       # Tensorflow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # Tensorflow v1.x

import numpy as np
from tensorflow import keras

# Tensorflow v2.x Support
if re.search( r'2.\d+', tf.__version__ ):
    import tensorflow.keras.backend as K
    from tensorflow.keras import optimizers
    from tensorflow.keras import regularizers
    # from keras.callbacks import ModelCheckpoint
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation, Average, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, Lambda, MaxPooling1D, Multiply
# Tensorflow v1.15.x Support
else:
    import keras.backend as K
    from keras import optimizers
    from keras import regularizers
    # from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.layers import Activation, Average, BatchNormalization, Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, Lambda, MaxPooling1D, Multiply

# Custom Modules
from NNLBD.Models           import BaseModel
from NNLBD.Models.BaseModel import Model_Saving_Callback



############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class CNNModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "cnn", model_type = "open_discovery",
                  optimizer = 'adam', activation_function = 'sigmoid', loss_function = "binary_crossentropy", number_of_embedding_dimensions = 200,
                  learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.8, batch_size = 32, prediction_threshold = 0.5, shuffle = True,
                  use_csr_format = True, per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0", verbose = 2,
                  debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False,
                  early_stopping_metric_monitor = "loss", early_stopping_persistence = 3, use_batch_normalization = False,
                  trainable_weights = False, embedding_path = "", final_layer_type = "dense", feature_scale_value = 1.0, learning_rate_decay = 0.004 ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, network_model = network_model, model_type = model_type,
                          optimizer = optimizer, activation_function = activation_function, loss_function = loss_function,  learning_rate = learning_rate,
                          batch_size = batch_size, use_csr_format = use_csr_format, prediction_threshold = prediction_threshold, shuffle = shuffle,
                          number_of_embedding_dimensions = number_of_embedding_dimensions, momentum = momentum, epochs = epochs, per_epoch_saving = per_epoch_saving,
                          device_name = device_name, verbose = verbose, debug_log_file_handle = debug_log_file_handle, dropout = dropout,
                          enable_tensorboard_logs = enable_tensorboard_logs, enable_early_stopping = enable_early_stopping, use_gpu = use_gpu,
                          early_stopping_metric_monitor = early_stopping_metric_monitor, early_stopping_persistence = early_stopping_persistence,
                          use_batch_normalization = use_batch_normalization, trainable_weights = trainable_weights, embedding_path = embedding_path,
                          final_layer_type = final_layer_type, feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay )
        self.version       = 0.10
        self.network_model = "cnn"   # Force Setting Model To 'CNN' Model.


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

    """
        Converts Randomized Batches Of Model Inputs & Outputs From CSR_Matrix Format
          To Numpy Arrays For Model Training

        Inputs:
            X              : Model Concept Inputs (CSR_Matrix)
            Y              : Model Concept Outputs (CSR_Matrix)
            batch_size     : Batch Size (Integer)
            steps_per_batch: Number Of Iterations Per Epoch (Integer)
            shuffle        : Shuffles Data Prior To Conversion (Boolean)

        Outputs:
            X_batch        : Numpy 2D Matrix Of Model Concept Inputs (Numpy Array)
            Y_batch        : Numpy 2D Matrix Of Model Concept Outputs (Numpy Array)

            Modification Of Code From Source: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
    """
    def Batch_Generator( self, X, Y, batch_size, steps_per_batch, shuffle ):
        number_of_instances = X.shape[0]      # Should Be The Same As 'self.trained_instances'
        counter             = 0
        sample_index        = np.arange( number_of_instances )

        if shuffle:
            np.random.shuffle( sample_index )

        while True:
            start_index = batch_size * counter
            end_index   = batch_size * ( counter + 1 )

            # Check - Fixes Batch_Generator Training Errors With The Number Of Instances % Batch Sizes != 0
            end_index   = number_of_instances if end_index > number_of_instances else end_index

            batch_index = sample_index[start_index:end_index]
            X_batch     = X[batch_index,:]
            Y_batch     = Y[batch_index,:].todense()
            counter     += 1

            yield X_batch, Y_batch

            # Reset The Batch Index After Final Batch Has Been Reached
            if counter == steps_per_batch:
                if shuffle:
                    np.random.shuffle( sample_index )
                counter = 0

    """
        Trains Model Using Training Data, Fits Model To Data

        Inputs:
            training_file_path          : Evaluation File Path (String)
            number_of_hidden_dimensions : Number Of Hidden Dimensions When Building Neural Network Architecture (Integer)
            learning_rate               : Learning Rate (Float)
            epochs                      : Number Of Training Epochs (Integer)
            batch_size                  : Size Of Each Training Batch (Integer)
            momentum                    : Momentum Value (Float)
            dropout                     : Dropout Value (Float)
            verbose                     : Sets Training Verbosity - Options: 0 = Silent, 1 = Progress Bar, 2 = One Line Per Epoch (Integer)
            per_epoch_saving            : Toggle To Save Model After Each Training Epoch (Boolean: True, False)
            use_csr_format              : Toggle To Use Compressed Sparse Row (CSR) Formatted Matrices For Storing Training/Evaluation Data (Boolean: True, False)

        Outputs:
            None
    """
    def Fit( self, train_input_1 = None, train_input_2 = None, train_input_3 = None, train_outputs = None,
             epochs = 30, batch_size = 32, momentum = 0.05, dropout = 0.01, verbose = 1, shuffle = True,
             use_csr_format = True, per_epoch_saving = False ):
        # Update 'BaseModel' Class Variables
        if epochs           != 30:    self.Set_Epochs( epochs )
        if batch_size       != 32:    self.Set_Batch_Size( batch_size )
        if momentum         != 0.05:  self.Set_Momentum( momentum )
        if dropout          != 0.01:  self.Set_Dropout( dropout )
        if verbose          != 1:     self.Set_Verbose( verbose )
        if shuffle          != True:  self.Set_Shuffle( shuffle )
        if use_csr_format   != True:  self.Set_Use_CSR_Format( use_csr_format )
        if per_epoch_saving != False: self.Set_Per_Epoch_Saving( per_epoch_saving )

        self.Print_Log( "Error: CNN Model Is Not Implemented / Exiting Program", force_print = True )
        # raise( NotImplementedError )
        exit()

        if self.use_csr_format:
            self.trained_instances           = train_input_1.shape[0]
            number_of_train_output_instances = train_outputs.shape[0]
        else:
            self.trained_instances           = len( train_input_1 )
            number_of_train_output_instances = len( train_outputs )

        self.Print_Log( "CNNModel::Fit() - Model Training Settings" )
        self.Print_Log( "                - Epochs             : " + str( self.epochs             ) )
        self.Print_Log( "                - Batch Size         : " + str( self.batch_size         ) )
        self.Print_Log( "                - Verbose            : " + str( self.verbose            ) )
        self.Print_Log( "                - Shuffle            : " + str( self.shuffle            ) )
        self.Print_Log( "                - Use CSR Format     : " + str( self.use_csr_format     ) )
        self.Print_Log( "                - Per Epoch Saving   : " + str( self.per_epoch_saving   ) )
        self.Print_Log( "                - No. of Train Inputs: " + str( self.trained_instances  ) )

        # Compute Number Of Steps Per Batch (Use CSR Format == True)
        steps_per_batch = 0

        if self.batch_size >= self.trained_instances:
            steps_per_batch = 1
        else:
            steps_per_batch = self.trained_instances // self.batch_size if self.trained_instances % self.batch_size == 0 else self.trained_instances // self.batch_size + 1

        # Setup Saving The Model After Each Epoch
        if self.per_epoch_saving:
            self.Print_Log( "                - Adding Model Saving Callback" )
            self.callback_list.append( Model_Saving_Callback() )

        # Perform Model Training
        self.Print_Log( "CNNModel::Fit() - Executing Model Training", force_print = True )

        with tf.device( self.device_name ):
            if self.use_csr_format:
                self.model_history = self.model.fit_generator( generator = self.Batch_Generator( train_input_1, train_outputs, batch_size = self.batch_size, steps_per_batch = steps_per_batch, shuffle = self.shuffle ),
                                                               epochs = self.epochs, steps_per_epoch = steps_per_batch, verbose = self.verbose, callbacks = self.callback_list )
            else:
                self.model_history = self.model.fit( train_input_1, train_outputs, shuffle = self.shuffle, batch_size = self.batch_size,
                                                     epochs = self.epochs, verbose = self.verbose, callbacks = self.callback_list )

        # Print Last Epoch Metrics
        if self.verbose == False:
            final_epoch = self.model_history.epoch[-1]
            history     = self.model_history.history
            self.Print_Log( "", force_print = True )
            self.Print_Log( "CNNModel::Final Training Metric(s) At Epoch: " + str( final_epoch ), force_print = True )

            # Iterate Through Available Metrics And Print Their Formatted Values
            for metric in history.keys():
                self.Print_Log( "CNNModel::  - " + str( metric.capitalize() ) + ":\t{:.4f}" . format( history[metric][-1] ), force_print = True )

        self.Print_Log( "CNNModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "CNNModel::Fit() - Complete" )

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input_vector   : Vectorized Primary Model Input (Numpy Array)
            secondary_input_vector : Vectorized Secondary Model Input (Numpy Array)

        Outputs:
            prediction              : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, inputs ):
        self.Print_Log( "CNNModel::Predict() - Predicting Using Inputs: " + str( inputs ) )

        with tf.device( self.device_name ):
            return self.model.predict( [inputs] )

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs_1, inputs_2, inputs_3, outputs, verbose ):
        self.Print_Log( "CNNModel::Evaluate() - Executing Model Evaluation" )

        inputs = np.hstack( ( inputs_1, inputs_2 ) )

        with tf.device( self.device_name ):
            loss, accuracy, precision, recall, f1_score = self.model.evaluate( inputs, outputs, verbose = verbose )

            self.Print_Log( "CNNModel::Evaluate() - Complete" )

            return loss, accuracy, precision, recall, f1_score

    ############################################################################################
    #                                                                                          #
    #    Keras Model(s)                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Build The Keras Model

        Inputs:
            number_of_train_1_inputs    : (Integer)
            number_of_train_2_inputs    : (Integer)
            number_of_hidden_dimensions : (Integer)
            number_of_outputs           : (Integer)

        Outputs:
            None
    """
    def Build_Model( self, number_of_features, number_of_embedding_dimensions, number_of_outputs, embeddings = [] ):
        # Update 'BaseModel' Class Variables
        self.number_of_features             = number_of_features             if number_of_features             != self.number_of_features             else self.number_of_features
        self.number_of_embedding_dimensions = number_of_embedding_dimensions if number_of_embedding_dimensions != self.number_of_embedding_dimensions else self.number_of_embedding_dimensions
        self.number_of_outputs              = number_of_outputs              if number_of_outputs              != self.number_of_outputs              else self.number_of_outputs

        # Change Me
        self.Print_Log( "CNNModel::Build_Model() - Model Settings" )
        self.Print_Log( "                           - Network Model              : " + str( self.network_model                  ) )
        self.Print_Log( "                           - Learning Rate              : " + str( self.learning_rate                  ) )
        self.Print_Log( "                           - Dropout                    : " + str( self.dropout                        ) )
        self.Print_Log( "                           - Momentum                   : " + str( self.momentum                       ) )
        self.Print_Log( "                           - Optimizer                  : " + str( self.optimizer                      ) )
        self.Print_Log( "                           - Activation Function        : " + str( self.activation_function            ) )
        self.Print_Log( "                           - No. of Features            : " + str( self.number_of_features             ) )
        self.Print_Log( "                           - No. of Embedding Dimensions: " + str( self.number_of_embedding_dimensions ) )
        self.Print_Log( "                           - No. of Outputs             : " + str( self.number_of_outputs              ) )
        self.Print_Log( "                           - Trainable Weights          : " + str( self.trainable_weights              ) )
        self.Print_Log( "                           - Feature Scaling Value      : " + str( self.feature_scale_value            ) )

        #######################
        #                     #
        #  Build Keras Model  #
        #                     #
        #######################

        # Batch Normalization Model
        if self.use_batch_normalization:
            input_layer         = Input( shape = ( number_of_features, ), name = "Input_Layer" )
            embedding_layer     = Embedding( number_of_features, number_of_embedding_dimensions, name = 'Embedding_Layer', weights = [embeddings], trainable = self.trainable_weights )( input_layer )

            # Perform Feature Scaling Prior To Generating An Embedding Representation
            if self.feature_scale_value != 1.0:
                feature_scale_value = self.feature_scale_value  # Fixes Python Recursion Limit Error (Model Tries To Save All 'self' Variable When Used With Lambda Function)
                embedding_layer = Lambda( lambda x: x * feature_scale_value )( embedding_layer )

            convolution_layer_1 = Conv1D( filters = 24, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_1" )( embedding_layer )
            batch_norm_layer_1  = BatchNormalization( name = "Batch_Norm_Layer_1" )( convolution_layer_1 )
            pooling_layer_1     = MaxPooling1D( pool_size = 2, name = "Pooling_Layer_1" )( batch_norm_layer_1 )
            convolution_layer_2 = Conv1D( filters = 12, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_2" )( pooling_layer_1 )
            batch_norm_layer_2  = BatchNormalization( name = "Batch_Norm_Layer_2" )( convolution_layer_2 )
            pooling_layer_2     = MaxPooling1D( pool_size = 2, name = "Pooling_Layer_2" )( batch_norm_layer_2 )
            convolution_layer_3 = Conv1D( filters = 6, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_3" )( pooling_layer_2 )
            dropout_layer       = Dropout( name = "Dropout_Layer", rate = self.dropout )( convolution_layer_3 )
            flatten_layer       = Flatten( name = "Flatten_Layer" )( dropout_layer )
            dense_layer_1       = Dense( units = 12 * 3, activation = 'relu', name = 'Dense_Layer' )( flatten_layer )
            batch_norm_layer_8  = BatchNormalization( name = "Batch_Norm_Layer_8" )( dense_layer_1 )
            output_layer        = Dense( units = number_of_outputs, activation = self.activation_function, name = 'Localist_Output_Representation' )( batch_norm_layer_8 )

        # Traditional Model Without Batch Normalization Using Functional API
        else:
            input_layer         = Input( shape = ( number_of_features, ), name = "Input_Layer" )
            embedding_layer     = Embedding( number_of_features, number_of_embedding_dimensions, name = 'Embedding_Layer', weights = [embeddings], trainable = self.trainable_weights )( input_layer )

            # Perform Feature Scaling Prior To Generating An Embedding Representation
            if self.feature_scale_value != 1.0:
                embedding_layer = Lambda( lambda x: x * self.feature_scale_value )( embedding_layer )

            convolution_layer_1 = Conv1D( filters = 24, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_1" )( embedding_layer )
            pooling_layer_1     = MaxPooling1D( pool_size = 2, name = "Pooling_Layer_1" )( convolution_layer_1 )
            convolution_layer_2 = Conv1D( filters = 12, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_2" )( pooling_layer_1 )
            pooling_layer_2     = MaxPooling1D( pool_size = 2, name = "Pooling_Layer_2" )( convolution_layer_2 )
            convolution_layer_3 = Conv1D( filters = 6, kernel_size = 2, activation = 'relu', name = "Convolution_Layer_3" )( pooling_layer_2 )
            dropout_layer       = Dropout( name = "Dropout_Layer", rate = self.dropout )( convolution_layer_3 )
            flatten_layer       = Flatten( name = "Flatten_Layer" )( dropout_layer )
            dense_layer_1       = Dense( units = 12 * 3, activation = 'relu', name = 'Dense_Layer' )( flatten_layer )
            output_layer        = Dense( units = number_of_outputs, activation = self.activation_function, name = 'Localist_Output_Representation' )( dense_layer_1 )

        self.model              = Model( inputs = input_layer, outputs = output_layer, name = self.network_model + "_model" )

        if self.optimizer == "adam":
            adam_opt = optimizers.Adam( lr = self.learning_rate )
            self.model.compile( loss = self.loss_function, optimizer = adam_opt, metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )
        elif self.optimizer == "sgd":
            sgd = optimizers.SGD( lr = self.learning_rate, momentum = self.momentum )
            self.model.compile( loss = self.loss_function, optimizer = sgd, metrics = [ 'accuracy', super().Precision, super().Recall, super().F1_Score ] )

        # Print Model Summary
        self.Print_Log( "CNNModel::Build_Model() - =========================================================" )
        self.Print_Log( "CNNModel::Build_Model() - =                     Model Summary                     =" )
        self.Print_Log( "CNNModel::Build_Model() - =========================================================" )

        self.model.summary( print_fn = lambda x:  self.Print_Log( "CNNModel::Build_Model() - " + str( x ) ) )      # Capture Model.Summary()'s Print Output As A Function And Store In Variable 'x'

        self.Print_Log( "CNNModel::Build_Model() - =========================================================" )
        self.Print_Log( "CNNModel::Build_Model() - =                                                       =" )
        self.Print_Log( "CNNModel::Build_Model() - =========================================================" )


    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    exit()
