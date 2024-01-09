#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    07/05/2023                                                                   #
#    Revised: 12/03/2023                                                                   #
#                                                                                          #
#    Generates A BERT-Based Neural Network Used For LBD.                                   #
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

# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Removes TensorFlow GPU CUDA Checking Error/Warning Messages
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

# Custom Modules
from NNLBD.Models.Base import BaseModel
from NNLBD.Layers      import Embedding_Extraction_Layer
from NNLBD.Misc        import Utils

if not Utils().Check_For_Installed_Modules( ["numpy", "tensorflow", "transformers"] ):
    print( "BERTModel - Error: Required Modules Not Installed" )
    exit()

# Downloaded Module Imports
import tensorflow as tf
#tf.logging.set_verbosity( tf.logging.ERROR )                       # TensorFlow v2.x
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )    # TensorFlow v1.x

import numpy as np
from tensorflow import keras

# TensorFlow v1.15.x Support
if re.search( r'1.\d+', tf.__version__ ):
    print( "BERTModel::Init() - Error: BERT Model Does Not Support Tensorflow v1.x" )
    exit()

# TensorFlow v2.x Support
import tensorflow.keras.backend as K
from tensorflow.keras         import optimizers
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.losses  import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.layers  import Average, Concatenate, Dense, Dropout, Lambda, Multiply, Reshape

import transformers
transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
from transformers import BertConfig, TFBertModel

# Sets Random Seed For Ranking Reproducibility
# np.random.seed( 0 )
# tf.random.set_seed( 0 )
# tf.compat.v1.set_random_seed( 0 )


############################################################################################
#                                                                                          #
#    Keras Model Class                                                                     #
#                                                                                          #
############################################################################################

class BERTModel( BaseModel ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, network_model = "bert", model_type = "open_discovery",
                  optimizer = 'adam', activation_function = 'sigmoid', loss_function = "binary_crossentopy", number_of_hidden_dimensions = 200,
                  number_of_embedding_dimensions = 200, learning_rate = 0.005, epochs = 30, momentum = 0.05, dropout = 0.5, batch_size = 32,
                  prediction_threshold = 0.5, shuffle = True, use_csr_format = True, per_epoch_saving = False, use_gpu = True, device_name = "/gpu:0",
                  verbose = 2, debug_log_file_handle = None, enable_tensorboard_logs = False, enable_early_stopping = False, early_stopping_metric_monitor = "loss",
                  early_stopping_persistence = 3, use_batch_normalization = False, trainable_weights = False, embedding_path = "", final_layer_type = "dense",
                  feature_scale_value = 1.0, learning_rate_decay = 0.004, model_path = "bert-base-cased", embedding_modification = "concatenate",
                  weight_decay = 0.0001, margin = 30.0, scale = 0.35, use_cosine_annealing = False, cosine_annealing_min = 1e-6, cosine_annealing_max = 2e-4,
                  bilstm_dimension_size = 64, bilstm_merge_mode = "concat", skip_gpu_init = False ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, network_model = network_model, model_type = model_type,
                          batch_size = batch_size, prediction_threshold = prediction_threshold, shuffle = shuffle, use_csr_format = use_csr_format,
                          optimizer = optimizer, activation_function = activation_function, loss_function = loss_function, momentum = momentum,
                          number_of_hidden_dimensions = number_of_hidden_dimensions, number_of_embedding_dimensions = number_of_embedding_dimensions,
                          learning_rate = learning_rate, epochs = epochs, per_epoch_saving = per_epoch_saving, use_gpu = use_gpu, device_name = device_name,
                          verbose = verbose, debug_log_file_handle = debug_log_file_handle, dropout = dropout, enable_tensorboard_logs = enable_tensorboard_logs,
                          enable_early_stopping = enable_early_stopping, early_stopping_metric_monitor = early_stopping_metric_monitor,
                          early_stopping_persistence = early_stopping_persistence, use_batch_normalization = use_batch_normalization,
                          trainable_weights = trainable_weights, embedding_path = embedding_path, final_layer_type = final_layer_type,
                          feature_scale_value = feature_scale_value, learning_rate_decay = learning_rate_decay, model_path = model_path, weight_decay = weight_decay,
                          embedding_modification = embedding_modification, margin = margin, scale = scale, use_cosine_annealing = use_cosine_annealing,
                          cosine_annealing_min = cosine_annealing_min, cosine_annealing_max = cosine_annealing_max, bilstm_dimension_size = bilstm_dimension_size,
                          bilstm_merge_mode = bilstm_merge_mode, skip_gpu_init = skip_gpu_init )
        self.version = 0.01

        # Check(s) - Set Default Parameters If Not Specified
        self.network_model = "bert"
        self.bert_config   = None
        self.model_history = type( 'obj', ( object, ), { 'epoch' : [], 'history' : {} } )    # Create Object To Simulate Keras Model Instance History Object


    ############################################################################################
    #                                                                                          #
    #    Keras Model Functions                                                                 #
    #                                                                                          #
    ############################################################################################

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
             epochs = None, batch_size = None, momentum = None, dropout = None, verbose = None, shuffle = None,
             use_csr_format = None, per_epoch_saving = None, use_cosine_annealing = None, cosine_annealing_min = None,
             cosine_annealing_max = None ):
        # Update 'BaseModel' Class Variables
        if epochs               is not None: self.Set_Epochs( epochs )
        if batch_size           is not None: self.Set_Batch_Size( batch_size )
        if momentum             is not None: self.Set_Momentum( momentum )
        if dropout              is not None: self.Set_Dropout( dropout )
        if verbose              is not None: self.Set_Verbose( verbose )
        if shuffle              is not None: self.Set_Shuffle( shuffle )
        if use_csr_format       is not None: self.Set_Use_CSR_Format( use_csr_format )
        if per_epoch_saving     is not None: self.Set_Per_Epoch_Saving( per_epoch_saving )
        if use_cosine_annealing is not None: self.Set_Use_Cosine_Annealing( use_cosine_annealing )
        if cosine_annealing_min is not None: self.Set_Cosine_Annealing_Min( cosine_annealing_min )
        if cosine_annealing_max is not None: self.Set_Cosine_Annealing_Max( cosine_annealing_max )

        # Add Model Callback Functions
        super().Add_Enabled_Model_Callbacks()

        primary_input_ids, primary_attention_masks, primary_token_type_ids       = train_input_1
        secondary_input_ids, secondary_attention_masks, secondary_token_type_ids = train_input_2
        tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids    = train_input_3

        self.trained_instances = primary_input_ids.shape[0]

        # Create Tensorflow Dataset Of Input/Output Tensors
        train_dataset = tf.data.Dataset.from_tensor_slices( ( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                                              secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                                              tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids,
                                                              train_outputs ) )

        # Shuffle Dataset If Specified. This Also Reshuffles After Every Epoch/Iteration
        if self.shuffle:
            train_dataset = train_dataset.shuffle( buffer_size = self.trained_instances, reshuffle_each_iteration = True )

        # Set Dataset To Use Batches Per Epoch/Iteration
        train_dataset = train_dataset.batch( self.batch_size )

        self.Print_Log( "BERTModel::Fit() - Model Training Settings" )
        self.Print_Log( "                 - Epochs             : " + str( self.epochs            ) )
        self.Print_Log( "                 - Batch Size         : " + str( self.batch_size        ) )
        self.Print_Log( "                 - Verbose            : " + str( self.verbose           ) )
        self.Print_Log( "                 - Shuffle            : " + str( self.shuffle           ) )
        self.Print_Log( "                 - Use CSR Format     : " + str( self.use_csr_format    ) )
        self.Print_Log( "                 - Per Epoch Saving   : " + str( self.per_epoch_saving  ) )
        self.Print_Log( "                 - No. of Train Inputs: " + str( self.trained_instances ) )

        # Perform Model Training
        self.Print_Log( "BERTModel::Fit() - Executing Model Training", force_print = True )

        # Create Optimizer
        if self.optimizer == "adam":
            optimizer = optimizers.Adam( lr = self.learning_rate )
        elif self.optimizer == "adamw":
            optimizer = optimizers.AdamW( lr = self.learning_rate )
        elif self.optimizer == "sgd":
            optimizer = optimizers.SGD( lr = self.learning_rate, momentum = self.momentum )

        # Create Loss Criterion
        if self.loss_function == "binary_crossentropy":
            loss_criterion = BinaryCrossentropy( from_logits = True )
            train_accuracy = BinaryAccuracy( name = 'train_accuracy' )
        elif self.loss_function == "categorical_crossentropy":
            loss_criterion = CategoricalCrossentropy( from_logits = True )
            train_accuracy = CategoricalAccuracy( name = 'train_accuracy' )
        elif self.loss_function == "sparse_categorical_crossentropy":
            loss_criterion = SparseCategoricalCrossentropy( from_logits = True )
            train_accuracy = SparseCategoricalAccuracy( name = 'train_accuracy' )

        # Model Metrics
        train_loss = tf.keras.metrics.Mean( name = "Training_Loss" )
        precision  = tf.keras.metrics.Mean( name = "Precision" )
        recall     = tf.keras.metrics.Mean( name = "Recall" )
        f1_score   = tf.keras.metrics.Mean( name = "F1_Score" )

        # Train The Model
        with tf.device( self.device_name ):
            for epoch in range( 1, self.epochs + 1 ):
                # Reset Metrics
                train_loss.reset_states()
                train_accuracy.reset_states()
                precision.reset_states()
                recall.reset_states()
                f1_score.reset_states()

                for idx, ( primary_input_ids, primary_attention_masks, primary_token_type_ids, \
                           secondary_input_ids, secondary_attention_masks, secondary_token_type_ids, \
                           tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids, labels ) in enumerate( train_dataset ):

                    with tf.GradientTape() as tape:
                        # 'training = True' Is Only Needed If There Are Layers With Different
                        #   Behavior During Training Versus Inference (e.g. Dropout).
                        predictions = self.model( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                                  secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                                  tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids, training = True )

                        loss = loss_criterion( labels, predictions )
                    gradients = tape.gradient( loss, self.model.trainable_variables )
                    optimizer.apply_gradients( zip( gradients, self.model.trainable_variables ) )

                    train_loss( loss )
                    train_accuracy( labels, predictions )

                    # Compute Precision, Recall & F1-Score
                    precision_val, recall_val, f1_score_val = self.Compute_Metrics( y_pred = predictions, y_true = labels )

                    precision( precision_val )
                    recall( recall_val )
                    f1_score( f1_score_val )

            print(
                f'Epoch: { epoch } - '
                f'Loss: { train_loss.result() } - '
                f'Accuracy: { train_accuracy.result() } - '
                f'Precision: { precision.result() } - '
                f'Recall: { recall.result() } - '
                f'F1_Score: { f1_score.result() } '
            )

            # Store Training Metrics
            if "loss" not in self.model_history.history:
                self.model_history.epoch.append( epoch - 1 )
                self.model_history.history['loss']      = [train_loss.result().numpy()]
                self.model_history.history['accuracy']  = [train_accuracy.result().numpy()]
                self.model_history.history['Precision'] = [precision.result().numpy()]
                self.model_history.history['Recall']    = [recall.result().numpy()]
                self.model_history.history['F1_Score']  = [f1_score.result().numpy()]
            else:
                self.model_history.epoch.append( epoch - 1 )
                self.model_history.history['loss'].append( train_loss.result().numpy() )
                self.model_history.history['accuracy'].append( train_accuracy.result().numpy() )
                self.model_history.history['Precision'].append( precision.result().numpy() )
                self.model_history.history['Recall'].append( recall.result().numpy() )
                self.model_history.history['F1_Score'].append( f1_score.result().numpy() )

        self.Print_Log( "BERTModel::Fit() - Finished Model Training", force_print = True )
        self.Print_Log( "BERTModel::Fit() - Complete" )

    """
        Outputs Model's Prediction Vector Given Inputs

        Inputs:
            primary_input_vector   : Vectorized Primary Model Input (Numpy Array)
            secondary_input_vector : Vectorized Secondary Model Input (Numpy Array)

        Outputs:
            prediction             : Vectorized Model Prediction or String Of Predicted Tokens Obtained From Prediction Vector (Numpy Array or String)
    """
    def Predict( self, primary_input_vector, secondary_input_vector, tertiary_input_vector ):
        self.Print_Log( "BERTModel::Predict() - Predicting Using Input #1: " + str( primary_input_vector ) )
        self.Print_Log( "BERTModel::Predict() - Predicting Using Input #2: " + str( secondary_input_vector ) )
        self.Print_Log( "BERTModel::Predict() - Predicting Using Input #3: " + str( tertiary_input_vector ) )

        primary_input_ids, primary_attention_masks, primary_token_type_ids       = primary_input_vector
        secondary_input_ids, secondary_attention_masks, secondary_token_type_ids = secondary_input_vector
        tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids    = tertiary_input_vector

        dataset = tf.data.Dataset.from_tensor_slices( ( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                                        secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                                        tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids ) ).batch( self.batch_size )

        with tf.device( self.device_name ):
            predictions = []

            for idx, ( primary_input_ids, primary_attention_masks, primary_token_type_ids, \
                       secondary_input_ids, secondary_attention_masks, secondary_token_type_ids, \
                       tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids ) in enumerate( dataset ):
                # 'training = False' Is Only Needed If There Are Layers With Different
                #   Behavior During Training Versus Inference (e.g. Dropout).
                model_pred = self.model( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                         secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                         tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids, training = False )

                predictions.extend( model_pred.numpy() )

            predictions = np.asarray( predictions )
            return predictions

    """
        Evaluates Model's Ability To Predict Evaluation Data

        Inputs:
            test_file_path : Evaluation File Path (String)

        Outputs:
            Metrics        : Loss, Accuracy, Precision, Recall & F1-Score
    """
    def Evaluate( self, inputs_1, inputs_2, inputs_3, outputs ):
        self.Print_Log( "BERTModel::Evaluate() - Executing Model Evaluation" )

        primary_input_ids, primary_attention_masks, primary_token_type_ids       = inputs_1
        secondary_input_ids, secondary_attention_masks, secondary_token_type_ids = inputs_2
        tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids    = inputs_3

        eval_dataset = tf.data.Dataset.from_tensor_slices( ( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                                             secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                                             tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids,
                                                             outputs ) ).batch( self.batch_size )

        # Create Loss Criterion
        if self.loss_function == "binary_crossentropy":
            loss_criterion = BinaryCrossentropy( from_logits = True )
            test_accuracy  = BinaryAccuracy( name = 'test_accuracy' )
        elif self.loss_function == "categorical_crossentropy":
            loss_criterion = CategoricalCrossentropy( from_logits = True )
            test_accuracy  = CategoricalAccuracy( name = 'test_accuracy' )
        elif self.loss_function == "sparse_categorical_crossentropy":
            loss_criterion = SparseCategoricalCrossentropy( from_logits = True )
            test_accuracy  = SparseCategoricalAccuracy( name = 'test_accuracy' )

        test_loss = tf.keras.metrics.Mean( name = 'test_loss' )

        with tf.device( self.device_name ):
            for idx, ( primary_input_ids, primary_attention_masks, primary_token_type_ids, \
                       secondary_input_ids, secondary_attention_masks, secondary_token_type_ids, \
                       tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids, labels ) in enumerate( eval_dataset ):
                # 'training = False' Is Only Needed If There Are Layers With Different
                #   Behavior During Training Versus Inference (e.g. Dropout).
                predictions = self.model( primary_input_ids, primary_attention_masks, primary_token_type_ids,
                                          secondary_input_ids, secondary_attention_masks, secondary_token_type_ids,
                                          tertiary_input_ids, tertiary_attention_masks, tertiary_token_type_ids, training = False )

                loss = loss_criterion( labels, predictions )
                test_loss( loss )
                test_accuracy( labels, predictions )

            self.Print_Log( "BERTModel::Evaluate() - Complete" )

            return loss, test_accuracy.result().numpy(), None, None, None

    """
        Prints A Model Depiction In PNG Format

        NOTE: Ensure pydot and graphviz are installed before calling this function.
    """
    def Plot_Model( self, with_shapes = False ):
        self.Print_Log( "BERTModel::Plot_Model() - Warning: This Function Call Is Only Compatible With Keras Model Instances", force_print = True )
        self.Print_Log( "BERTModel::Plot_Model() -          This Model Is Coded Using Tensorflow API", force_print = True )
        return


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
    def Build_Model( self, number_of_train_1_inputs = 0, number_of_train_2_inputs = 0, number_of_hidden_dimensions = 0, number_of_outputs = 1,
                     number_of_primary_embeddings = 0, number_of_secondary_embeddings = 0, primary_embeddings = [], secondary_embeddings = [],
                     embedding_modification = "concatenate", final_layer_type = None, weight_decay = None, max_sequence_length = 0 ):
        # Update 'BaseModel' Class Variables
        if number_of_train_1_inputs       != self.number_of_primary_inputs       : self.number_of_primary_inputs       = number_of_train_1_inputs
        if number_of_train_2_inputs       != self.number_of_secondary_inputs     : self.number_of_secondary_inputs     = number_of_train_2_inputs
        if number_of_hidden_dimensions    != self.number_of_hidden_dimensions    : self.number_of_hidden_dimensions    = number_of_hidden_dimensions
        if number_of_outputs              != self.number_of_outputs              : self.number_of_outputs              = number_of_outputs
        if embedding_modification is not None: self.embedding_modification = embedding_modification
        if final_layer_type       is not None: self.final_layer_type       = final_layer_type
        if weight_decay           is not None: self.weight_decay           = weight_decay

        self.Print_Log( "BERTModel::Build_Model() - Model Settings" )
        self.Print_Log( "                         - Network Model             : " + str( self.network_model                  ) )
        self.Print_Log( "                         - Embedding Modification    : " + str( self.embedding_modification         ) )
        self.Print_Log( "                         - Final Layer Type          : " + str( self.final_layer_type               ) )
        self.Print_Log( "                         - Learning Rate             : " + str( self.learning_rate                  ) )
        self.Print_Log( "                         - Dropout                   : " + str( self.dropout                        ) )
        self.Print_Log( "                         - Momentum                  : " + str( self.momentum                       ) )
        self.Print_Log( "                         - Optimizer                 : " + str( self.optimizer                      ) )
        self.Print_Log( "                         - Activation Function       : " + str( self.activation_function            ) )
        self.Print_Log( "                         - Weight Decay              : " + str( self.weight_decay                   ) )
        self.Print_Log( "                         - # Of Primary Embeddings   : " + str( number_of_primary_embeddings        ) )
        self.Print_Log( "                         - # Of Secondary Embeddings : " + str( number_of_secondary_embeddings      ) )
        self.Print_Log( "                         - No. of Primary Inputs     : " + str( self.number_of_primary_inputs       ) )
        self.Print_Log( "                         - No. of Secondary Inputs   : " + str( self.number_of_secondary_inputs     ) )
        self.Print_Log( "                         - No. of Hidden Dimensions  : " + str( self.number_of_hidden_dimensions    ) )
        self.Print_Log( "                         - No. of Outputs            : " + str( self.number_of_outputs              ) )
        self.Print_Log( "                         - Trainable Weights         : " + str( self.trainable_weights              ) )
        self.Print_Log( "                         - Feature Scaling Value     : " + str( self.feature_scale_value            ) )

        # Check(s)
        embedding_modification_list = ["average", "concatenate", "hadamard"]

        if self.final_layer_type not in self.final_layer_type_list:
            self.Print_Log( "BERTModel::Build_Model() - Error: Invalid Final Layer Type", force_print = True )
            self.Print_Log( "                         - Options: " + str( self.final_layer_type_list ), force_print = True )
            self.Print_Log( "                         - Specified Option: " + str( self.final_layer_type ), force_print = True )
            return

        if self.embedding_modification not in embedding_modification_list:
            self.Print_Log( "BERTModel::Build_Model() - Error: Invalid Embedding Modification Type", force_print = True )
            self.Print_Log( "                         - Options: " + str( embedding_modification_list ), force_print = True )
            self.Print_Log( "                         - Specified Option: " + str( self.embedding_modification ), force_print = True )
            return

        #######################
        #                     #
        #  Build Keras Model  #
        #                     #
        #######################

        # Setup BERT Model Configuration
        bert_model          = self.model_path

        # This Is Technically Not Needed As The BERT Configuration Is Automatically Loaded When Calling 'TFBertModel.from_pretrained()'
        self.bert_config    = BertConfig.from_pretrained( bert_model, num_labels = number_of_outputs )

        # Set Embedding Dimension Size
        embedding_dimension_size = self.bert_config.hidden_size

        # Setup The BERT Model - Determine If We're Loading From A File Or HuggingFace Model Archive By Model Name
        encoder             = TFBertModel.from_pretrained( bert_model, from_pt = True, config = self.bert_config )

        # Pass BERT-Specific Inputs To The Encoder, Extract Embeddings Per Input Sequence Sub-Word
        #   NOTE: Calling "encoder( input_ids, ... )" Achieves The Same, But Results In A Nested Layer Error When Saving, Then Loading The Model.
        #         Calling "encoder.bert( inputs_ids, ... )" Resolves The Nested Layer Issue.
        #         Source: https://github.com/keras-team/keras/issues/14345
        self.model_encoder_layer = encoder.bert

        # Determine If We're Refining BERT Layers In Addition To The Attached Layers Or Just Training The Attached Layers
        #   i.e. Set Encoder Layer Weights/Variables To Trainable Or Freeze Them
        encoder.trainable = self.trainable_weights

        # Create Model
        self.model = Model( encoder = self.model_encoder_layer, activation_function = self.activation_function, embedding_modification = self.embedding_modification,
                            embedding_dim = embedding_dimension_size, number_of_outputs = self.number_of_outputs, dropout_ratio = self.dropout,
                            feature_scale_value = self.feature_scale_value )

        return True


    ############################################################################################
    #                                                                                          #
    #    Model Metrics                                                                         #
    #                                                                                          #
    ############################################################################################

    def Compute_Metrics( self, y_pred, y_true, model_type = "triplet_link_prediction" ):
        precision, recall, f1_score = -1, -1, -1

        if model_type == "triplet_link_prediction":
            # Threshold Predictions
            y_pred, y_true = [ 1 if _ > 0.5 else 0 for _ in y_pred.numpy() ], y_true.numpy()

            # Aggregate TP, FP & FN Counts -> Outer Dict = True Labels, Inner Dict = Predicted Labels
            confusion_matrix = { 0 : { 0 : 0, 1 : 0 }, 1 : { 0 : 0,  1 : 0 } }

            # Confusion Matrix:
            #    True Class
            #    = ------- =
            #  P = TP | FP =
            #  r = ------- =
            #  e = FN | TN =
            #  d = ------- =
            #
            for i in range( len( y_pred ) ):
                confusion_matrix[y_pred[i]][y_true[i]] += 1

            tp = confusion_matrix[1][1]
            fp = confusion_matrix[1][0]
            fn = confusion_matrix[0][1]
            tn = confusion_matrix[0][0]

            precision = tp / ( tp + fp ) if tp > 0 or fp > 0 else 0.0
            recall    = tp / ( tp + fn ) if tp > 0 or fn > 0 else 0.0
            f1_score  = 2 * ( precision * recall ) / ( precision + recall ) if precision > 0 or recall > 0 else 0.0

        return precision, recall, f1_score


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
#    Tensorflow Model                                                                      #
#                                                                                          #
############################################################################################

class Model( tf.keras.models.Model ):
    def __init__( self, encoder = None, activation_function = "sigmoid", embedding_modification = "average",
                  embedding_dim = 768, number_of_outputs = 1, dropout_ratio = 0.1, feature_scale_value = 1.0 ):
        super( Model, self ).__init__()

        # ================ #
        # Member Variables #
        # ================ #
        if embedding_modification == "concatenate": embedding_dim *= 3
        self.embedding_dim = embedding_dim
        self.embedding_mod = embedding_modification
        self.feature_scale = feature_scale_value

        # ============ #
        # Model Layers #
        # ============ #
        self.encoder       = encoder
        self.dropout       = Dropout( name = "Dropout_Layer", rate = dropout_ratio )
        self.concatenate   = Concatenate( name = "Concatenate_Layer", axis = 1 )
        self.average       = Average( name = "Average_Layer" )
        self.multiply      = Multiply( name = "Multiply_Layer" )
        self.scaler        = Lambda( lambda x: x * self.feature_scale )
        self.dense_layer   = Dense( name = "Dense_Layer", units = embedding_dim, activation = "relu" )
        self.output_layer  = Dense( name = "Output_Layer", units = number_of_outputs, activation = activation_function )

    def call( self, token_input_ids_1 = None, attention_mask_1 = None, token_type_ids_1 = None,
              token_input_ids_2 = None, attention_mask_2 = None, token_type_ids_2 = None,
              token_input_ids_3 = None, attention_mask_3 = None, token_type_ids_3 = None ):
        primary_embeddings, secondary_embeddings, tertiary_embeddings = None, None, None

        # If Provided, Extract First Set Of Subword Embeddings & Compute Average Of Subword Embeddings To Produce A Single Representation
        if token_input_ids_1 is not None:
            primary_embeddings = self.encoder( input_ids = token_input_ids_1, attention_mask = attention_mask_1,
                                               token_type_ids = token_type_ids_1 ).last_hidden_state

            primary_embedding_mask = Reshape( target_shape = ( attention_mask_1.shape[-1], 1 ), input_shape = ( attention_mask_1.shape[-1], ) )( attention_mask_1 )
            primary_embeddings     = Embedding_Extraction_Layer( output_embedding_type = "average", hidden_size = self.embedding_dim )( [primary_embeddings, primary_embedding_mask] )

        # If Provided, Extract Second Set Of Subword Embeddings & Compute Average Of Subword Embeddings To Produce A Single Representation
        if token_input_ids_2 is not None:
            secondary_embeddings = self.encoder( input_ids = token_input_ids_2, attention_mask = attention_mask_2,
                                                 token_type_ids = token_type_ids_2 ).last_hidden_state

            secondary_embedding_mask = Reshape( target_shape = ( attention_mask_2.shape[-1], 1 ), input_shape = ( attention_mask_2.shape[-1], ) )( attention_mask_2 )
            secondary_embeddings     = Embedding_Extraction_Layer( output_embedding_type = "average", hidden_size = self.embedding_dim )( [secondary_embeddings, secondary_embedding_mask] )

        # If Provided, Extract Third Set Of Subword Embeddings & Compute Average Of Subword Embeddings To Produce A Single Representation
        if token_input_ids_3 is not None:
            tertiary_embeddings = self.encoder( input_ids = token_input_ids_3, attention_mask = attention_mask_3,
                                                token_type_ids = token_type_ids_3 ).last_hidden_state

            tertiary_embedding_mask = Reshape( target_shape = ( attention_mask_3.shape[-1], 1 ), input_shape = ( attention_mask_3.shape[-1], ) )( attention_mask_3 )
            tertiary_embeddings     = Embedding_Extraction_Layer( output_embedding_type = "average", hidden_size = self.embedding_dim )( [tertiary_embeddings, tertiary_embedding_mask] )

        # Perform Embedding Modification
        embedding = None

        if self.embedding_mod == "concatenate":
            if primary_embeddings is not None and secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.concatenate( [primary_embeddings, secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and secondary_embeddings is not None:
                embedding = self.concatenate( [primary_embeddings, secondary_embeddings] )
            elif secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.concatenate( [secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.concatenate( [primary_embeddings, tertiary_embeddings] )
        elif self.embedding_mod == "multiply":
            if primary_embeddings is not None and secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.multiply( [primary_embeddings, secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and secondary_embeddings is not None:
                embedding = self.multiply( [primary_embeddings, secondary_embeddings] )
            elif secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.multiply( [secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.multiply( [primary_embeddings, tertiary_embeddings] )
        elif self.embedding_mod == "average":
            if primary_embeddings is not None and secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.average( [primary_embeddings, secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and secondary_embeddings is not None:
                embedding = self.average( [primary_embeddings, secondary_embeddings] )
            elif secondary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.average( [secondary_embeddings, tertiary_embeddings] )
            elif primary_embeddings is not None and tertiary_embeddings is not None:
                embedding = self.average( [primary_embeddings, tertiary_embeddings] )

        # Apply Feature Scaling
        if self.feature_scale != 1.0:
            embedding = self.scaler( embedding )

        # Apply Dropout
        embedding = self.dropout( embedding )

        # Pass Embedding(s) Through Dense Layer, Then Output Layer
        x = self.dense_layer( embedding )

        return self.output_layer( x )


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NNLBD.Models import BERTModel\n" )
    print( "     model = BERTModel( network_model = \"bert\", print_debug_log = True," )
    print( "                        per_epoch_saving = False, use_csr_format = False )" )
    print( "     model.Fit( \"data/cui_mini\", epochs = 30, batch_size = 4, verbose = 1 )" )
    exit()
