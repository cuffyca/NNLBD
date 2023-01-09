#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    01/08/2021                                                                   #
#    Revised: 05/08/2021                                                                   #
#                                                                                          #
#    Tests DataLoader::Vectorize_Model_Data() function which includes multi-threading      #
#        support.                                                                          #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python data_loader_threaded_test.py"                                 #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Python Libraries
import os, sys, time, types
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD.DataLoader import StdDataLoader, CrichtonDataLoader
from NNLBD.Misc       import Utils

def Main():
    print( "~Begin" )

    # Start Elapsed Time Timer
    start_time     = time.time()

    file_path      = "../data/train_cs1_closed_discovery_without_aggregators_mod"
    embedding_path = "../vectors/HOC/test_modified_cs1.embeddings"

    file_path      = "../data/cui_mini_closed_discovery"
    embedding_path = "../vectors/test/vectors_random_cui_mini"

    matrix_save_path = "./test_vectorized_model_matrices/"

    utils = Utils()

    data_loader = StdDataLoader( print_debug_log = True, skip_out_of_vocabulary_words = True )
    data_loader.Read_Data( file_path )
    data_loader.Load_Embeddings( embedding_path )
    data_loader.Generate_Token_IDs()

    print( "Vectorizing Data" )
    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Data( number_of_threads = 1, use_csr_format = True,
                                                                                                model_type = "open_discovery",
                                                                                                pad_inputs = True, pad_output = True )

    # Check(s)
    if primary_inputs is None or secondary_inputs is None or tertiary_inputs is None or outputs is None:
        print( "Error Occurred During Data Vectorization" )
        exit()
    elif isinstance( primary_inputs, list ) and isinstance( outputs, list ) and len( primary_inputs ) == 0 and len( outputs ) == 0:
        print( "Error Occurred During Data Vectorization (List)" )
        exit()
    elif isinstance( primary_inputs, csr_matrix ) and isinstance( outputs, csr_matrix ) and primary_inputs[0:].shape[0] == 0 and outputs[0:].shape[0] == 0:
        print( "Error Occurred During Data Vectorization (CSR Matrix)" )
        exit()

    # Test Vectorized Data Saving And Loading
    data_loader.Save_Vectorized_Model_Data( matrix_save_path )

    primary_inputs, secondary_inputs, tertiary_inputs, outputs = None, None, None, None

    data_loader.Load_Vectorized_Model_Data( matrix_save_path )

    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Get_Primary_Inputs(), data_loader.Get_Secondary_Inputs(), data_loader.Get_Tertiary_Inputs(), data_loader.Get_Outputs()

    if isinstance( primary_inputs, np.ndarray   ):
        print( "Primary Input Length  : " + str( len( primary_inputs   ) ) + "\n" + str( primary_inputs   ) )

    if isinstance( secondary_inputs, np.ndarray ):
        print( "Secondary Input Length: " + str( len( secondary_inputs ) ) + "\n" + str( secondary_inputs ) )

    if isinstance( tertiary_inputs, np.ndarray  ):
        print( "Tertiary Input Length : " + str( len( tertiary_inputs  ) ) + "\n" + str( tertiary_inputs  ) )

    if isinstance( outputs, np.ndarray          ):
        print( "Output Length         : " + str( len( outputs          ) ) + "\n" + str( outputs          ) )

    if isinstance( primary_inputs, csr_matrix   ): print( "Primary Input Shape:\n"   + str( primary_inputs.shape   ) )
    if isinstance( secondary_inputs, csr_matrix ): print( "Secondary Input Shape:\n" + str( secondary_inputs.shape ) )
    if isinstance( tertiary_inputs, csr_matrix  ): print( "Tertiary Input Shape:\n"  + str( tertiary_inputs.shape  ) )
    if isinstance( outputs, csr_matrix          ): print( "Output Shape:\n"          + str( outputs.shape          ) )

    if isinstance( primary_inputs, csr_matrix   ): print( "Primary Inputs:\n"   + str( primary_inputs.todense()   ) )
    if isinstance( secondary_inputs, csr_matrix ): print( "Secondary Inputs:\n" + str( secondary_inputs.todense() ) )
    if isinstance( tertiary_inputs, csr_matrix  ): print( "Tertiary Inputs:\n"  + str( tertiary_inputs.todense()  ) )
    if isinstance( outputs, csr_matrix          ): print( "Outputs:\n"          + str( outputs.todense()          ) )

    elapsed_time = "{:.2f}".format( time.time() - start_time )
    print( "Elapsed Time: " + str( elapsed_time ) + " secs" )

    # Clean-Up
    utils.Delete_Path( matrix_save_path, delete_all_contents = True )

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()