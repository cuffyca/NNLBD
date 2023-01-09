#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    01/23/2021                                                                   #
#    Revised: 05/08/2021                                                                   #
#                                                                                          #
#    Tests DataLoader::Vectorize_Model_Inpuis() function options.                          #
#        support.                                                                          #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python data_loader_vectorized_model_input_test.py"                   #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Python Libraries
import sys
import numpy as np

sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD.DataLoader import StdDataLoader, CrichtonDataLoader

def Main():
    print( "~Begin" )

    file_path      = "../data/cui_mini_closed_discovery"
    embedding_path = "../vectors/test/vectors_random_cui_mini"

    file_path_alt      = "../data/train_cs1_closed_discovery_without_aggregators_mod"
    embedding_path_alt = "../vectors/HOC/test_modified_cs1.embeddings.bin"

    data_loader = StdDataLoader( print_debug_log = False, skip_out_of_vocabulary_words = True )
    data_loader.Read_Data( file_path )
    data_loader.Load_Embeddings( embedding_path )
    data_loader.Generate_Token_IDs()

    print( "" )

    print( "#############################################" )
    print( "# Testing DataLoader With Embeddings Loaded #" )
    print( "#############################################" )

    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0]]     else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[10]]    else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1,3,7]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = False'" )

    if np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if np.array_equal( secondary_inputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1,3,7]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0]]  else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[10]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []     else print( "  Tertiary Input : Error" )

    if np.array_equal( outputs[0], np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = False'" )

    if np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if np.array_equal( secondary_inputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )

    if np.array_equal( outputs[0], np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = True'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]]    else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[10], [10], [10]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1], [3], [7]]    else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = True'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []              else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1], [3], [7]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = True'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]]    else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[10], [10], [10]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = True'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == [] else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]]    else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[1], [3], [7]]    else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[10], [10], [10]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = False'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == [] else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[10], [10], [10]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]]    else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[1], [3], [7]]    else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = False'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == [] else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )

    print( "\n" )

    data_loader = None
    data_loader = StdDataLoader( print_debug_log = False, skip_out_of_vocabulary_words = True )
    data_loader.Read_Data( file_path )
    data_loader.Generate_Token_IDs()

    print( "################################################" )
    print( "# Testing DataLoader Without Embeddings Loaded #" )
    print( "################################################" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0]]     else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[0]]    else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1,2,9]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = False'" )

    if np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if np.array_equal( secondary_inputs[0], np.array([1, 0, 0]) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1,2,9]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0]]  else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[0]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []     else print( "  Tertiary Input : Error" )

    if np.array_equal( outputs[0], np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1]) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = False )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = False'" )

    if np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if np.array_equal( secondary_inputs[0], np.array([1, 0, 0]) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []        else print( "  Tertiary Input : Error" )

    if np.array_equal( outputs[0], np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1]) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = True'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]] else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[0], [0], [0]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []              else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1], [2], [9]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = True'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([1, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([1, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([1, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []              else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[1], [2], [9]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = True'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]] else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[0], [0], [0]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "open_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = True )

    print( "\nChecking Open Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = True'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([1, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([1, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([1, 0, 0]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == [] else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = False, pad_output = False, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = False', 'pad_output = False', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]] else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[1], [2], [9]] else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []              else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[0], [0], [0]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = True, pad_output = False, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = True', 'pad_output = False', 'separate_outputs = False'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []              else print( "  Tertiary Input : Error" )
    print( "  Output         : OK" ) if outputs          == [[0], [0], [0]] else print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = False, pad_output = True, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = False', 'pad_output = True', 'separate_outputs = False'" )

    print( "  Primary Input  : OK" ) if primary_inputs   == [[0], [0], [0]]    else print( "  Primary Input  : Error" )
    print( "  Secondary Input: OK" ) if secondary_inputs == [[1], [2], [9]]    else print( "  Secondary Input: Error" )
    print( "  Tertiary Input : OK" ) if tertiary_inputs  == []                 else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([1, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([1, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([1, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )


    primary_inputs, secondary_inputs, tertiary_inputs, outputs = data_loader.Encode_Model_Instance( "C001", "treats", "", "C002 C004 C008", model_type = "closed_discovery",
                                                                                                     pad_inputs = True, pad_output = True, separate_outputs = False )

    print( "\nChecking Closed Discovery: 'pad_inputs = True', 'pad_output = True', 'separate_outputs = False'" )

    if ( np.array_equal( primary_inputs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( primary_inputs[2], np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) ) ):
        print( "  Primary Input  : OK" )
    else:
        print( "  Primary Input  : Error" )

    if ( np.array_equal( secondary_inputs[0], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[1], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) ) and
         np.array_equal( secondary_inputs[2], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) ) ):
        print( "  Secondary Input: OK" )
    else:
        print( "  Secondary Input: Error" )

    print( "  Tertiary Input : OK" ) if tertiary_inputs  == [] else print( "  Tertiary Input : Error" )

    if ( np.array_equal( outputs[0], np.array([1, 0, 0]) ) and
         np.array_equal( outputs[1], np.array([1, 0, 0]) ) and
         np.array_equal( outputs[2], np.array([1, 0, 0]) ) ):
        print( "  Output         : OK" )
    else:
        print( "  Output         : Error" )

    print( "\n~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()