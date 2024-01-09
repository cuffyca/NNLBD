#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    07/14/2023                                                                   #
#    Revised: 12/03/2023                                                                   #
#                                                                                          #
#    Standard Data Loader Classs For The NNLBD Package.                                    #
#                                                                                          #
#    Expected Format:    concept_a_data\tconcept_b_data\tconcept_c_data                    #
#                        concept_a_data\tconcept_b_data\tconcept_c_data                    #
#                        concept_a_data\tconcept_b_data\tconcept_c_data                    #
#                                              ...                                         #
#                        concept_a_data\tconcept_b_data\tconcept_c_data                    #
#                                                                                          #
#                                                                                          #
#                        Where 'concept_x_data' = concept<>synonym_a<>synonym_b<>          #
#                                                 ...<>synonym_n<>concept_definition       #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import re, threading
import numpy as np
import transformers
transformers.logging.set_verbosity_error()  # Silence HuggingFace Transformers Warnings/Info Statements
from transformers import BertTokenizer

# Custom Modules
from NNLBD.DataLoader import DataLoader


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class BERTDataLoader( DataLoader ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False, debug_log_file_handle = None,
                  restrict_output = True, output_is_embeddings = False, bert_model = "bert-base-cased", lowercase = False ):
        super().__init__( print_debug_log = print_debug_log, write_log_to_file = write_log_to_file, shuffle = shuffle,
                          skip_out_of_vocabulary_words = skip_out_of_vocabulary_words, debug_log_file_handle = debug_log_file_handle,
                          restrict_output = restrict_output, output_is_embeddings = output_is_embeddings )
        self.version    = 0.01
        self.bert_model = bert_model
        self.tokenizer  = BertTokenizer.from_pretrained( self.bert_model, do_lower_case = lowercase )

        # Standard BERT Special Tokens
        self.sub_word_cls_token    = "[CLS]"
        self.sub_word_sep_token    = "[SEP]"
        self.sub_word_pad_token    = "[PAD]"

        # Set [CLS] & [PAD] Token Variables Using BERT Tokenizer
        self.sub_word_cls_token_id = self.tokenizer.convert_tokens_to_ids( self.sub_word_cls_token )
        self.sub_word_sep_token_id = self.tokenizer.convert_tokens_to_ids( self.sub_word_sep_token )
        self.sub_word_pad_token_id = self.tokenizer.convert_tokens_to_ids( self.sub_word_pad_token )

        # Let's Set The DataLoader's Maximum Sub-Word Sequence Length Now. This Should Be 512 For HuggingFace/PyTorch Implementation
        self.max_sequence_length   = self.tokenizer.max_model_input_sizes["bert-base-uncased"]

    """
        Performs Checks Against The Specified Data File/Data List To Ensure File Integrity Before Further Processing
            (Only Checks First 10 Lines In Data File)
    """
    def Check_Data_File_Format( self, file_path = "", data_list = [], str_delimiter = '\t' ):
        read_file_handle  = None
        error_flag        = False
        number_of_indices = 10

        # Read Data From File / Favors 'data_list' Versus File Path
        if file_path == "" and len( data_list ) == 0:
            self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Using Internal Data File Stored In Memory" )
            data_list = self.Get_Data()
        elif file_path != "" and len( data_list ) == 0:
            self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Opening Data File From Path" )

            try:
                read_file_handle = open( file_path, "r", encoding = "utf8" )
            except FileNotFoundError:
                self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Error: Unable To Open Data File \"" + str( file_path ) + "\"", force_print = True )

        # Check Data List Again
        if file_path == "" and len( data_list ) == 0:
            self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Warning: No Data In Data List" )
            return True

        if len( data_list ) > 0 and number_of_indices > len( data_list ): number_of_indices = len( data_list )

        self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Verifying File Integrity" )

        # Check First 10 Lines In Data File
        for idx in range( number_of_indices ):
            line = read_file_handle.readline() if read_file_handle else data_list[idx]
            if not line: break

            line_elements = line.split( str_delimiter )
            if len( line_elements ) < 3: error_flag = True
            if error_flag:
                self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Error On Line " + str( idx ) )
                break

        # Close File Handle
        if read_file_handle: read_file_handle.close()

        # Notify User Of Data Integrity Error Found
        if error_flag:
            self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Data Integrity Error", force_print = True )
            self.Print_Log( "                                         - Expected Data In The Following Format: 'concept_1_data\\tconcept_2_data\\tconcept_3_data'", force_print = True )

        self.Print_Log( "BERTDataLoader::Check_Data_File_Format() - Complete" )

        return False if error_flag else True

    """
        Vectorized/Binarized Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list              : String (ie. C002  )
            model_type             : Model Type (String)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            stack_inputs           : True = Stacks Inputs, False = Does Not Stack Inputs - Used For BiLSTM Model (Boolean)
            keep_in_memory         : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            number_of_threads      : Number Of Threads To Deploy For Data Vectorization (Integer)
            str_delimiter          : String Delimiter - Used To Separate Elements Within An Instance (String)
            is_validation_data     : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data     : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)

        Outputs:
            primary_input_vector   : CSR Matrix or Numpy Array
            secondary_input_vector : CSR Matrix or Numpy Array
            tertiary_input_vector  : CSR Matrix or Numpy Array
            output_vector          : CSR Matrix or Numpy Array
    """
    def Encode_Model_Data( self, data_list = [], model_type = "open_discovery", use_csr_format = False, pad_inputs = True, pad_output = True,
                           stack_inputs = False, keep_in_memory = True, number_of_threads = 4, str_delimiter = '\t', is_validation_data = False, is_evaluation_data = False ):
        # Check(s)
        if len( data_list ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        if len( data_list ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error: Not Data To Vectorize / 'data_list' Is Empty", force_print = True )
            return None, None, None, None

        if number_of_threads < 1:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Warning: Number Of Threads < 1 / Setting Number Of Threads = 1", force_print = True )
            number_of_threads = 1

        if self.Check_Data_File_Format() == False:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error: Data Integrity Violation Found", force_print = True )
            return None, None, None, None

        threads          = []
        primary_inputs   = []
        secondary_inputs = []
        tertiary_inputs  = []
        outputs          = []

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Vectorizing Data Using Settings" )
        self.Print_Log( "                                    - Model Type        : " + str( model_type         ) )
        self.Print_Log( "                                    - Use CSR Format    : " + str( use_csr_format     ) )
        self.Print_Log( "                                    - Pad Inputs        : " + str( pad_inputs         ) )
        self.Print_Log( "                                    - Pad Output        : " + str( pad_output         ) )
        self.Print_Log( "                                    - Stack Inputs      : " + str( stack_inputs       ) )

        total_number_of_lines = len( data_list )

        if number_of_threads > total_number_of_lines:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Warning: 'number_of_threads > len( data_list )' / Setting 'number_of_threads = total_number_of_lines'" )
            number_of_threads = total_number_of_lines

        lines_per_thread = int( ( total_number_of_lines + number_of_threads - 1 ) / number_of_threads )

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Number Of Threads: " + str( number_of_threads ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Lines Per Thread : " + str( lines_per_thread  ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Total Lines In File Data: " + str( total_number_of_lines ) )

        ###########################################
        #          Start Worker Threads           #
        ###########################################

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Starting Worker Threads" )

        # Create Storage Locations For Threaded Data Segments
        tmp_thread_data = [None for i in range( number_of_threads )]

        for thread_id in range( number_of_threads ):
            starting_line_index = lines_per_thread * thread_id
            ending_line_index   = starting_line_index + lines_per_thread if starting_line_index + lines_per_thread < total_number_of_lines else total_number_of_lines

            new_thread = threading.Thread( target = self.Worker_Thread_Function, args = ( thread_id, data_list[starting_line_index:ending_line_index], tmp_thread_data,
                                                                                          model_type, use_csr_format, pad_inputs, pad_output, stack_inputs, str_delimiter,
                                                                                          starting_line_index + 1, ) )
            new_thread.start()
            threads.append( new_thread )

        ###########################################
        #           Join Worker Threads           #
        ###########################################

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Waiting For Worker Threads To Finish" )

        for thread in threads:
            thread.join()

        if len( tmp_thread_data ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error Vectorizing Model Data / No Data Returned From Worker Threads", force_print = True )
            return None, None, None, None

        # Concatenate Vectorized Model Data Segments From Threads
        for model_data in tmp_thread_data:
            if model_data is None or len( model_data ) < 4:
                self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error: Expected At Least Four Vectorized Elements / Received None Or < 4", force_print = True )
                continue

            # Vectorized Inputs/Outputs
            primary_input_data   = model_data[0]
            secondary_input_data = model_data[1]
            tertiary_input_data  = model_data[2]
            output_data          = model_data[3]

            # Extract And Store BERT-Specific Inputs
            if len( primary_inputs ) == 0:
                primary_inputs = primary_input_data
            else:
                primary_inputs[0].extend( primary_input_data[0] )       # Input UDS
                primary_inputs[1].extend( primary_input_data[1] )       # Attention Masks
                primary_inputs[2].extend( primary_input_data[2] )       # Token Type IDs

            if len( secondary_inputs ) == 0:
                secondary_inputs = secondary_input_data
            else:
                secondary_inputs[0].extend( secondary_input_data[0] )   # Input UDS
                secondary_inputs[1].extend( secondary_input_data[1] )   # Attention Masks
                secondary_inputs[2].extend( secondary_input_data[2] )   # Token Type IDs

            if len( tertiary_inputs ) == 0:
                tertiary_inputs = tertiary_input_data
            else:
                tertiary_inputs[0].extend( tertiary_input_data[0] )     # Input UDS
                tertiary_inputs[1].extend( tertiary_input_data[1] )     # Attention Masks
                tertiary_inputs[2].extend( tertiary_input_data[2] )     # Token Type IDs

            if len( outputs ) == 0:
                outputs = output_data
            else:
                outputs.extend( output_data )

        # Convert Inputs & Outputs To Numpy Matrices
        primary_inputs   = np.asarray( primary_inputs   )
        secondary_inputs = np.asarray( secondary_inputs )
        tertiary_inputs  = np.asarray( tertiary_inputs  )
        outputs          = np.asarray( outputs          )

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Primary Input Shape  : " + str( primary_inputs.shape ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Secondary Input Shape: " + str( secondary_inputs.shape ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Tertiary Input Shape : " + str( tertiary_inputs.shape ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Output Shape         : " + str( outputs.shape ) )

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Vectorized Primary Inputs  :\n" + str( primary_inputs   ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Vectorized Secondary Inputs:\n" + str( secondary_inputs ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Vectorized Tertiary Inputs :\n" + str( tertiary_inputs  ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Vectorized Outputs         :\n" + str( outputs          ) )

        # Clean-Up
        threads         = []
        tmp_thread_data = []

        self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Complete" )

        #####################
        # List Final Checks #
        #####################
        if isinstance( primary_inputs, list ) and len( primary_inputs ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error: Primary Input Matrix Is Empty" )
            return None, None, None, None

        if isinstance( secondary_inputs, list ) and len( secondary_inputs ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Error: Secondary Input Matrix Is Empty" )
            return None, None, None, None

        # Only Crichton Data-sets Use A Tertiary Input, So This Matrix Is Not Guaranteed To Be Non-Empty
        if isinstance( tertiary_inputs, list ) and len( tertiary_inputs ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Warning: Tertiary Input Matrix Is Empty" )

        if isinstance( outputs, list ) and len( outputs ) == 0:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Warning: Outputs Matrix Is Empty" )
            return None, None, None, None

        # These Can Be Saved Via DataLoader::Save_Vectorized_Model_Data() Function Call.
        if keep_in_memory:
            self.Print_Log( "BERTDataLoader::Encode_Model_Data() - Storing In Memory" )

            if is_validation_data:
                self.val_primary_inputs    = primary_inputs
                self.val_secondary_inputs  = secondary_inputs
                self.val_tertiary_inputs   = tertiary_inputs
                self.val_outputs           = outputs
            elif is_evaluation_data:
                self.eval_primary_inputs   = primary_inputs
                self.eval_secondary_inputs = secondary_inputs
                self.eval_tertiary_inputs  = tertiary_inputs
                self.eval_outputs          = outputs
            else:
                self.primary_inputs        = primary_inputs
                self.secondary_inputs      = secondary_inputs
                self.tertiary_inputs       = tertiary_inputs
                self.outputs               = outputs

        return primary_inputs, secondary_inputs, tertiary_inputs, outputs

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            primary_input          : List Of Strings (ie. C002  )
            secondary_input        : List Of Strings (ie. TREATS)
            tertiary_input         : List Of Strings
            outputs                : List Of Strings (May Be A List Of Lists Tokens. ie. [C001 C003 C004], [C002 C001 C010])
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            separate_outputs       : Separates Outputs By Into Individual Vectorized Instances = True, Combine Outputs Per Instance = False. (Only Valid For 'Open Discovery')
            descriptor_separator   : Separates A Single Term Description -> Desired Format: "CUI<^>Preferred_Term_a<^>...<^>Preferred_Term_n<^>CUI_Definition" (Boolean)
            instance_separator     : String Delimiter - Used To Separate Instances (String)
            term_index             : Index Of The Concept Descriptor To Encode (Int)

        Outputs:
            primary_input_vector   : Numpy Array
            secondary_input_vector : Numpy Array
            tertiary_input_vector  : Numpy Array
            output_vector          : Numpy Array
    """
    def Encode_Model_Instance( self, primary_input, secondary_input, tertiary_input = "", outputs = "", model_type = "open_discovery",
                               pad_inputs = False, pad_output = False, separate_outputs = False, descriptor_separator = "<^>",
                               instance_separator = '<:>', term_index = -1 ):
        # Split Elements Using String Delimiter
        primary_input_instances   = primary_input.split( instance_separator )
        secondary_input_instances = secondary_input.split( instance_separator )
        tertiary_input_instances  = tertiary_input.split( instance_separator )
        output_instances          = outputs.split( instance_separator )

        # Placeholders For Vectorized Inputs/Outputs
        primary_input_array   = []
        secondary_input_array = []
        tertiary_input_array  = []
        output_array          = []

        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorizing Inputs" )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() -             Primary Input  : " + str( primary_input   ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() -             Secondary Input: " + str( secondary_input ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() -             Tertiary Input : " + str( tertiary_input  ) )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() -             Outputs        : " + str( outputs         ) )

        descriptor_separator_regex = re.escape( descriptor_separator )

        for primary_input, secondary_input, tertiary_input, output in zip( primary_input_instances, secondary_input_instances, tertiary_input_instances, output_instances ):
            if re.search( descriptor_separator_regex + r'$', primary_input   ): primary_input   = re.sub( descriptor_separator_regex + r'$', "", primary_input )
            if re.search( descriptor_separator_regex + r'$', secondary_input ): secondary_input = re.sub( descriptor_separator_regex + r'$', "", secondary_input )
            if re.search( descriptor_separator_regex + r'$', tertiary_input  ): tertiary_input  = re.sub( descriptor_separator_regex + r'$', "", tertiary_input )

            temp_seen_list  = []
            primary_input   = [ _ for _ in primary_input.split( descriptor_separator )   if not ( _ in temp_seen_list or temp_seen_list.append( _ ) ) ] # set() Does Not Preserve Element Order, It's An Unordered Data Structure

            temp_seen_list  = []
            secondary_input = [ _ for _ in secondary_input.split( descriptor_separator ) if not ( _ in temp_seen_list or temp_seen_list.append( _ ) ) ] # set() Does Not Preserve Element Order, It's An Unordered Data Structure

            temp_seen_list  = []
            tertiary_input  = [ _ for _ in tertiary_input.split( descriptor_separator )  if not ( _ in temp_seen_list or temp_seen_list.append( _ ) ) ]  # set() Does Not Preserve Element Order, It's An Unordered Data Structure

            temp_seen_list  = []

            self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorizing Inputs \"" + str( primary_input ) + "\" & \"" + str( secondary_input ) + "\" & \"" + str( tertiary_input ) + "\"" )
            self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorizing Outputs \"" + str( output ) + "\"" )

            ######################
            #   Encode Outputs   #
            ######################
            if outputs != "":
                output_array.append( 1.0 if float( output ) > 0.0 else 0.0 )

            # Check
            if len( output_array ) == 0:
                self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Error: Length Of 'output_array' == 0`" )
                return [], [], [], []

            # Encode Inputs
            ###########################
            # Vectorize Primary Input #
            ###########################
            primary_input = primary_input[term_index]
            primary_input = self.tokenizer( " ".join( primary_input ), is_split_into_words = False, add_special_tokens = True, truncation = True, padding = 'max_length', max_length = self.max_sequence_length )
            primary_input_array.append( primary_input.data )

            #############################
            # Vectorize Secondary Input #
            #############################
            secondary_input = secondary_input[term_index]
            secondary_input = self.tokenizer( " ".join( secondary_input ), is_split_into_words = False, add_special_tokens = True, truncation = True, padding = 'max_length', max_length = self.max_sequence_length )
            secondary_input_array.append( secondary_input.data )

            ############################
            # Vectorize Tertiary Input #
            ############################
            tertiary_input = tertiary_input[term_index]
            tertiary_input = self.tokenizer( " ".join( tertiary_input ), is_split_into_words = False, add_special_tokens = True, truncation = True, padding = 'max_length', max_length = self.max_sequence_length )
            tertiary_input_array.append( tertiary_input.data )

        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorized Primary Input   \"" + str( primary_input_array   ) + "\"" )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorized Secondary Input \"" + str( secondary_input_array ) + "\"" )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorized Tertiary Input  \"" + str( tertiary_input_array  ) + "\"" )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Vectorized Outputs         \"" + str( output_array          ) + "\"" )
        self.Print_Log( "BERTDataLoader::Encode_Model_Instance() - Complete" )

        return primary_input_array, secondary_input_array, tertiary_input_array, output_array


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    def Load_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "BERTDataLoader::Load_Token_ID_Key_Data() - Warning: Not Supported / No Data To Save" )
        pass

    def Save_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "BERTDataLoader::Save_Token_ID_Key_Data() - Warning: Not Supported / No Data To Save" )
        pass

    """
        Generates IDs For Each Token Given The Following File Format

            Expected Format:    C001	TREATS	C002	C004	C009
                                C001	ISA	    C003
                                C002	TREATS	C005	C006	C010
                                C002	AFFECTS	C003	C004	C007 	C008
                                C003	TREATS	C002
                                C003	AFFECTS	C001	C005 	C007
                                C003	ISA	    C008
                                C004	ISA	    C010
                                C005	TREATS	C003	C004	C006
                                C005	AFFECTS	C001	C009	C010

        Inputs:
            data_list                   : List Of Data By Line (List)
            restrict_output             : Signals The DataLoader's Encoding Functions To Reduce The Model Output To Output Tokens Only Appearing In The Model Data
            scale_embedding_weight_value: Scales Embedding Weights By Specified Value ie. embedding_weights *= scale_embedding_weight_value (Float)

        Outputs:
            None
    """
    def Generate_Token_IDs( self, data_list = [], restrict_output = None, scale_embedding_weight_value = 1.0 ):
        self.Print_Log( "BERTDataLoader::Generate_Token_IDs() - Warning: Not Necessary / BERT Tokenizer Contains All Token IDs" )
        self.token_id_dictionary["<BERT_MODEL>"] = 0
        pass

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Primary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "primary_input_matrix.npy" ):
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Primary Matrix" )
            self.primary_inputs = np.load( file_path + "primary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Secondary Matrix
        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Secondary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "secondary_input_matrix.npy" ):
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Secondary Matrix" )
            self.secondary_inputs = np.load( file_path + "secondary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Secondary Input Matrix Not Found" )

        # Tertiary Matrix
        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Tertiary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "tertiary_input_matrix.npy" ):
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Tertiary Matrix" )
            self.tertiary_inputs = np.load( file_path + "tertiary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Output Matrix
        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Output Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "output_matrix.npy" ):
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Output Matrix" )
            self.outputs = np.load( file_path + "output_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Output Matrix Not Found" )

        self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Complete" )

        return False

    """
        Saves Vectorized Model Inputs/Outputs To File.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Save_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Saving Primary Matrix" )

        if self.primary_inputs is None:
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Warning: Primary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.primary_inputs, np.ndarray ):
            np.save( file_path + "primary_input_matrix.npy", self.primary_inputs, allow_pickle = True )

        # Secondary Matrix
        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Saving Secondary Matrix" )

        if self.secondary_inputs is None:
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Warning: Secondary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.secondary_inputs, np.ndarray ):
            np.save( file_path + "secondary_input_matrix.npy", self.secondary_inputs, allow_pickle = True )

        # Tertiary Matrix
        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Saving Tertiary Matrix" )

        if self.tertiary_inputs is None:
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Warning: Tertiary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.tertiary_inputs, np.ndarray ):
            np.save( file_path + "tertiary_input_matrix.npy", self.tertiary_inputs, allow_pickle = True )

        # Output Matrix
        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Saving Output Matrix" )

        if self.outputs is None:
            self.Print_Log( "BERTDataLoader::Load_Vectorized_Model_Data() - Warning: Output == 'None' / No Data To Save" )
        elif isinstance( self.outputs, np.ndarray ):
            np.save( file_path + "output_matrix.npy", self.outputs, allow_pickle = True )

        self.Print_Log( "BERTDataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Reinitialize Token ID Values For Primary And Secondary ID Dictionaries

        Inputs:
            None

        Outputs:
            None
    """
    def Reinitialize_Token_ID_Values( self ):
        self.Print_Log( "BERTDataLoader::Reinitialize_Token_ID_Values() - Warning: Not Necessary / BERT Tokenizer Contains All Token IDs" )
        pass


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
    #    Worker Thread Function                                                                #
    #                                                                                          #
    ############################################################################################

    """
        DataLoader Model Data Vectorization Worker Thread

        Inputs:
            thread_id              : Thread Identification Number (Integer)
            data_list              : List Of String Instances To Vectorize (Data Chunk Determined By BERTDataLoader::Encode_Model_Data() Function)
            dest_array             : Placeholder For Threaded Function To Store Outputs (Do Not Modify) (List)
            model_type             : Model Type (String)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            stack_inputs           : True = Stacks Inputs, False = Does Not Stack Inputs - Used For BiLSTM Model (Boolean)
            str_delimiter          : String Delimiter - Used To Separate Elements Within An Instance (String)
            data_start_idx         : Starting Index Of Data List Chunk (Used For Debugging) (Int)

        Outputs:
            primary_inputs         : CSR Matrix or Numpy Array
            secondary_inputs       : CSR Matrix or Numpy Array
            tertiary_inputs        : CSR Matrix or Numpy Array
            outputs                : CSR Matrix or Numpy Array

        Note:
            Outputs Are Stored In A List Per Thread Which Is Managed By BERTDataLoader::Encode_Model_Data() Function.

    """
    def Worker_Thread_Function( self, thread_id, data_list, dest_array, model_type = "open_discovery", use_csr_format = False,
                                pad_inputs = True, pad_output = True, stack_inputs = False, str_delimiter = '\t', data_start_idx = 0 ):
        if self.Is_Embeddings_Loaded() and pad_inputs == True:
            self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Detected Embeddings Loaded / Setting 'pad_inputs = False'" )
            pad_inputs = False

        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Vectorizing Data Using Settings" )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Use CSR Format    : " + str( use_csr_format            ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Pad Inputs        : " + str( pad_inputs                ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Pad Output        : " + str( pad_output                ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Stack Inputs      : " + str( stack_inputs              ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - String Delimiter  : " + str( str_delimiter             ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Restrict Output   : " + str( self.restrict_output      ) )
        self.Print_Log( "                                         - Thread ID: " + str( thread_id ) + " - Output Embeddings : " + str( self.output_is_embeddings ) )

        # Reduce Input/Output Space During Vectorization Of Data (Faster Data Vectorization)
        temp_pad_inputs = False if use_csr_format else pad_inputs
        temp_pad_output = False if use_csr_format else pad_output

        # Vectorized Input/Output Placeholder Lists
        primary_inputs   = []
        secondary_inputs = []
        tertiary_inputs  = []
        outputs          = []

        # Data Format: 'Concept_A_Data\tConcept_B_Data\tConcept_C_Data\tValue'
        for idx, line in enumerate( data_list, start = data_start_idx ):
            self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Data Line: " + str( line.strip() ) )

            if not line: continue

            line     = line.strip()
            elements = line.split( str_delimiter )

            # Check(s)
            #   Expects Each Instance To Be Composed Of 4 Elements
            if len( elements ) != 4:
                self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Error Parsing Line: " + str( idx ) + " -> Expected 4 Elements In Line", force_print = True )
                self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " ->               Line: " + str( line ) )
                continue
            #   Expects Last Element To Be Jaccard Similarity Coefficient
            if not re.sub( "\.|[\-+Ee]", "", elements[-1] ).isnumeric():
                self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " -> Error Parsing Line: " + str( idx ) + " -> Last Element Is Not Numeric", force_print = True )
                self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " ->               Line: " + str( line ) )
                continue

            # Vectorize Inputs/Outputs
            primary_input_array   = []
            secondary_input_array = []
            tertiary_input_array  = []
            output_array          = []

            primary_input_array, secondary_input_array, tertiary_input_array, output_array = self.Encode_Model_Instance( elements[0], elements[1], elements[2], elements[3],
                                                                                                                         pad_inputs = temp_pad_inputs, pad_output = temp_pad_output )

            # Check(s)
            if len( primary_input_array ) == 0 or len( secondary_input_array ) == 0 or len( output_array ) == 0:
                self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Warning: Error Vectorizing Input/Output Data", force_print = True )
                self.Print_Log( "                                         - Data Line Index: " + str( idx ) + " Line: \"" + str( line ) + "\"", force_print = True )
                continue

            ##################
            # Primary Inputs #
            ##################
            for instance in primary_input_array:
                input_ids, attention_mask, token_type_ids = instance['input_ids'], instance['attention_mask'], instance['token_type_ids']

                if len( primary_inputs ) == 0:
                    primary_inputs.append( [input_ids] )
                    primary_inputs.append( [attention_mask] )
                    primary_inputs.append( [token_type_ids] )
                else:
                    primary_inputs[0].append( input_ids )
                    primary_inputs[1].append( attention_mask )
                    primary_inputs[2].append( token_type_ids )

            ####################
            # Secondary Inputs #
            ####################
            for instance in secondary_input_array:
                input_ids, attention_mask, token_type_ids = instance['input_ids'], instance['attention_mask'], instance['token_type_ids']

                if len( secondary_inputs ) == 0:
                    secondary_inputs.append( [input_ids] )
                    secondary_inputs.append( [attention_mask] )
                    secondary_inputs.append( [token_type_ids] )
                else:
                    secondary_inputs[0].append( input_ids )
                    secondary_inputs[1].append( attention_mask )
                    secondary_inputs[2].append( token_type_ids )

            ###################
            # Tertiary Inputs #
            ###################
            for instance in tertiary_input_array:
                input_ids, attention_mask, token_type_ids = instance['input_ids'], instance['attention_mask'], instance['token_type_ids']

                if len( tertiary_inputs ) == 0:
                    tertiary_inputs.append( [input_ids] )
                    tertiary_inputs.append( [attention_mask] )
                    tertiary_inputs.append( [token_type_ids] )
                else:
                    tertiary_inputs[0].append( input_ids )
                    tertiary_inputs[1].append( attention_mask )
                    tertiary_inputs[2].append( token_type_ids )

            ###########
            # Outputs #
            ###########
            for instance in output_array:
                outputs.append( instance )

        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Primary Input Data:  \n" + str( primary_inputs   ) )
        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Secondary Input Data:\n" + str( secondary_inputs ) )
        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Tertiary Input Data: \n" + str( tertiary_inputs  ) )
        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Thread ID: " + str( thread_id ) + " - Output Data:         \n" + str( outputs          ) )

        # Check(s)
        if len( primary_inputs ) == 0 and len( secondary_inputs ) == 0 and len( tertiary_inputs ) == 0 and len( outputs ) == 0:
            dest_array[thread_id] = None
            return

        # Assign Thread Vectorized Data To Temporary DataLoader Placeholder Array
        dest_array[thread_id] = [primary_inputs, secondary_inputs, tertiary_inputs, outputs]

        self.Print_Log( "BERTDataLoader::Worker_Thread_Function() - Complete" )



############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    print( "     Example Code Below:\n" )
    print( "     from NNLBD.Models import DataLoader\n" )
    print( "     data_loader = BERTDataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()
