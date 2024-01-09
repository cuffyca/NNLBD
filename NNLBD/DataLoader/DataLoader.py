#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/08/2020                                                                   #
#    Revised: 07/16/2023                                                                   #
#                                                                                          #
#    Base Data Loader Classs For The NNLBD Package.                                        #
#                                                                                          #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################


# Standard Modules
import itertools, re, scipy, struct
import numpy as np
from scipy.sparse import csr_matrix

# Custom Modules
from NNLBD.Misc import Utils


############################################################################################
#                                                                                          #
#    Data Loader Model Class                                                               #
#                                                                                          #
############################################################################################

class DataLoader( object ):
    def __init__( self, print_debug_log = False, write_log_to_file = False, shuffle = True, skip_out_of_vocabulary_words = False,
                  debug_log_file_handle = None, restrict_output = True, output_is_embeddings = False ):
        self.version                      = 0.19
        self.debug_log                    = print_debug_log                 # Options: True, False
        self.write_log                    = write_log_to_file               # Options: True, False
        self.debug_log_file_handle        = debug_log_file_handle           # Debug Log File Handle
        self.debug_log_file_name          = "DataLoader_Log.txt"            # File Name (String)
        self.file_data_header_type        = "general_data"                  # "general_data", "open_discovery" or "closed_discovery"
        self.token_id_dictionary          = {}                              # Token ID Dictionary: Used For Converting Tokens To Token IDs (One-Hot Encodings)
        self.primary_id_dictionary        = {}
        self.secondary_id_dictionary      = {}
        self.tertiary_id_dictionary       = {}
        self.output_id_dictionary         = {}
        self.skip_out_of_vocabulary_words = skip_out_of_vocabulary_words    # Options: True, False
        self.restrict_output              = restrict_output                 # Reduces Outputs To Unique Tokens Seen Within The Model Data
        self.output_is_embeddings         = output_is_embeddings            # Used To Determine How To Encode Output (i.e Standard Encoding Or Embeddings)
        self.number_of_primary_tokens     = 0
        self.number_of_secondary_tokens   = 0
        self.number_of_tertiary_tokens    = 0
        self.number_of_output_tokens      = 0
        self.data_list                    = []
        self.primary_inputs               = []
        self.val_primary_inputs           = []
        self.eval_primary_inputs          = []
        self.secondary_inputs             = []
        self.val_secondary_inputs         = []
        self.eval_secondary_inputs        = []
        self.tertiary_inputs              = []
        self.val_tertiary_inputs          = []
        self.eval_tertiary_inputs         = []
        self.outputs                      = []
        self.val_outputs                  = []
        self.eval_outputs                 = []
        self.embeddings                   = []
        self.embeddings_loaded            = False
        self.simulate_embeddings_loaded   = False
        self.is_cui_data                  = False
        self.shuffle                      = shuffle                         # Not Used Currently
        self.current_line_index           = 0
        self.reached_eof                  = False
        self.read_file_handle             = None
        self.generated_embedding_ids      = False
        self.embedding_type_list          = ["primary", "secondary", "tertiary", "output"]
        self.padding_token                = "<*>padding<*>"
        self.output_embeddings            = []
        self.data_cache_dir               = "./.data_cache"
        self.data_cache_hash_dict         = {}
        self.utils                        = Utils()

        # Check For Data Cache Directory
        if not Utils().Check_If_Path_Exists( str( self.data_cache_dir ) ):
            Utils().Create_Path( str( self.data_cache_dir ) )
            Utils().Write_Data_To_File( str( self.data_cache_dir ) + "/file_hash_list.txt", "" )
        else:
            file_hash_list = Utils().Read_Data( str( self.data_cache_dir ) + "/file_hash_list.txt" )
            self.data_cache_hash_dict = { file_hash.split()[0] : file_hash.split()[1] for file_hash in file_hash_list }

        # Create Log File Handle
        if self.write_log and self.debug_log_file_handle is None:
            self.debug_log_file_handle = open( self.debug_log_file_name, "w" )

    """
        Remove Variables From Memory
    """
    def __del__( self ):
        # Write Data Cache Hash Dict To Cache Directory
        data_cache_hash_dict = [ str( k ) + "<>" + str( v ) + "\n" for k, v in self.data_cache_hash_dict.items() ]
        data_cache_hash_dict = "".join( data_cache_hash_dict )

        if not Utils().Check_If_Path_Exists( str( self.data_cache_dir ) ):
            Utils().Create_Path( str( self.data_cache_dir ) )

        Utils().Write_Data_To_File( str( self.data_cache_dir ) + "/file_hash_list.txt", data_cache_hash_dict )

        self.Clear_Data()
        self.Close_Read_File_Handle()
        del self.utils
        if self.write_log and self.debug_log_file_handle is not None: self.debug_log_file_handle.close()

    """
        Performs Checks Against The Specified Data File/Data List To Ensure File Integrity Before Further Processing
            (Only Checks First 10 Lines In Data File)
    """
    def Check_Data_File_Format( self, file_path = "", data_list = [], is_crichton_format = False, str_delimiter = '\t' ):
        self.Print_Log( "DataLoader::Check_Data_File_Format() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
    """
    def Pre_Encoding_Model_Data_Routine( self, file_path = "", model_type = "open_discovery" ):
        file_hash = Utils().Get_Hash_Of_File( file_path = file_path )

        if file_hash != -1 and file_hash in self.data_cache_hash_dict:
            print( "Did Stuff" )

        return False

    """
    """
    def Post_Encoding_Model_Data_Routine( self, file_path = "", model_type = "open_discovery" ):
        pass

    """
        Vectorized/Binarized Model Data - Used For Training/Evaluation Data

        Inputs:
            data_list              : String (ie. C002  )
            model_type             : Model Type (String)
            use_csr_format         : True = Output Model Inputs/Output As Scipy CSR Matrices, False = Output Model Inputs/Outputs As Numpy Arrays
            is_crichton_format     : Sets If Data Is In Regular TSV Format (Boolean)
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            stack_inputs           : True = Stacks Inputs, False = Does Not Stack Inputs - Used For BiLSTM Model (Boolean)
            keep_in_memory         : True = Keep Model Data In Memory After Vectorizing, False = Discard Data After Vectorizing (Data Is Always Returned) (Boolean)
            number_of_threads      : Number Of Threads To Deploy For Data Vectorization (Integer)
            str_delimiter          : String Delimiter - Used To Separate Elements Within An Instance (String)
            is_validation_data     : True = Data To Be Encoded Is Validation Data, False = Data To Be Encoded Is Not Validation Data (Stores Encoded Data In Respective Variables) (Boolean)
            is_evaluation_data     : True = Data To Be Encoded Is Evaluation Data, False = Data To Be Encoded Is Not Evaluation Data (Stores Encoded Data In Respective Variables) (Boolean)
    """
    def Encode_Model_Data( self, data_list = [], model_type = "open_discovery", use_csr_format = False, is_crichton_format = False, pad_inputs = True, pad_output = True,
                           stack_inputs = False, keep_in_memory = True, number_of_threads = 4, str_delimiter = '\t', is_validation_data = False, is_evaluation_data = False ):
        self.Print_Log( "DataLoader::Encode_Model_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Vectorized/Binarized Model Data - Single Input Instances And Output Instance

        Inputs:
            primary_input          : List Of Strings (ie. C002  )
            secondary_input        : List Of Strings (ie. TREATS)
            tertiary_input         : List Of Strings
            outputs                : List Of Strings (May Be A List Of Lists Tokens. ie. [C001 C003 C004], [C002 C001 C010])
            model_type             : Model Type (String)
            is_crichton_format     : Sets If Data Is In Regular TSV Format (Boolean)
            pad_inputs             : Pads Model Inputs. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            pad_output             : Pads Model Output. True = One-Hot Vector, False = Array Of Non-Zero Indices (Boolean)
            separate_outputs       : Separates Outputs By Into Individual Vectorized Instances = True, Combine Outputs Per Instance = False. (Only Valid For 'Open Discovery')
            instance_separator     : String Delimiter - Used To Separate Instances (String)
    """
    def Encode_Model_Instance( self, primary_input, secondary_input, tertiary_input = "", outputs = "", model_type = "open_discovery",
                               is_crichton_format = False, pad_inputs = False, pad_output = False, separate_outputs = False, instance_separator = '<:>' ):
        self.Print_Log( "DataLoader::Encode_Model_Instance() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError


    ############################################################################################
    #                                                                                          #
    #    Data Functions                                                                        #
    #                                                                                          #
    ############################################################################################

    """
        Fetches Concept Unique Identifier (CUI) Data From The File

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

                                or

            Crichton Format:    C001	TREATS	C002	0.01
                                C001	ISA	    C003    0.5
                                C002	TREATS	C005	0.998
                                C002	AFFECTS	C003	0.442
                                C003	TREATS	C002    0.1235
                                C003	AFFECTS	C001	0.642
                                C003	ISA	    C008    0.12
                                C004	ISA	    C010    0.00001
                                C005	TREATS	C003	0.1234
                                C005	AFFECTS	C001	0.0

        Inputs:
            file_path : file path (String)

        Outputs:
            data_list : File Data By Line As Each List Element (List)
    """
    def Read_Data( self, file_path, convert_to_lower_case = True, keep_in_memory = True, read_all_data = True, number_of_lines_to_read = 32 ):
        data_list = []
        in_file   = None

        # Load Training File
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "DataLoader::Read_Data() - Error: Data File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return []

        # Read Concept Unique Identifier-Predicate Occurrence Data From File
        self.Print_Log( "DataLoader::Read_Data() - Reading Data File: \"" + str( file_path ) + "\"" )


        # Read All Data And Store In Memory
        if read_all_data == True:
            try:
                with open( file_path, "r", encoding = "utf8" ) as in_file:
                    data_list = in_file.readlines()
            except FileNotFoundError:
                self.Print_Log( "DataLoader::Read_Data() - Error: Unable To Open Data File \"" + str( file_path ) + "\"", force_print = True )
                return []
            finally:
                in_file.close()

            # Check For File Data Type Header
            if data_list[0] == "node1\tnode2\tnode3\tlabel\n":
                self.Print_Log( "DataLoader::Read_Data() - Crichton Format" )
                data_list = data_list[1:]
            elif data_list[0] == "a_concept\tc_concept\tb_concepts\n":
                self.Print_Log( "DataLoader::Read_Data() - Detected 'Open Discovery+' Data Format" )
                self.file_data_header_type = "closed_discovery+"
                data_list = data_list[1:]
            elif data_list[0] == "a_concept\tb_concept\tc_concepts\n":
                self.Print_Log( "DataLoader::Read_Data() - Detected 'Closed Discovery+' Data Format" )
                self.file_data_header_type = "open_discovery+"
                data_list = data_list[1:]
            elif data_list[0] == "a_concept\tb_concept\tc_concept\n":
                self.Print_Log( "DataLoader::Read_Data() - Detected 'General' Data Format" )
                self.file_data_header_type = "open_discovery"
                data_list = data_list[1:]
            elif data_list[0] == "a_concept\tc_concept\tb_concept\n":
                self.Print_Log( "DataLoader::Read_Data() - Detected 'General' Data Format" )
                self.file_data_header_type = "closed_discovery"
                data_list = data_list[1:]

        # Read Data In Segments Of 'number_of_lines_to_read' Chunks
        else:
            # Open File If Not In Memory
            if self.read_file_handle is None:
                try:
                    self.read_file_handle = open( file_path, "r", encoding = "utf8" )
                except FileNotFoundError:
                    self.Print_Log( "DataLoader::Read_Data() - Error: Unable To Open Data File \"" + str( file_path ) + "\"", force_print = True )
                    return []
            # Read Data From File
            else:
                for i in range( number_of_lines_to_read ):
                    # Read Line From File
                    line = self.read_file_handle.readline()

                    # Check For File Data Type Header
                    if line == "a_concept\tc_concept\tb_concepts\n":
                        self.file_data_header_type = "closed_discovery+"
                        continue
                    elif line == "a_concept\tb_concept\tc_concepts\n":
                        self.file_data_header_type = "open_discovery+"
                        continue
                    elif line == "a_concept\tb_concept\tc_concept\n":
                        self.file_data_header_type = "open_discovery"
                        continue
                    elif line == "a_concept\tc_concept\tb_concept\n":
                        self.file_data_header_type = "closed_discovery"
                        continue

                    # Line Contains Data
                    if line:
                        data_list.append( line )
                        self.current_line_index += 1
                    # Reached EOF / Reset Index Counter
                    else:
                        self.reached_eof = True
                        break

        self.Print_Log( "DataLoader::Read_Data() - File Data In Memory" )

        data_list = [ line.strip() for line in data_list ]    # Removes Trailing Space Characters From CUI Data Strings
        if convert_to_lower_case: data_list = [ line.lower() for line in data_list ]    # Converts All String Characters To Lowercase

        self.Print_Log( "DataLoader::Read_Data() - File Data List Length: " + str( len( data_list ) ) )

        if keep_in_memory:
            self.Print_Log( "DataLoader::Read_Data() - Storing In Memory" )
            self.data_list = data_list

        self.Print_Log( "DataLoader::Read_Data() - Complete" )

        return data_list

    """
        Checks Embedding File Format And Decides Which Embedding Load Function To Call: Text Or Binary Embedding Loader

        Inputs:
            file_path        : file path (String)
            lowercase        : Lowercases All Read Text (Bool)
            encoding_format  : Encoding Format Of Embeddings (Words). Used In Load_Binary_Embeddings() (String)
            store_embeddings : True = Keep In Memory, False = Return To User Without Storing In Memory (Boolean)

        Outputs:
            embedding_data   : List Static Embeddings
    """
    def Load_Embeddings( self, file_path, lowercase = False, encoding_format = "utf-8", store_embeddings = True ):
        # Check(s)
        if file_path == "":
            self.Print_Log( "DataLoader::Load_Embeddings() - Warning: No File Path Specified", force_print = True )
            return []

        self.Print_Log( "DataLoader::Load_Embeddings() - File: \"" + str( file_path ) + "\"" )

        if self.utils.Check_If_File_Exists( file_path ):
            self.Print_Log( "DataLoader::Load_Embeddings() - Loading Embeddings" )

            # Check If File Is Binary-Formatted (Assume Not Until Proven Otherwise)
            is_binary_format, read_chunk_size = False, 512
            number_of_chunks_to_read          = 10

            with open( file_path, 'rb' ) as read_file_handle:

                for _ in range( number_of_chunks_to_read ):
                    file_data = read_file_handle.read( read_chunk_size )

                    # Check If Null Character Is In Read File Chunk
                    if b'\0' in file_data:
                        is_binary_format = True
                        break
                    # We've Reached The End Of The File
                    elif len( file_data ) < read_chunk_size: break

                read_file_handle.close()

            # Determined File Is Binary-Encoded Embeddings
            if is_binary_format:
                return self.Load_Binary_Embeddings( file_path = file_path, lowercase = lowercase,
                                                    store_embeddings = store_embeddings, encoding_format = encoding_format )

            # Determine File Is Plain Text Embeddings
            return self.Load_Text_Embeddings( file_path = file_path, lowercase = lowercase,
                                              store_embeddings = store_embeddings )

        self.Print_Log( "DataLoader::Load_Embeddings() - Error: Embedding File Not Found In Path \"" + str( file_path ) + "\"", force_print = True )
        return []

    """
        Reads Word2vec Formatted Binary Formatted Embeddings

        Inputs:
            file_path                   : file path (String)
            lowercase                   : Lowercases All Read Text (Bool)
            encoding_format             : Encoding Format Of Embeddings (Words). Used In Load_Binary_Embeddings() (String)
            store_embeddings            : True = Keep In Memory, False = Return To User Without Storing In Memory (Boolean)
            try_to_load_multi_word_terms: Attempts To Load Multi-Word Term Vectors (i.e. Words Separated By Whitespace) (Boolean) - Waning: Not Finished

        Outputs:
            embedding_data              : List Plain Text Embeddings
    """
    def Load_Binary_Embeddings( self, file_path, lowercase = False, encoding_format = "utf-8", store_embeddings = True, try_to_load_multi_word_terms = False ):
        if not file_path or file_path == "":
            self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Error: Embedding File Path Empty Or Not Defined: " + str( file_path ), force_print = True )
            return []

        self.Print_Log( "DataLoader::Load_Binary_Embeddings() - File: " + str( file_path ) )

        if self.utils.Check_If_File_Exists( file_path ):
            self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Loading Embeddings" )
            text_word_embeddings = []

            with open( file_path, 'rb' ) as read_file_handle:
                # Read Heading Information: Number Of Vectors and Vector Length
                header_elements = read_file_handle.readline().split()

                # Check For Multi-Word Term Flag
                multi_word_term_flag = True if len( header_elements ) == 3 else False

                number_of_embeddings, embedding_length = map( int, header_elements[0:2] )
                unpacked_embedding_size                = np.float32().itemsize * embedding_length

                # Store Header Information
                # text_word_embeddings.append( str( number_of_embeddings ) + " " + str( embedding_length ) )

                word_bytes, word_embedding_bytes = [], []

                for _ in range( int( number_of_embeddings ) ):
                    # Read Word
                    word, temp_word      = b'', b''
                    word_found, read_cnt = 0, False

                    while True:
                        character  = read_file_handle.read( 1 )
                        temp_word += character
                        read_cnt  += 1

                        if character == b' ':
                            # Account For Trailing Whitespace ' ' Before Line Ending
                            if word == b'' and temp_word == b' ': continue

                            if try_to_load_multi_word_terms:
                                # Check If 'temp_word' Is Able To Be Unpacked
                                #   i.e. 'temp_word' Is 4 Bytes And Unpackable To Floatf
                                #   If It Is, Don't Add To Word, It's Actually Part Of The Word Embedding
                                try:
                                    if word_found:
                                        num_within_range, num_of_loops = 0, round( len( temp_word ) / 4 )

                                        # Skip 'temp_word's Which Contain Four (Or More) Alpha Or Numeric Characters
                                        #    i.e. More Than Likely They're Part Of Multi-Word Terms
                                        if re.match( r'^\w{4,}|^\d{4,}', temp_word.decode( encoding_format ) ):
                                            pass
                                        else:
                                            for l_idx in range( num_of_loops ):
                                                # Ignore String Byte Of All Integers
                                                temp_word_byte = temp_word[l_idx*4:(l_idx*4)+4]

                                                # Try To Unpack The Byte Array, If It Succeeds Go To The Next Step
                                                #   If It Fails, Then It's Probably Part Of A Multi-Word Term... Add It To The Current Term
                                                temp_float = struct.unpack( "f", temp_word_byte )

                                                # Check If Value Is Within Normal Word Embedding Boundaries
                                                #   If Not, Let It Slide... It's Probably A Word Within A Multi-Word Term
                                                #   (Yes, There Will Be Cases In Which A Sub-Word Converts To A Float)
                                                #   e.g. 'chlo' in 'chloride'
                                                if -1.0 <= temp_float[0] <= 1.0: num_within_range += 1

                                            # If All Byte Arrays Of 4 Elements Check Out, Then It's More Than Likely Part Of The Word Embedding
                                            #   Go Back To Before We Extracted 'temp_word'
                                            if num_of_loops > 0 and num_within_range == num_of_loops:
                                                read_file_handle.seek( curr_word_pos, 0 )
                                                break

                                # Unpacking 'temp_word' Failed. Add 'temp_word' To 'word' And Continue
                                except Exception:
                                    pass

                            word          += temp_word
                            curr_word_pos  = read_file_handle.tell()
                            temp_word      = b''
                            word_found     = True
                            if not try_to_load_multi_word_terms: break
                        # Non-Text Characters Were Extracted From The File (i.e. Embedding Float Elements)
                        elif word_found and r'\x' in str( temp_word ):
                            read_file_handle.seek( curr_word_pos, 0 )
                            break
                        # We've Called 'read_file_handle.read( 1 )' 500 Or More Times And Read Nothing
                        #   Report Warning: The Number Of Embeddings != The Number Of Extracted Embeddings
                        elif read_cnt >= 500 and word == b'' and temp_word == b'':
                            self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Error: Number Of Embeddings != Number Of Extracted Embeddings", force_print = True )
                            self.Print_Log( "DataLoader::Load_Binary_Embeddings() -        Number Of Embeddings: " + str( number_of_embeddings ),   force_print = True )
                            self.Print_Log( "DataLoader::Load_Binary_Embeddings() -        Number Of Extractede Embeddings: " + str( _ ),           force_print = True )
                            break

                    # Try Decoding The Extracted Word Embedding String (Used For Debugging)
                    try:
                        t_word = word.decode( encoding_format )
                    except Exception:
                        self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Error Reading Binary Embeddings", force_print = True )
                        self.Print_Log( "DataLoader::Load_Binary_Embeddings() -     Last Word: " + str( word_bytes[-1].decode( encoding_format ) ), force_print = True )
                        self.Print_Log( "DataLoader::Load_Binary_Embeddings() -     Curr Word: " + str( word ), force_print = True )
                        return []

                    # Store Word And Read Word Embedding
                    word_bytes.append( word )
                    word_embedding_bytes.append( np.fromfile( read_file_handle, np.float32, embedding_length ) )

                    # Determine Current File Position After Numpy Read
                    curr_word_pos += unpacked_embedding_size

                # Decode Bytes To Strings And Remove Newline From Beginning of Word
                words = [ bytes.decode( encoding = encoding_format ).lstrip() for bytes in word_bytes ]
                if lowercase: words = [ word.lower() for word in words ]

                # If Multi-Word Term Flag, Substitute Underscore In Multi-Word Term With Whitespace
                if multi_word_term_flag: words = [ re.sub( r'\_', " ", word ) for word in words ]

                # Store Word And Word Embedding
                text_word_embeddings = [ word + " ".join( map( str, word_embedding ) ) for word, word_embedding in zip( words, word_embedding_bytes ) ]

            # Store Embeddings
            if store_embeddings:
                self.embeddings        = text_word_embeddings
                self.embeddings_loaded = True

            self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Complete" )
            return text_word_embeddings

        self.Print_Log( "DataLoader::Load_Binary_Embeddings() - Error: Embedding File Not Found In Path \"" + str( file_path ) + "\"", force_print = True )
        return []

    """
        Saves Word2vec Formatted Embeddings In W2V Binary Format

        Inputs:
            embeddings      : List Of Embeddings - Standard W2V Format (List)
            save_file_path  : Binary Embedding Save File Path (String)
            encoding_format : Encoding Format Of Embeddings (String)
            multi_word_mod  : Save Using Multi-Word Term Modificiation (Boolean)

        Outputs:
            Boolean
    """
    def Save_Binary_Embeddings( self, embeddings = [], save_file_path = "", encoding_format = "utf-8", multi_word_mod = True ):
        if len( embeddings ) == 0:
            self.Print_Log( "DataLoader::Save_Binary_Embeddings() - Error: Embedding File Path Empty Or Not Defined: " + str( save_file_path ), force_print = True )
            return False
        if not save_file_path or save_file_path == "":
            self.Print_Log( "DataLoader::Save_Binary_Embeddings() - Error: Embedding File Path Empty Or Not Defined: " + str( save_file_path ), force_print = True )
            return False

        self.Print_Log( "DataLoader::Save_Binary_Embeddings() - Saving Binary Embeddings To File: " + str( save_file_path ) )

        with open( save_file_path, 'wb' ) as write_file_handle:
            # Determine Header Information: Number Of Embeddings And Vector Length
            number_of_embeddings = len( embeddings )

            # Determine Vector Length: Determine Where Word Embedding Ends (Accounts For Multi-Word Term Embeddings)
            idx_to_check, temp_vector_length =  [1, -1], []

            for idx in idx_to_check:
                split_idx = 0

                for element in embeddings[idx].split():
                    if re.match( r'^-*\d+\.\d+|^-*\d+[Ee][-+]*\d+', element ): break
                    split_idx += 1

                # Check: There Must Be At Least One Word Within The Word Embedding
                if split_idx == 0: split_idx = 1

                temp_vector_length.append( len( embeddings[idx].split()[split_idx:] ) )

            vector_length = min( temp_vector_length )

            # Determine If Multi-Word Mod Is Necessary / Turn Off If Not Needed
            if multi_word_mod:
                contains_multi_word_terms = False

                for i in range( number_of_embeddings ):
                    embedding_elements   = embeddings[i].split()
                    word, word_embedding = embedding_elements[0:-vector_length], embedding_elements[-vector_length:]
                    if len( word ) > 1: contains_multi_word_terms = True

                multi_word_mod = contains_multi_word_terms

            if multi_word_mod:
                header_info = bytes( str( number_of_embeddings ) + " " + str( vector_length ) + " " + str( "<1>" ) + "\n", encoding = encoding_format )
            else:
                header_info = bytes( str( number_of_embeddings ) + " " + str( vector_length ) + "\n", encoding = encoding_format )

            # Write Header Information
            write_file_handle.write( header_info )

            for i in range( number_of_embeddings ):
                embedding_elements   = embeddings[i].split()
                word, word_embedding = embedding_elements[0:-vector_length], embedding_elements[-vector_length:]

                if len( word_embedding ) < vector_length:
                    self.Print_Log( "DataLoader::Save_Binary_Embeddings() - Error: Embedding Length Less Than Expected Length", force_print = True )
                    self.Print_Log( "DataLoader::Save_Binary_Embeddings() -        Word    : " + str( word ),                   force_print = True )
                    self.Print_Log( "DataLoader::Save_Binary_Embeddings() -        Expected: " + str( vector_length ),          force_print = True )
                    self.Print_Log( "DataLoader::Save_Binary_Embeddings() -        Obtained: " + str( len( word_embedding ) ),  force_print = True )
                    return False

                # If Multi-Word Mod Flag, Substitute Whitespace Between Word Of Multi-Word Term With Underscore
                if multi_word_mod:
                    word = bytes( str( "_".join( word ) ) + " ", encoding = encoding_format )
                else:
                    word = bytes( str( " ".join( word ) ) + " ", encoding = encoding_format )

                # Write Word In Binary Format
                write_file_handle.write( word )

                # Pack Embedding
                word_embedding = np.asarray( word_embedding, dtype = np.float32 )
                write_file_handle.write( word_embedding.tobytes() )

            write_file_handle.close()

            self.Print_Log( "DataLoader::Save_Binary_Embeddings() - Complete" )
            return True

    """
        Loads Static Embeddings From File
          Expects Standard/Plain Text Vector Format

          i.e. token_a 0.1001 0.1345 ... 0.8002
               ...
               token_n 0.9355 0.1749 ... 0.6042

        Inputs:
            file_path        : file path (String)
            lowercase        : Lowercases All Read Text (Bool)
            store_embeddings : True = Keep In Memory, False = Return To User Without Storing In Memory (Boolean)

        Outputs:
            embedding_data   : List Plain Text Embeddings
    """
    def Load_Text_Embeddings( self, file_path, lowercase = False, store_embeddings = True, location = "a" ):
        embedding_data = []

        # Check(s)
        if file_path == "":
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - Warning: No File Path Specified", force_print = True )
            return []

        self.Print_Log( "DataLoader::Load_Text_Embeddings() - File: \"" + str( file_path ) + "\"" )

        if self.utils.Check_If_File_Exists( file_path ):
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - Loading Embeddings" )
            embedding_data = self.utils.Read_Data( file_path = file_path, lowercase = lowercase )
        else:
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - Error: Embedding File Not Found In Path \"" + str( file_path ) + "\"", force_print = True )
            return []

        # Check(s)
        if len( embedding_data ) == 0:
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - Error: Embedding File Contains No Data / Length == 0", force_print = True )
            return []

        # Detect Number Of Embeddings And Embedding Dimensions (Word2vec Format/Header)
        number_of_embeddings = 0
        embedding_length     = 0
        possible_header_info = embedding_data[0]

        # Set Embedding Variables And Remove Word2vec Header From Data
        if re.match( r'^\d+\s+\d+', possible_header_info ):
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - Detected Word2vec Embedding Header" )
            header_elements      = possible_header_info.split()
            number_of_embeddings = header_elements[0]
            embedding_length     = header_elements[1]
            embedding_data       = embedding_data[1:]
            self.Print_Log( "                                   - Number Of Reported Embeddings          : " + str( number_of_embeddings ) )
            self.Print_Log( "                                   - Number Of Reported Embedding Dimensions: " + str( embedding_length     ) )
        else:
            self.Print_Log( "DataLoader::Load_Text_Embeddings() - No Word2vec Embedding Header Detected / Computing Header Info" )
            number_of_embeddings = len( embedding_data )

            # Determine Vector Length: Determine Where Word Embedding Ends (Accounts For Multi-Word Term Embeddings)
            idx_to_check, temp_vector_length =  [1, -1], []

            for idx in idx_to_check:
                split_idx = 0

                for element in embedding_data[idx].split():
                    if re.match( r'^-*\d+\.\d+|^-*\d+[Ee][-+]*\d+', element ): break
                    split_idx += 1

                # Check: There Must Be At Least One Word Within The Word Embedding
                if split_idx == 0: split_idx = 1

                temp_vector_length.append( len( embedding_data[idx].split()[split_idx:] ) )

            embedding_length = min( temp_vector_length )

        self.Print_Log( "DataLoader::Load_Text_Embeddings() - Number Of Actual Embeddings          : " + str( number_of_embeddings ) )
        self.Print_Log( "DataLoader::Load_Text_Embeddings() - Number Of Actual Embedding Dimensions: " + str( embedding_length     ) )

        # Store Embeddings
        if store_embeddings:
            self.embeddings        = embedding_data
            self.embeddings_loaded = True

        self.Print_Log( "DataLoader::Load_Text_Embeddings() - Complete" )
        return embedding_data

    """
        Saves Word2vec Formatted Embeddings In Plain Text Format

        Inputs:
            embeddings       : List Of Embeddings - Standard W2V Format (List)
            save_file_path   : Binary Embedding Save File Path (String)
            encoding_format  : Encoding Format Of Embeddings (String)

        Outputs:
            Boolean
    """
    def Save_Text_Embeddings( self, embeddings = [], save_file_path = "", encoding_format = "utf-8" ):
        if len( embeddings ) == 0:
            self.Print_Log( "DataLoader::Save_Text_Embeddings() - Error: Embedding File Path Empty Or Not Defined: " + str( save_file_path ), force_print = True )
            return False
        if not save_file_path or save_file_path == "":
            self.Print_Log( "DataLoader::Save_Text_Embeddings() - Error: Embedding File Path Empty Or Not Defined: " + str( save_file_path ), force_print = True )
            return False

        self.Print_Log( "DataLoader::Save_Text_Embeddings() - Saving Plain Text Embeddings To File: " + str( save_file_path ) )

        with open( save_file_path, 'w', encoding = encoding_format ) as write_file_handle:
            # Determine Header Information
            number_of_embeddings = len( embeddings )

            # Determine Vector Length: Determine Where Word Embedding Ends (Accounts For Multi-Word Term Embeddings)
            idx_to_check, temp_vector_length =  [1, -1], []

            for idx in idx_to_check:
                split_idx = 0

                for element in embeddings[idx].split():
                    if re.match( r'^-*\d+\.\d+|^-*\d+[Ee][-+]*\d+', element ): break
                    split_idx += 1

                # Check: There Must Be At Least One Word Within The Word Embedding
                if split_idx == 0: split_idx = 1

                temp_vector_length.append( len( embeddings[idx].split()[split_idx:] ) )

            vector_length = min( temp_vector_length )

            # Write Header Info
            header_info = str( number_of_embeddings ) + " " + str( vector_length ) + "\n"

            # Write Header Information
            write_file_handle.write( header_info )

            for i in range( number_of_embeddings ):
                word, word_embedding = embeddings[i].split( " ", maxsplit = 1 )
                word           = str( word )
                word_embedding = str( word_embedding )

                # Clean Word Embedding
                word_embedding = re.sub( "\[|\]|\n|^\s+|\s+$",  "", word_embedding )
                word_embedding = re.sub( "\s+", " ", word_embedding )

                write_file_handle.write( word + " " + word_embedding + "\n" )

            write_file_handle.close()

        self.Print_Log( "DataLoader::Save_Text_Embeddings() - Complete" )
        return True

    def Load_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "DataLoader::Load_Token_ID_Key_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    def Save_Token_ID_Key_Data( self, file_path ):
        self.Print_Log( "DataLoader::Save_Token_ID_Key_Data() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Generates IDs For Each Token Given The Following File Format
    """
    def Generate_Token_IDs( self, data_list ):
        # Iterate Through The Data And Generate The Unique Input/Output Lists
        # Check(s) - If User Does Not Specify Data, Use The Data Stored In Memory
        if len( data_list ) == 0:
            self.Print_Log( "DataLoader::Generate_Token_IDs() - Warning: No Data Specified By User / Using Data Stored In Memory" )
            data_list = self.data_list

        self.Print_Log( "DataLoader::Generate_Token_IDs() - Building Token ID Dictionaries" )

        # Insert Padding At First Index Of The Token ID Dictionaries
        padding_token = self.Get_Padding_Token()

        if padding_token not in self.token_id_dictionary     : self.token_id_dictionary[padding_token]     = 0
        if padding_token not in self.primary_id_dictionary   : self.primary_id_dictionary[padding_token]   = 0
        if padding_token not in self.secondary_id_dictionary : self.secondary_id_dictionary[padding_token] = 0
        if padding_token not in self.tertiary_id_dictionary  : self.tertiary_id_dictionary[padding_token]  = 0
        if padding_token not in self.output_id_dictionary    : self.output_id_dictionary[padding_token]    = 0

        # Process Elements In Data List, Line-By-Line
        self.Print_Log( "DataLoader::Generate_Token_IDs() - Building Unique Input/Output Dictionaries" )

        for sequence in data_list:
            self.Print_Log( "DataLoader::Generate_Token_IDs() - Processing Sequence: " + str( sequence ) )

            sequence_tokens = sequence.split()

            # IF We're Not Using Embeddings, Build Unique Token ID Dictionary
            if self.generated_embedding_ids == False:
                for token in sequence_tokens:
                    # Check To See If Element Is Already In Dictionary, If Not Add The Element
                    if token not in self.token_id_dictionary:
                        index = len( self.token_id_dictionary )
                        self.Print_Log( "DataLoader::Generate_Token_IDs() - Adding Token: \"" + str( token ) + "\" => Embedding Row Index: " + str( index ) )
                        self.token_id_dictionary[token] = index

            # Build Unique Primary Input ID Dictionary
            if sequence_tokens[0] not in self.primary_id_dictionary:
                index = len( self.primary_id_dictionary )
                self.Print_Log( "DataLoader::Generate_Token_IDs() - Adding To Primary Dictionary: " + str( sequence_tokens[0] ) + " => " + str( index ) )
                self.primary_id_dictionary[sequence_tokens[0]] = index

            # Build Unique Secondary Input ID Dictionary
            if sequence_tokens[1] not in self.secondary_id_dictionary:
                index = len( self.secondary_id_dictionary )
                self.Print_Log( "DataLoader::Generate_Token_IDs() - Adding To Secondary Dictionary: " + str( sequence_tokens[1] ) + " => " + str( index ) )
                self.secondary_id_dictionary[sequence_tokens[1]] = index

            # Build Unique Output ID Dictionary
            for token in sequence_tokens[2:]:
                if token not in self.output_id_dictionary:
                    index = len( self.output_id_dictionary )
                    self.Print_Log( "DataLoader::Generate_Token_IDs() - Adding To Output Dictionary: " + str( token ) + " => " + str( index ) )
                    self.output_id_dictionary[token] = index

        self.Print_Log( "DataLoader::Generate_Token_IDs() - Dictionaries Built" )

        self.number_of_primary_tokens   = len( self.primary_id_dictionary   )
        self.number_of_secondary_tokens = len( self.secondary_id_dictionary )
        self.number_of_output_tokens    = len( self.output_id_dictionary    )

    """
        Load Vectorized Model Inputs/Outputs To File. This Favors CSR_Matrix Files Before Numpy Arrays.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Load_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Primary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "primary_input_matrix.npz" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Primary Matrix" )
            self.primary_inputs = scipy.sparse.load_npz( file_path + "primary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "primary_input_matrix.npy" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Primary Matrix" )
            self.primary_inputs = np.load( file_path + "primary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Secondary Matrix
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Secondary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "secondary_input_matrix.npz" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Secondary Matrix" )
            self.secondary_inputs = scipy.sparse.load_npz( file_path + "secondary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "secondary_input_matrix.npy" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Secondary Matrix" )
            self.secondary_inputs = np.load( file_path + "secondary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Secondary Input Matrix Not Found" )

        # Tertiary Matrix
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Tertiary Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "tertiary_input_matrix.npz" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Tertiary Matrix" )
            self.tertiary_inputs = scipy.sparse.load_npz( file_path + "tertiary_input_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "tertiary_input_matrix.npy" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Tertiary Matrix" )
            self.tertiary_inputs = np.load( file_path + "tertiary_input_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Primary Input Matrix Not Found" )

        # Output Matrix
        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Output Matrix" )

        if self.utils.Check_If_File_Exists( file_path + "output_matrix.npz" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading CSR Format Output Matrix" )
            self.outputs = scipy.sparse.load_npz( file_path + "output_matrix.npz" )
        elif self.utils.Check_If_File_Exists( file_path + "output_matrix.npy" ):
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Loading Numpy Format Output Matrix" )
            self.outputs = np.load( file_path + "output_matrix.npy", allow_pickle = True )
        else:
            self.Print_Log( "Warning: Output Matrix Not Found" )

        self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Complete" )

        return False

    """
        Saves Vectorized Model Inputs/Outputs To File.

        Inputs:
            file_path : File Path/Directory (String)

        Outputs:
            None
    """
    def Save_Vectorized_Model_Data( self, file_path ):
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Save Directory: \"" + str( file_path ) + "\"" )

        self.utils.Create_Path( file_path )

        if not re.search( r"\/$", file_path ): file_path += "/"

        # Primary Matrix
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Saving Primary Matrix" )

        if self.primary_inputs is None:
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Warning: Primary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.primary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "primary_input_matrix.npz", self.primary_inputs )
        elif isinstance( self.primary_inputs, np.ndarray ):
            np.save( file_path + "primary_input_matrix.npy", self.primary_inputs, allow_pickle = True )

        # Secondary Matrix
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Saving Secondary Matrix" )

        if self.secondary_inputs is None:
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Warning: Secondary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.secondary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "secondary_input_matrix.npz", self.secondary_inputs )
        elif isinstance( self.secondary_inputs, np.ndarray ):
            np.save( file_path + "secondary_input_matrix.npy", self.secondary_inputs, allow_pickle = True )

        # Tertiary Matrix
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Saving Tertiary Matrix" )

        if self.tertiary_inputs is None:
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Warning: Tertiary Inputs == 'None' / No Data To Save" )
        elif isinstance( self.tertiary_inputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "tertiary_input_matrix.npz", self.tertiary_inputs )
        elif isinstance( self.tertiary_inputs, np.ndarray ):
            np.save( file_path + "tertiary_input_matrix.npy", self.tertiary_inputs, allow_pickle = True )

        # Output Matrix
        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Saving Output Matrix" )

        if self.outputs is None:
            self.Print_Log( "DataLoader::Load_Vectorized_Model_Data() - Warning: Output == 'None' / No Data To Save" )
        elif isinstance( self.outputs, csr_matrix ):
            scipy.sparse.save_npz( file_path + "output_matrix.npz", self.outputs )
        elif isinstance( self.outputs, np.ndarray ):
            np.save( file_path + "output_matrix.npy", self.outputs, allow_pickle = True )

        self.Print_Log( "DataLoader::Save_Vectorized_Model_Data() - Complete" )

        return False

    """
        Fetches Token ID From String.

        Inputs:
            token    : Token (String)

        Outputs:
            token_id : Token ID Value (Integer)
    """
    def Get_Token_ID( self, token ):
        self.Print_Log( "DataLoader::Get_Token_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Fetches Token String From ID Value.

        Inputs:
            index_value  : Token ID Value (Integer)

        Outputs:
            key          : Token String (String)
    """
    def Get_Token_From_ID( self, index_value ):
        self.Print_Log( "DataLoader::Get_Token_From_ID() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Inputs:
            embedding_type : Type Of Embeddings (Primary, Secondary, Tertiary or Output)

        Outputs:
            embeddings     : List/Matrix Of Embeddings
    """
    def Get_Model_Embeddings( self, embedding_type = "primary" ):
        self.Print_Log( "StdDataLoader::Get_Model_Embeddings() - Fetching Embeddings" )
        self.Print_Log( "StdDataLoader::Get_Model_Embeddings() -     Embedding Type: " + str( embedding_type ) )

        if embedding_type is not None and embedding_type not in self.embedding_type_list:
            self.Print_Log( "StdDataLoader::Get_Model_Embeddings() - Error: Unknown Embedding Type: " + str( embedding_type ), force_print = True )
            return []

        embeddings = None

        # Get 'Primary', 'Secondary' & 'Tertiary' Embeddings (Full Output)
        #   Model Type Does Not Matter For These Embeddings, They're Always Using The Full Embedding List/Matrix
        if embedding_type == "primary" or embedding_type == "secondary" or embedding_type == "tertiary" :
            embeddings = self.Get_Embeddings( embedding_type = None )

        # Get 'Output' Embeddings (Full Or Reduced Output)
        if embedding_type == "output":
            # Use Restricted Embeddings If Specified i.e. 'self.restrict_output == True'.
            #   Otherwise, Use Full Output For Embeddings.
            if self.Get_Restrict_Output():
                if len ( self.output_embeddings ) > 0:
                    embeddings = self.output_embeddings
                else:
                    embeddings = self.Get_Embeddings( embedding_type = embedding_type )
                    self.output_embeddings = embeddings
            else:
                embeddings = self.Get_Embeddings( embedding_type = None )

        return embeddings

    """
        Fetches Next Batch Of Data Instances From File. (Assumes File Is Not Entirely Read Into Memory).

        Inputs:
            number_of_elements_to_fetch  : (Integer)

        Outputs:
            data_list                    : List Of Elements Read From File (String)
    """
    def Get_Next_Batch( self, file_path, number_of_elements_to_fetch ):
        # Load Training File
        if self.utils.Check_If_File_Exists( file_path ) == False:
            self.Print_Log( "DataLoader::Get_Next_Batch() - Error: Data File \"" + str( file_path ) + "\" Does Not Exist", force_print = True )
            return [-1]

        self.Print_Log( "DataLoader::Get_Next_Batch() - Fetching The Next " + str( number_of_elements_to_fetch ) + " Elements" )

        if self.Reached_End_Of_File():
            self.Print_Log( "DataLoader::Get_Next_Batch() - Reached EOF" )
            return []

        data_list = self.Read_Data( file_path, read_all_data = False, keep_in_memory = False, number_of_lines_to_read = number_of_elements_to_fetch )

        self.Print_Log( "DataLoader::Get_Next_Batch() - Fetched " + str( len( data_list ) ) + " Elements" )
        self.Print_Log( "DataLoader::Get_Next_Batch() - Complete" )

        return data_list

    """
        Resets The File Position Index Given The EOF Flag Has Been Set.

        Inputs:
            None

        Outputs:
            None
    """
    def Reset_File_Position_Index( self ):
        if self.Reached_End_Of_File() and self.read_file_handle is not None:
            self.current_line_index = 0
            self.reached_eof        = False
            self.read_file_handle.seek( 0 )

            self.Print_Log( "DataLoader::Reset_File_Position_Index() - Resetting File Position Elements" )
        else:
            self.Print_Log( "DataLoader::Reset_File_Position_Index() - Warning: File Position Index Not EOF" )

    """
        Reinitialize Token ID Values For Primary And Secondary ID Dictionaries

        Inputs:
            None

        Outputs:
            None
    """
    def Reinitialize_Token_ID_Values( self ):
        self.Print_Log( "DataLoader::Reinitialize_Token_ID_Values() - Called Parent Function / Not Implemented / Call Child Function", force_print = True )
        raise NotImplementedError

    """
        Generates All Permutations Of Inputs And Outputs

        Inputs:
            inputs        : String Of Model Inputs Of The Expected Format 'input_a input_b output_a'

        Outputs:
            permutations  :
    """
    def Generate_Instance_Permutations( self, inputs ):
        self.Print_Log( "DataLoader::Generate_Instance_Permutations() - Re-initializing Token ID Values" )

        # Check(s)
        if inputs is None or inputs == "":
            self.Print_Log( "DataLoader::Generate_Instance_Permutations() - Passed Empty String Parameter" )
            return []

        input_elements = inputs.split()

        # Check(s)
        if len( input_elements ) < 3:
            self.Print_Log( "DataLoader::Generate_Instance_Permutations() - Not Enough Elements In Input Parameter" )
            return []

        input_element_permutations = itertools.permutations( input_elements )
        input_element_permutations = list( input_element_permutations )

        self.Print_Log( "DataLoader::Generate_Instance_Permutations() - Complete" )
        return input_element_permutations

    """
        Compares The Tokens From The Loaded Data File To The Loaded Embedding Tokens.
            Reports Tokens Present In The Loaded Data Which Are Not Present In The Embedding Representations.

            Note: Load Training/Testing Data, Embeddings And Call self.Generate_Token_IDs() Prior To Calling This Function.

        Inputs:
            None

        Outputs:
            None
    """
    def Generate_Token_Embedding_Discrepancy_Report( self ):
        # Check(s)
        if self.Is_Embeddings_Loaded() == False:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Error: No Embeddings Loaded In Memory" )
            return

        if self.Is_Data_Loaded() == False:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Error: No Data Loaded In Memory" )
            return

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Locating OOV Tokens / Comparing Data To Embedding Tokens" )

        out_of_vocabulary_tokens = []

        for data in self.Get_Data():
            data_tokens = data.split()

            for token in data_tokens:
                if self.Get_Token_ID( token ) == -1 and token not in out_of_vocabulary_tokens: out_of_vocabulary_tokens.append( token )

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Generating Discrepancy Report" )

        if len( out_of_vocabulary_tokens ) > 0:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Found " + str( len( out_of_vocabulary_tokens ) ) + " OOV Tokens" )

            report  = "Total Number Of OOV Tokens: " + str( len( out_of_vocabulary_tokens ) ) + "\n"
            report += "OOV Tokens:\n"

            for token in out_of_vocabulary_tokens:
                report += str( token ) + "\t"

            report += "\n"
        else:
            self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - No OOV Tokens Found" )

            report = "No OOV Tokens Found"

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Writing Discrepancy Report To File: \"./discrepancy_report.txt\"" )

        self.utils.Write_Data_To_File( "./discrepancy_report.txt", report )

        # Clean-Up
        report                   = ""
        out_of_vocabulary_tokens = []

        self.Print_Log( "DataLoader::Generate_Token_Embedding_Discrepancy_Report() - Complete" )

    """
        Generates A Unique Token List Given Training Data File Path. Saves to './unique_tokens.txt'

        Inputs:
            data_file_path : File Path (String)
            str_delimiter  : String Character Which Separates Tokens Within A Line

        Outputs:
            None
    """
    def Generate_Data_Unique_Token_List( self, data_file_path, str_delimiter ):
        # Check(s)
        if self.utils.Check_If_File_Exists( data_file_path ) == False:
            self.Print_Log( "DataLoader::Generate_Data_Unique_Token_List() - Error: File Path Does Not Exist" )
            return

        self.Print_Log( "DataLoader::Generate_Data_Unique_Token_List() - Generating Unique Token List" )

        file_handle      = None
        unique_tokens    = []
        unique_token_str = ""

        try:
            file_handle = open( data_file_path, "r" )

            for line_index, line in enumerate( file_handle ):
                if not line:   break

                line          = line.strip()
                line_elements = line.split( str_delimiter )

                # Check
                for element in line_elements:
                    if element not in unique_tokens:
                        unique_tokens.append( element )
                        unique_token_str += str( element ) + "\n"

        except FileNotFoundError:
            self.Print_Log( "DataLoader::Generate_Data_Unique_Token_List() - Error: Unable To Open Data File \"" + str( data_file_path ) + "\"", force_print = True )
            return
        finally:
            if file_handle       is not None: file_handle.close()

        if len( unique_token_str ) > 0: self.utils.Write_Data_To_File( "./unique_tokens.txt", unique_token_str )

        # Clean-Up
        unique_tokens    = []
        unique_token_str = ""

        self.Print_Log( "DataLoader::Generate_Data_Unique_Token_List() - Complete" )

    """
        Checks First Five Lines Of Data For CUIs. Used To Distinguish Between SemMedDB Data And Other Data-sets.

        Inputs:
            data_list     : List Of Data Instances
            str_delimiter : String Delimiter (String)

        Outputs:
            boolean       : True - Data Contains CUIs, False - No CUIs Detected
    """
    def Is_Data_Composed_Of_CUIs( self, data_list = [], str_delimiter = '\t' ):
        number_of_cuis_detected  = 0
        number_of_lines_to_check = 5
        internal_data_used       = False

        self.Print_Log( "DataLoader::Is_Data_Composed_Of_CUIs() - Checking If Data Is Composed Of CUIs" )

        if len( data_list ) == 0:
            internal_data_used = True
            data_list == self.Get_Data()

        # Check
        if len( data_list ) == 0:
            self.Print_Log( "DataLoader::Is_Data_Composed_Of_CUIs() - No Data To Check / Data List Is Empty" )
            return False

        if len( data_list ) < number_of_lines_to_check: number_of_lines_to_check = len( data_list )

        for index in range( number_of_lines_to_check ):
            line = data_list[index]
            line_elements = line.split( str_delimiter )

            for element in line_elements:
                if re.search( r"^[Cc]\d+", element ):
                    number_of_cuis_detected += 1

        self.Print_Log( "DataLoader::Is_Data_Composed_Of_CUIs() - Number Of CUIs Detected: " + str( number_of_cuis_detected )  )
        self.Print_Log( "DataLoader::Is_Data_Composed_Of_CUIs() - Complete" )

        if internal_data_used and number_of_cuis_detected > 0: self.is_cui_data = True

        return False if number_of_cuis_detected == 0 else True

    """
        Generates A List Of Indices That Contain Non-Zero Elements

        Inputs:
            data_list     : List Of Integers/Floats

        Outputs:
            non_zero_list : List Of Integers (Indices)
    """
    def Get_Indices_Of_Non_Zero_Elements( self, data_list ):
        non_zero_list = []

        for index, element in enumerate( data_list ):
            if element != 0:
                non_zero_list.append( index )

        return non_zero_list


    """
        Closes Read_Data() In-Line Read File Handle

        Inputs:
            None

        Outputs:
            None
    """
    def Close_Read_File_Handle( self ):
        if self.read_file_handle is not None: self.read_file_handle.close()

    """
        Clears Embedding Data From Memory

        Inputs:
            None

        Outputs:
            None
    """
    def Clear_Embedding_Data( self ):
        self.embeddings        = []
        self.output_embeddings = []
        self.embeddings_loaded = False

    """
        Clears Data From Memory

        Inputs:
            None

        Outputs:
            None
    """
    def Clear_Data( self ):
        self.number_of_primary_tokens     = 0
        self.number_of_secondary_tokens   = 0
        self.number_of_tertiary_tokens    = 0
        self.data_list                    = []
        self.primary_inputs               = []
        self.secondary_inputs             = []
        self.outputs                      = []
        self.embeddings                   = []
        self.output_embeddings            = []
        self.token_id_dictionary          = {}
        self.primary_id_dictionary        = {}
        self.secondary_id_dictionary      = {}
        self.tertiary_id_dictionary       = {}
        self.output_id_dictionary         = {}
        self.embeddings_loaded            = False
        self.simulate_embeddings_loaded   = False
        self.is_cui_data                  = False
        self.current_line_index           = 0
        self.reached_eof                  = False
        self.read_file_handle             = None
        self.generated_embedding_ids      = False


    ############################################################################################
    #                                                                                          #
    #    Supporting Functions                                                                  #
    #                                                                                          #
    ############################################################################################

    """
        Prints Debug Text To Console
    """
    def Print_Log( self, text, print_new_line = True, force_print = False ):
        if self.debug_log or force_print:
            print( text ) if print_new_line else print( text, end = " " )
        if self.write_log:
            self.Write_Log( text, print_new_line )

    """
        Prints Debug Log Text To File
    """
    def Write_Log( self, text, print_new_line = True ):
        if self.write_log and self.debug_log_file_handle is not None:
            self.debug_log_file_handle.write( text + "\n" ) if print_new_line else self.debug_log_file_handle.write( text )


    ############################################################################################
    #                                                                                          #
    #    Accessor Functions                                                                    #
    #                                                                                          #
    ############################################################################################

    def Get_Embeddings( self, embedding_type = None ):
        # Check(s)
        if embedding_type is not None and embedding_type not in self.embedding_type_list:
            self.Print_Log( "DataLoader::Get_Embeddings() - Error: Unknown Embedding Type: " + str( embedding_type ), force_print = True )
            return []

        embeddings = None

        # 0th Element Is Padding / Shifts Embedding Matrix Down A Row By 1
        if   embedding_type == "primary"  : embeddings = np.zeros( ( self.Get_Number_Of_Primary_Elements(),   self.Get_Embedding_Dimension_Size() ) )
        elif embedding_type == "secondary": embeddings = np.zeros( ( self.Get_Number_Of_Secondary_Elements(), self.Get_Embedding_Dimension_Size() ) )
        elif embedding_type == "tertiary" : embeddings = np.zeros( ( self.Get_Number_Of_Tertiary_Elements(),  self.Get_Embedding_Dimension_Size() ) )
        elif embedding_type == "output"   : embeddings = np.zeros( ( self.Get_Number_Of_Output_Elements(),    self.Get_Embedding_Dimension_Size() ) )

        # Only Fetch Embedding If There Exists Something In The Respective Dictionary Outside Of The '<*>PADDING<*>' Token
        if embedding_type == "primary" and len( self.primary_id_dictionary ) > 1:
            for index, token in enumerate( self.primary_id_dictionary ):
                if index == 0: continue # Skip Padding
                embeddings[index] = np.asarray( self.embeddings[self.token_id_dictionary[token]], dtype = 'float32' )
        elif embedding_type == "secondary" and len( self.secondary_id_dictionary ) > 1:
            for index, token in enumerate( self.secondary_id_dictionary ):
                if index == 0: continue # Skip Padding
                embeddings[index] = np.asarray( self.embeddings[self.token_id_dictionary[token]], dtype = 'float32' )
        elif embedding_type == "tertiary" and len( self.tertiary_id_dictionary ) > 1:
            for index, token in enumerate( self.tertiary_id_dictionary ):
                if index == 0: continue # Skip Padding
                embeddings[index] = np.asarray( self.embeddings[self.token_id_dictionary[token]], dtype = 'float32' )
        elif embedding_type == "output" and len( self.output_id_dictionary ) > 1:
            for index, token in enumerate( self.output_id_dictionary ):
                if index == 0: continue # Skip Padding
                embeddings[index] = np.asarray( self.embeddings[self.token_id_dictionary[token]], dtype = 'float32' )
        elif embedding_type is None:
            return self.embeddings
        else:
            self.Print_Log( "DataLoader::Get_Embeddings() - Error: Unable To Fetch Embeddings - Embedding Type: " + str( embedding_type ), force_print = True )

        return embeddings

    def Get_Primary_ID_Dictionary( self ):          return self.primary_id_dictionary

    def Get_Secondary_ID_Dictionary( self ):        return self.secondary_id_dictionary

    def Get_Tertiary_ID_Dictionary( self ):         return self.tertiary_id_dictionary

    def Get_Output_ID_Dictionary( self ):           return self.output_id_dictionary

    def Get_Token_ID_Dictionary( self ):            return self.token_id_dictionary

    def Get_Number_Of_Unique_Features( self ):      return len( self.token_id_dictionary )

    def Get_Number_Of_Primary_Elements( self ):     return self.number_of_primary_tokens

    def Get_Number_Of_Secondary_Elements( self ):   return self.number_of_secondary_tokens

    def Get_Number_Of_Tertiary_Elements( self ):    return self.number_of_tertiary_tokens

    def Get_Number_Of_Output_Elements( self ):      return self.number_of_output_tokens

    def Get_Skip_Out_Of_Vocabulary_Words( self ):   return self.skip_out_of_vocabulary_words

    def Get_Restrict_Output( self ):                return self.restrict_output

    def Get_Output_Is_Embeddings( self ):           return self.output_is_embeddings

    def Get_Data( self ):                           return self.data_list

    def Get_Primary_Inputs( self ):                 return self.primary_inputs

    def Get_Val_Primary_Inputs( self ):             return self.val_primary_inputs

    def Get_Eval_Primary_Inputs( self ):            return self.eval_primary_inputs

    def Get_Secondary_Inputs( self ):               return self.secondary_inputs

    def Get_Val_Secondary_Inputs( self ):           return self.val_secondary_inputs

    def Get_Eval_Secondary_Inputs( self ):          return self.eval_secondary_inputs

    def Get_Tertiary_Inputs( self ):                return self.tertiary_inputs

    def Get_Val_Tertiary_Inputs( self ):            return self.val_tertiary_inputs

    def Get_Eval_Tertiary_Inputs( self ):           return self.eval_tertiary_inputs

    def Get_Outputs( self ):                        return self.outputs

    def Get_Val_Outputs( self ):                    return self.val_outputs

    def Get_Eval_Outputs( self ):                   return self.eval_outputs

    def Get_Number_Of_Embeddings( self ):           return len( self.embeddings )

    # Note: Call 'Generate_Token_IDs()' Prior To Calling This Function Or Subtract '1' From The Return Value
    def Get_Embedding_Dimension_Size( self ):       return len( self.embeddings[1] ) if len( self.embeddings ) > 0 else 0

    def Is_Embeddings_Loaded( self ):               return self.embeddings_loaded

    def Simulate_Embeddings_Loaded_Mode( self ):    return self.simulate_embeddings_loaded

    def Is_Data_Loaded( self ):                     return True if len( self.Get_Data() ) > 0 else False

    def Is_Dictionary_Loaded( self ):               return True if self.Get_Number_Of_Unique_Features() > 0 else False

    def Reached_End_Of_File( self ):                return True if self.reached_eof else False

    def Get_Is_CUI_Data( self ):                    return self.is_cui_data

    def Get_Padding_Token( self ):                  return self.padding_token


    ############################################################################################
    #                                                                                          #
    #    Mutator Functions                                                                     #
    #                                                                                          #
    ############################################################################################

    def Set_Token_ID_Dictionary( self, id_dictionary ):     self.token_id_dictionary = id_dictionary

    def Set_Restrict_Output( self, value ):                 self.restrict_output = value

    def Set_Output_Is_Embeddings( self, value ):            self.output_is_embeddings = value

    def Set_Debug_Log_File_Handle( self, file_handle ):     self.debug_log_file_handle = file_handle

    def Set_Simulate_Embeddings_Loaded_Mode( self, value ):
        self.simulate_embeddings_loaded = value
        self.generated_embedding_ids    = value
        self.embeddings_loaded          = True if value == True else self.embeddings_loaded


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
    print( "     data_loader = DataLoader( print_debug_log = True )" )
    print( "     data = data_loader.Read_Data( \"path_to_file\" )" )
    exit()
