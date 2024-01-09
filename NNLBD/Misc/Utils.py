#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    Date:    10/07/2020                                                                   #
#    Revised: 01/03/2024                                                                   #
#                                                                                          #
#    Utilities Class For The NNLBD Package.                                                #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Modules
import hashlib, importlib.util, os, re, shutil

############################################################################################
#                                                                                          #
#    Utils Model Class                                                                     #
#                                                                                          #
############################################################################################

class Utils:
    def __init__( self ):
        pass

    def __del__( self ):
        pass

    """
        Pre-Run Check: Checks If Specified List Of Modules Are Installed

        Input:
            installed_modules : List Of Strings

        Output:
            Bool              : True -> All Are Installed / False -> One Or More Modules Are Not Installed

    """
    def Check_For_Installed_Modules( self, installed_modules : list = [] ):
        all_modules_installed = True

        for module in installed_modules:
            if importlib.util.find_spec( module ) is None:
                print( "Utils::Check_For_Installed_Modules() - Error: Module '" + str( module ) + "' Not Installed" )
                all_modules_installed = False

        return all_modules_installed

    """
        Fetches Current Working Directory/Path
    """
    def Get_Working_Directory( self ):
        return os.path.abspath( os.getcwd() )

    """
        Creates Path Along With All Folders/Directories Within The Specified Path
    """
    def Create_Path( self, file_path ):
        if not file_path or file_path == "":
            return

        file_path = re.sub( r'\\+', "/", file_path )
        folders = file_path.split( "/" )

        # Check Existing Path And Create If It Doesn't Exist
        current_path = ""

        for folder in folders:
            if self.Check_If_Path_Exists( current_path + folder ) == False:
                os.mkdir( current_path + folder )

            current_path += folder + "/"

    """
        Checks If The Specified Path Exists and It Is A Directory (Not A File)
    """
    def Check_If_Path_Exists( self, path ):
        if os.path.exists( path ) and os.path.isdir( path ):
            return True
        return False

    """
        Checks If The Specified File Exists and It Is A File (Not A Directory)
    """
    def Check_If_File_Exists( self, file_path ):
        if os.path.exists( file_path ) and os.path.isfile( file_path ):
            return True
        return False

    """
        Checks If Directory/Path Is Empty
    """
    def Is_Directory_Empty( self, path ):
        if self.Check_If_Path_Exists( path ):
            if not os.listdir( path ):
                return True
            else:
                return False

        print( "Utils::Is_Directory_Empty() - Warning: Path Is Either A File Or Not Valid" )

        return True

    """
        Checks If A Specified Path Contains Directories/Folders
    """
    def Check_If_Path_Contains_Directories( self, file_path ):
        file_path = re.sub( r'\\+', "/", file_path )
        folders   = file_path.split( "/" )
        return True if len( folders ) > 1 else False

    """
        Copies File From Source To Destination Path
    """
    def Copy_File( self, source_path, destination_path ):
        if not self.Check_If_File_Exists( source_path ):
            print( "Utils::Copy_File() - Error: Source File Does Not Exist" )
            return False

        if not self.Check_If_Path_Exists( path = destination_path ):
            print( "Utils::Copy_File() - Warning: Source Path Does Not Exist / Creating Path" )
            self.Create_Path( file_path = destination_path )

        # Copy File To Destination
        shutil.copy2( source_path, destination_path )

        return True

    """
        Checks If The Specified Path Exists and Deletes If True
    """
    def Delete_Path( self, path, delete_all_contents = False ):
        if self.Is_Directory_Empty( path ) == False and delete_all_contents == False:
            print( "Utils::Delete_Path() - Warning: Path Contains Files / Unable To Delete Path" )
            print( "                                Set 'delete_all_contents = True' To Delete Files And Path" )
            return

        if   self.Check_If_Path_Exists( path ) and delete_all_contents         : shutil.rmtree( path )
        elif self.Check_If_Path_Exists( path ) and delete_all_contents == False: os.rmdir( path )

    """
        Checks If The Specified File Exists and Deletes If True
    """
    def Delete_File( self, file_path ):
        if os.path.exists( file_path ): os.remove( file_path )

    """
        Reads Data From File And Stores Each Line In A List

        Inputs:
            file_path : File Path (String)
            lowercase : Lowercases All Text (Bool)
            encoding  : Encoding Format (String)

        Outputs:
            data_list : File Data By Line As Each List Element (List)
    """
    def Read_Data( self, file_path, lowercase = False, encoding = "utf-8" ):
        data_list = []

        # Load Training File
        if self.Check_If_File_Exists( file_path ) == False:
            return data_list

        # Read File Data
        try:
            with open( file_path, "r", encoding = encoding ) as in_file:
                data_list = in_file.readlines()
                data_list = [ line.strip() for line in data_list ]                  # Removes Trailing Space Characters From CUI Data Strings
                if lowercase: data_list = [ line.lower() for line in data_list ]    # Lowercase All Text
        except FileNotFoundError:
            print( "Utils::Read_Data() - Error: Unable To Open Data File \"" + str( file_path ) + "\"", 1 )

        return data_list

    """
        Writes Data To File

        Inputs:
            file_path : File Path (String)
            data      : Data To Write To File (String)

        Outputs:
            None
    """
    def Write_Data_To_File( self, file_path, data ):
        # Check
        if data is None or data == "":
            print( "Utils::Write_Data_To_File() - Error: No Data To Write" )
            return

        # Read File Data
        try:
            with open( file_path, "w" ) as out_file:
                out_file.write( str( data ) )
        except Exception as e:
            print( "Utils::Write_Data_To_File() - Error: Unable To Create File" + e )
            return
        finally:
            out_file.close()

    """
        Computes Hash Of File

        Input:
            file_path : File Path (String)

        Output:
            Hash      : Hash Value Or -1 (String)
    """
    def Get_Hash_Of_File( self, file_path : str = "" ):
        data = self.Read_Data( file_path = file_path, encoding = "utf8" )

        if isinstance( data, list ):
            data = "".join( data )

        if len( data ) > 0:
            result = hashlib.md5( data.encode() )
            return result.hexdigest()

        return -1

    """
        Computes Hash Of String Data

        Input:
            str  : (String)

        Output:
            Hash : Hash Value (String)
    """
    def Get_Hash_Of_String( self, data : str = "" ):
        result = hashlib.md5( data.encode() )
        return result.hexdigest()

    """
        Find Index Of Gold B Instance Within BERT Triplet Link Prediction Evaluation Data

        Inputs:
            file_path          : File Path (String)
            gold_b_instance    : Gold Instance In Format -> "concept_a\tconcept_b\tconcept_c" (String)
            instance_delimiter : Delimiter Used For 'gold_b_instance' (Default: "\t") (String)
            data_delimiter     : Delimiter Used For Data In 'file_path' (Default: "<>")
            encoding           : Read File Encoding Setting (Default: "utf8") (String)

        Outputs:
            index              : Index Of Gold B Instance In Evaluation Data
    """
    def Get_Gold_B_Instance_Idx_From_Data( self, file_path : str = "", gold_b_instance : str = "", instance_delimiter : str  = "\t",
                                           data_delimiter : str = "<^>", encoding : str = "utf8" ):
        # Check
        if not self.Check_If_File_Exists( file_path = file_path ):
            print( "Utils::Get_Gold_B_Instance_In_Data() - Error: \'" + str( file_path ) + "\' Not Found" )
            return -1

        # Split Gold B Instance Data
        gold_b_instance = gold_b_instance.lower().split( instance_delimiter )

        if len( gold_b_instance ) < 3:
            print( "Utils::Get_Gold_B_Instance_In_Data() - Error Splitting Gold B Instance: \'" + str( gold_b_instance ) + "\'" )
            return -1

        concept_a = gold_b_instance[0].split( data_delimiter )[0]
        concept_b = gold_b_instance[1].split( data_delimiter )[0]
        concept_c = gold_b_instance[2].split( data_delimiter )[0]

        # Read File Data
        try:
            with open( file_path, "r", encoding = encoding ) as rfh:
                idx = 0
                for idx, line in enumerate( rfh ):
                    line_elements = line.lower().split( instance_delimiter )

                    if len( line_elements ) < 3: continue

                    concept_a_elements = line_elements[0].split( data_delimiter )
                    concept_b_elements = line_elements[1].split( data_delimiter )
                    concept_c_elements = line_elements[2].split( data_delimiter )

                    # First Element Is Concept Identifier
                    if concept_a == concept_a_elements[0] and concept_b == concept_b_elements[0] and concept_c == concept_c_elements[0]:
                        return idx

        except Exception as e:
            print( "Utils::Get_Gold_B_Instance_In_Data() - Error: Unable To Read File" + e )
        finally:
            rfh.close()

        return -1


############################################################################################
#                                                                                          #
#    Main Function                                                                         #
#                                                                                          #
############################################################################################

# Runs main function when running file directly
if __name__ == '__main__':
    print( "**** This Script Is Designed To Be Implemented And Executed From A Driver Script ****" )
    exit()