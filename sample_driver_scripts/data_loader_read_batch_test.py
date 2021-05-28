#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    12/15/2020                                                                   #
#    Revised: 05/08/2020                                                                   #
#                                                                                          #
#    Tests DataLoader::Get_Next_Elements() function to load data in-line versus            #
#        all at once. Used for large files which are unable to fit in memory.              #
#                                                                                          #
#    How To Run:                                                                           #
#                    "python data_loader_read_batch_test.py"                               #
#                                                                                          #
#    Authors:                                                                              #
#    --------                                                                              #
#    Clint Cuffy   - cuffyca@vcu.edu                                                       #
#    VCU NLP Lab                                                                           #
#                                                                                          #
############################################################################################

# Standard Python Libraries
import sys
sys.path.insert( 0, "../" )

# Custom Modules
from NNLBD.DataLoader import StdDataLoader

def Main():
    print( "~Begin" )
    file_path = "../data/cui_mini"

    data_loader = StdDataLoader()

    # Assuming We're Training Over 10 Epochs
    for epoch in range( 10 ):
        # Fetch Elements Until The EOF Is Reached
        while data_loader.Reached_End_Of_File() == False:
            data_elements = data_loader.Get_Next_Batch( file_path, number_of_elements_to_fetch = 2 )

            # Only Process 'data_elements' If It Contains Data

            # Error Reading Data File
            if data_elements == [-1]:
                print( "Error Reading File \"" + str( file_path ) + "\"" )
                break
            # Process Data Element(s) From File
            elif data_elements:
                print( "Epoch " + str( epoch ) + " => " + str( data_elements ) )

        # Reset File Position Index
        if data_loader.Reached_End_Of_File():
            print( "EOF Reached / Resetting Position Index" )
            data_loader.Reset_File_Position_Index()

    print( "~Fin" )

# Runs main function when running file directly
if __name__ == '__main__':
    Main()