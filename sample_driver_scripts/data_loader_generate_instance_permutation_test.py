#!/usr/bin/python

############################################################################################
#                                                                                          #
#    Neural Network - Literature Based Discovery                                           #
#    -------------------------------------------                                           #
#                                                                                          #
#    ***** Used For Debugging Purposes *****                                               #
#                                                                                          #
#    Date:    12/15/2020                                                                   #
#    Revised: 12/18/2020                                                                   #
#                                                                                          #
#    Tests DataLoader::Generate_Instance_Permutations() function to load data in-line      #
#         versus all at once. Used for large files which are unable to fit in memory.      #
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
from NNLBD import DataLoader

def Main():
    print( "~Begin" )
    file_path = "../data/cui_mini"
    
    data_loader = DataLoader()
    data_loader.Read_Data( file_path )
    
    for instance in data_loader.Get_Data():
        instance_elements = instance.split()
        input_a           = instance_elements[0]
        input_b           = instance_elements[1]
        
        for output in instance_elements[2:]:
            print( "Instance: " + str( input_a ) + " " + str( input_b ) + " " + str( output ) )
            permutations = data_loader.Generate_Instance_Permutations( input_a + " " + input_b + " " + output )
            print( "Permutations: " + str( permutations ) + "\n" )
    
    print( "~Fin" )
    
# Runs main function when running file directly
if __name__ == '__main__':
    Main()