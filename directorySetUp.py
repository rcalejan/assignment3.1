"""Holds functions for directory setup."""
###################################################################
#                           Imports                               #
###################################################################
import os

###################################################################
#                          Functions                              #
###################################################################
def setUpDirectory() -> bool:
    """Sets up directory before indexing."""

    # Check to see if devs folder exists
    if os.path.exists('developer/DEV'):
        print("Using developer/DEV/* for indexing")
    else:
        print("Couldn't find DEV's folder at developer/DEV")
        return False
    
    directories = ["dumps", "docHashes", "finalIndex", "indexIndex", "masterIndex", "pageRanks", "tokenIdfIndex", "webGraph"]
    if any(os.path.exists(directory_name) for directory_name in directories):
        answer = input("IMPORTANT: FOUND EXISTING DIRECTORIES ALREADY CREATED. ARE YOU SURE YOU WANT TO DELETE THESE DIRECTORIES AND START AGAIN? THIS WILL COMPLETELY RESET INDEXING.\n\tEnter 'yes' to proceed: ")
        if answer != 'yes':
            print("INDEXING ABORTED")
            return False

    for directory_name in directories:
        # Create or clear dumps directory
        if not os.path.exists(directory_name):
            # Create the directory
            os.makedirs(directory_name)
            print(f"Directory '{directory_name}' created.")
        else:
            # Remove all the files inside the directory
            for filename in os.listdir(directory_name):
                file_path = os.path.join(directory_name, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"All files in directory '{directory_name}' deleted.")
    return True
    
    