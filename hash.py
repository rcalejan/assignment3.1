"""File contains definitions of functions for hashing documents."""

###################################################################
#                           Imports                               #
###################################################################
from zlib import crc32
import os
from glob import glob


###################################################################
#                Global Variabls and Constants                    #
###################################################################
DOCUMENT_PATHS = [file for dir in os.walk('developer/DEV') for file in glob(os.path.join(dir[0], '*.json'))]


###################################################################
#                          Functions                              #
###################################################################
def simHash(tokens: dict[str:int]) -> int:
    """Returns simhash value of tokenized words.
       Uses crc32."""
    
    # Initialize 32 bit vector
    vector = [0 for i in range(32)]
    num_bits = 32

    # Update vector for each word
    for key, value in tokens.items(): # Iterate through words and their counts
        value = value[0]
        hashed_value = crc32(bytes(key, 'utf-8')) # Compute 32 bit hash value for word
        bits = [(hashed_value >> bit) & 1 for bit in range(num_bits - 1, -1, -1)] # Convert integer hash value to list of binary digits
        for index, bit in enumerate(bits): # Iterate through bianry digits
            if bit == 1:
                vector[index] += value # If binary digit is 1 then add word count to vector position of binary digit
            elif bit == 0:
                vector[index] -= value # If binary digit is 1 then subtract word count to vector position of binary digit
    
    binary_list = [1 if value >=0 else 0 for value in vector] # Convert vector into binary.  1 if vector[i] is positive. 0 Otherwise.
    integer = int("".join(str(x) for x in binary_list), 2) # Convert binary to integer
    return integer # Return integer representation of hashed value


def isSimilar(frequencies: dict[str:int], sim_hashes: dict, current_id: int) -> bool:
    """Checks if file tokens are similar to a previously crawled page using sim_hash."""

    # Calcualte sim_hash value of the tokens using custom sim_hash function above
    hashed_value = simHash(frequencies)
    
    # Iterate through previously crawled documents and their sim_hash values
    for docId, value in sim_hashes.items():
        
        # Count how many bits the previously crawled sim_hash and the new sim_hash have in common.
        # If that number is greater than 31, then return True
        if bin(hashed_value ^ value).count("1") > 31: 
            print(f"Not Includeing doc {current_id} because it is too similar to doc {docId}")
            print(DOCUMENT_PATHS[current_id])
            print(DOCUMENT_PATHS[docId])
            return True
    
    # If new document is not similar to old ones then add document and its sim_hash to the hashed pages dictionary.
    sim_hashes[current_id] = hashed_value
    return False

