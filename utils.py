import numpy as np
import torch
import os
from datetime import datetime as dt
import shutil
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import math

# Converts "arr" to numpy array and saves it as text
def write_array(arr, output_file=None):
    np_array = np.array(arr).round(decimals=3)
    if output_file == None:
        output_file = "test.txt"
    # Add each row to the text file
    #with open(output_file, "w") as writer:
    #    writer.truncate(0)
    #    for row in np_array:
    #        writer.write(str(row) + "\n")
    np.savetxt(output_file, np_array, fmt="%.3f")
    return

# Creates a one hot encoding from an array of objects
def generate_one_hot_encoding(arr):
    # Determine number of unique objects
    np_arr = np.array(arr)
    unique_objects = np.unique(np_arr)
    # Create mapping between object and one hot encoding 
    one_hot_mapping = {}
    for index, unique_obj in enumerate(unique_objects):
        one_hot_mapping[unique_obj] = index
    # Create one enoding of arr
    one_hot_encoding = np.zeros((len(arr), len(unique_objects)))
    for index, obj in enumerate(arr):
        one_hot_index = one_hot_mapping[obj]
        one_hot_encoding[index, one_hot_index] = 1  
    return one_hot_encoding

# Converts chain sequences in a dictionary into one sequence for MODELLER
# Params:
# seq_dict (dict): dictionary where chain ids are keys and sequences are values
# header (str): the user specified header of FASTA sequence
# pretty (bool): flag to place line breaks every 75 characters
def sum_seq_in_dict (seq_dict, header="", pretty=True):
    result = ""
    # Add up all the chain sequences
    for key in seq_dict:
        result += seq_dict[key]
    # Remove trailing "\" char
    result = result[:-1] + "*"
    # Insert line breaks if pretty
    if pretty:
        result = insert_linebreaks (result)
    return header + result

# Inserts line breaks at specified intervals
# Params:
# string (string): the input string
# interval (int): the interval to insert line breaks 
def insert_linebreaks(string, interval=75):
    lines = []
    for i in range (0, len(string), interval):
        lines.append (string[i:i+interval])
    return '\n'.join(lines)

# Replaces a char in a string at given index
def str_replace (old_string, new_string, index):
    if index <= 0:
        return new_string + old_string
    if index >= len(old_string):
        return old_string + new_string
    
    return old_string[:index] + new_string + old_string[index + 1:]

# Insert a char
def str_insert (old_string, new_string, index):
    if index <= 0:
        return new_string + old_string
    if index >= len(old_string):
        return old_string + new_string
    
    return old_string[:index] + new_string + old_string[index:]

# Prints the parameters of a model
def print_params(model):
    i = 0
    print("Parameter List:")
    for name, param in model.named_parameters():
            trained = "NOT be trained"
            if param.requires_grad:
                trained = "be trained"
            print ("Param ", name, " will", trained)
            i += 1
    print ("There are ", i, "parameters")

# Implements gaussian cross entropy
def gauss_cross_entropy(mu1, var1, mu2, var2):
    term0 = 1.8378770664093453
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy

# Checks if the tensor has a nan value
def check_nan (tensor):
    if(torch.sum (torch.isnan (tensor)) > 0):
        return True
    return False 

if __name__ == '__main__':
    print ("No compile errors")