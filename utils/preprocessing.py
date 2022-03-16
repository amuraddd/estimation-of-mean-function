import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.testing import data_path

def preprocess_image(img, img_slice=20):
    """
    Input: 
        img: Image as nii file - data cube with slices for an image
        img_slice: integer to specify which slice to select
    Return 
        X: dataframe with two columns containing X and Y coordinates.
        Y: flattened pixel values
    """
    images = nib.load(img)
    data = images.get_fdata().T #transpose the original data - it should fit the format 95*79
    
    Y = data[img_slice].ravel() #flatten the matrix of pixels into a single array
    Y = pd.DataFrame(Y, columns=['pixel_value']) #for the first image
    
    #get the number of rows and columns for the matrix of pixels per image
    rows = data[img_slice].shape[0] #number of rows
    cols = data[img_slice].shape[1] #number of columns

    #generate coordinates
    row_indices = list()
    column_indices = list()

    row_coordinates =  (np.indices(dimensions=(rows, cols))[0]+1)*(1/rows) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize
    column_coordinates =  (np.indices(dimensions=(rows, cols))[1]+1)*(1/cols) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize

    for row in row_coordinates:
        for row_index in row:
            row_indices.append(row_index)

    for row in column_coordinates:
        for column_index in row:
            column_indices.append(column_index)
            
    X = pd.DataFrame(columns=['X_coordinate', 'Y_coordinate'])
    X['X_coordinate'] = np.array(row_indices)
    X['Y_coordinate'] = np.array(column_indices)
    
    return X, Y
