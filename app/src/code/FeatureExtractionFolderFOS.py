import os
import cv2
import numpy as np
import pandas as pd

from typing import Tuple

from SortImages import SortImages
from FeatureExtraction import FeatureExtraction

from FeaturesFOS import FeaturesFOS

class FeatureExtractionFolderFOS(FeatureExtraction):
    """
    A class for extracting first-order features from a folder of images.

    Parameters
    ----------
    _label : str
        A label for the images in the folder.
    folder : str
        The path to the folder containing the images.

    Attributes
    ----------
    __folder : str
        The path to the folder containing the images.

    Methods
    -------
    extractor() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        Extracts first-order features from a folder of images.

    """
     
    # * Initializing (Constructor)
    def __init__(
        self, 
        _label,
        folder
    ) -> None:
        """
        Initializes a FeatureExtractionFolderFOS object.

        Parameters
        ----------
        _label : str
            A label for the images in the folder.
        folder : str
            The path to the folder containing the images.
        """

        super().__init__(
            _label
        );

        self.__folder = folder;

    def extractor(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        """
        Extracts first-order features from a folder of images.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, str]
            A tuple containing the following:
                1. Dataframe: A pandas dataframe containing the features and labels of the images.
                2. X: A numpy ndarray containing only the features of the images.
                3. Y: A numpy ndarray containing only the labels of the images.
                4. FOF_: A string representing the first-order features.
        """
        
        # * data list to store the features
        data = []

        # * First order tag
        FOF_ = 'First_Order_Features'

        # * Using sort function
        sorted_files, total_images = SortImages.sort_images(self.__folder)
        count = 1

        # * Reading the files
        for file in sorted_files:

            try:
                
                # * split file
                filename, format  = os.path.splitext(file)
                
                print(f'Working with {count} of {total_images} images, {filename} -------- {format} ✅')
                count += 1

                # * Reading the image
                Path_File = os.path.join(self.__folder, file)
                Image = cv2.imread(Path_File)
                
                # * Extracting the first order features from the fos function
                features = FeaturesFOS.calculate_features(Image)

                # * Add new key called label and its value
                features['label'] = self._label;

                # * Append features in data list
                data.append(features)
            
                count += 1
                
            except OSError as e:
                print(f'Error: {e} ❌') #! Alert

        # * Return the new dataframe with the new data
        Dataframe = pd.DataFrame.from_dict(data)

        # * Return a dataframe with only the data without the labels
        X = Dataframe.iloc[:, :].values

        # * Return a dataframe with only the labels
        Y = Dataframe.iloc[:, -1].values

        # * Return the three dataframes
        return Dataframe, X, Y, FOF_