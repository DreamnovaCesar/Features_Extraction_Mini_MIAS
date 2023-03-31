import numpy as np
from typing import Dict
from Features import Features

class FeaturesFOS(Features):
    """
    A class for extracting various features from an input image and mask.
    
    Parameters
    ----------
    image : numpy.ndarray
        NumPy array representing an image with dimensions N1 x N2.
    mask : numpy.ndarray, optional. Defaults to None.
        NumPy array representing a mask image with dimensions N1 x N2. 
        Each pixel in the mask should be assigned a value of 1 if it belongs to the Region of Interest (ROI), and 0 otherwise. 
        If you want to consider the entire image as the ROI, you can give None as the value for the mask parameter.
    
    Returns
    -------
    Dict
        Dictionary of computed features.

    Attributes
    ----------
    _image : numpy.ndarray
        NumPy array representing an image with dimensions N1 x N2.
    _level_max : int
        Maximum grayscale level of the input image.
    _level_min : int
        Minimum grayscale level of the input image.
    _bins : int
        Number of bins in the image histogram.
    _labels : Tuple
        Tuple of feature labels.
    _mask : numpy.ndarray
        NumPy array representing a mask image with dimensions N1 x N2. 
        Each pixel in the mask should be assigned a value of 1 if it belongs to the Region of Interest (ROI), and 0 otherwise. 
        If you want to consider the entire image as the ROI, you can give None as the value for the mask parameter.
    
    Methods
    -------
    calculate_features():
        Computes various image features from the input image and mask.
    """

    def __init__(
            self, 
            image: np.ndarray, 
            mask: np.ndarray = None
        ):

        self._image = image.astype(np.uint8);
        self._level_max = 255;
        self._level_min = 0;
        self._bins = ((self._level_max - self._level_min) + 1);

        self._labels = ("Mean", "Variance", "Median", "Mode", "Skewness",
                        "Kurtosis", "Energy", "Entropy", "MinimalGrayLevel",
                        "MaximalGrayLevel", "CoefficientOfVariation",
                        "10Percentile", "25Percentile", "75Percentile",
                        "90Percentile","HistogramWidth");
        
        if(mask is None):
            self._mask = np.ones(
                self._image.shape, 
                dtype = np.uint8
            );
        
        else:
            self._mask = mask.astype(
                np.uint8
            );

    def calculate_features(self) -> Dict:
        """
        Computes various image features from the input image and mask.
        
        Returns
        -------
        Dict
            Dictionary of computed features.
        """

        # * Tuple percentiles
        percentile = (10, 25, 50, 75, 90);

        # * Initialize features dictionary
        features_dict = {};
        features_dict.fromkeys(self._labels);

        # * Get image and mask ravel arrays
        image_ravel = self._image.ravel();
        mask_ravel = self._mask.ravel();

        # * Apply mask to image
        roi = image_ravel[mask_ravel.astype(bool)];

        # * Calculate image histogram
        histogram = np.histogram(
            roi, 
            bins = self._bins, 
            range = [self._level_min, self._level_max], 
            density = True
        )[0];
        
        # * Calculate various image features
        list_bins = np.arange(0, self._bins);

        # * Mean
        features_dict[self._labels[0]] = np.dot(list_bins, histogram)

        # * Variance
        features_dict[self._labels[1]] = np.dot(((
            list_bins - features_dict[self._labels[0]]) ** 2), 
            histogram)
            
        # * Median
        features_dict[self._labels[2]] = np.percentile(roi, percentile[2]) 

        # * Mode
        features_dict[self._labels[3]] = np.argmax(histogram)

        # * Skewness
        features_dict[self._labels[4]] = np.dot(((
            list_bins - features_dict[self._labels[0]]) ** 3), histogram) / (np.sqrt(features_dict[self._labels[1]]) ** 3)
        
        # * Kurtosis
        features_dict[self._labels[5]] = np.dot(((
            list_bins - features_dict[self._labels[0]]) ** 4), histogram) / (np.sqrt(features_dict[self._labels[1]]) ** 4)
        
        # * Energy
        features_dict[self._labels[6]] = np.dot(
            histogram, 
            histogram)
        
        # * Entropy
        features_dict[self._labels[7]] = -np.dot(
            histogram, 
            np.log(histogram + 1e-16))
        
        # * MinimalGrayLevel
        features_dict[self._labels[8]] = min(roi)

        # * MaximalGrayLevel
        features_dict[self._labels[9]] = max(roi)

        # * CoefficientOfVariation
        features_dict[self._labels[10]] = np.sqrt(features_dict[self._labels[2]]) / features_dict[self._labels[0]]

        # * 10Percentile
        features_dict[self._labels[11]] = np.percentile(roi, percentile[0]) 

        # * 25Percentile
        features_dict[self._labels[12]] = np.percentile(roi, percentile[1]) 

        # * 75Percentile
        features_dict[self._labels[13]] = np.percentile(roi, percentile[3])

        # * 90Percentile 
        features_dict[self._labels[14]] = np.percentile(roi, percentile[4]) 

        # * HistogramWidth
        features_dict[self._labels[15]] = features_dict[self._labels[14]] - features_dict[self._labels[11]]

        return features_dict
    


