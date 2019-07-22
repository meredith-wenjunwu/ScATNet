# ROI

### List of Files

`main.py`  Main function to call for functions in module

`util.py`  Supporting function for preprocessing data and feature calculation

`bag.py` Iterator Class for constructing the bag of words

`classifier.py` Classifier related function

`cluster.py` Clustering related function

`feature.py` Function for feature calculation

`word.py` Iterator Class for constructing the bag of words

`viz.py` Visualization script

### Data Preprocessing

```python
preprocess_roi_csv(csv_file)
		"""
        Function that process csv files with ROI boxes
        Returns:
            Dict[Case ID]: Bounding box(int[])
            							[h_low, h_high, w_low, w_high]
    """
```

```
crop_saveroi(image_folder, dict_bbox, appendix)
		"""
        Function that crop ROI from large tif image
    """
```

### Feature Computation

To calculate kmeans clusters and features from a list of images in a folder and saved the results in another folder:

```python
python main.py --mode 'kmeans' --image_folder '/projects/medical4/ximing/DistractorProject/page3' --save_path '/projects/medical4/ximing/DistractorProject/feature_page3'
```

