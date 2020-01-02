# Richter's Eye
 DRIVENDATA :: Richter's Predictor: Modeling Earthquake Damage Challenge 8th Place (8/1659) Solution :: 01.01.2020

## Overview 
  Based on aspects of building location and construction, our goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal.

  The data was collected through surveys by the Central Bureau of Statistics that work under the National Planning Commission Secretariat of Nepal. This survey is one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics.

/* <a href="https://www.drivendata.org/competitions/57/nepal-earthquake/" target="_blank">Richter's Predictor: Modeling Earthquake Damage Challenge</a> hosted by <a href="https://www.drivendata.org/" target="_blank">DRIVENDATA</a>. *\ 

## Requirements
- numpy 
- keras 
- pandas
- lightgbm
- tensorflow
- scikit-learn

```
$ pip install -r requirements.txt
```

## The Solution 

### GEO-Embed
Machine Learning Methods are not efficient in classification/regression tasks if "identical data" (ID) given to model as input. So we propose to use autoencoder model to extract valuable information from identical data. We give specific location id ("geo_level_3_id") as input and larger location ids ("geo_level_1_id", "geo_level_2_id") as output, to Keras AutoEncoder Model. There is just one hidden layer and it has 16 neurons. Later then we assign this embedded features to training and testing data.

### 5-Fold Cross Validation Training with LightGBM Model
- LightGBM Model Parameters
```
lgb_params = {
        "objective" : "multiclass",
        "num_class":3,
        "metric" : "multi_error",
        "boosting": 'gbdt',
        "max_depth" : -1,
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "feature_fraction" : 0.5,
        "min_sum_hessian_in_leaf" : 0.1,
        "max_bin":8192,
        "verbosity" : 1,
        "num_threads":6,
        "seed": 1881
    }
```
### Ensemble Models
We ensemble K-Fold(CV) models with adding all confidence score by class. Then apply threshold to ensemble results.

## License 
This project is licensed under the GNU Affero General Public License v3.0 - see the <a href="LICENSE.md">LICENSE.md</a> file for details.

## Contact
E-mail: kutsal_baran@hotmail.com
