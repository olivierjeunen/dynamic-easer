# Embarrassingly Shallow Auto-Encoders for Dynamic Collaborative Filtering

## Get the data 

You can download the MovieLens-25M dataset from [their GroupLens website](https://grouplens.org/datasets/movielens/25m/).
We adopt their data format as the standard format for our experiments.
Other recommender systems datasets can be plugged in if they follow the same format as the MovieLens "ratings.csv" file.
Unfortunately, we do not have the rights to re-distribute any other datasets, but all other datasets used in the paper are publicly available.

```
wget http://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
```

## Run Experiments
Now you can run the experiments for a given setup.

```
python3 src/DynEASEr_Runtime.py -data ml-25m/ratings.csv
```

More information on parameters appears in the help message, with can be generated by running `python3 src/DynEASEr_Runtime.py --help`.
