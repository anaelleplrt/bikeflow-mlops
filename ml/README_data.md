# BikeFlow - Data documentation

## Source

UCI Machine Learning Repository - Bike Sharing Dataset.

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Original dataset: Capital Bikeshare, Washington DC, 2011-2012.

Citation: Fanaee-T and Gama (2013).
DOI: 10.1007/s13748-013-0040-3.

The source README requests citation when using the dataset
in publications.

## Dataset

We use hour.csv, containing 17,379 hourly observations.

Target: cnt (total number of rentals).

Features:
mnth, hr, weekday, workingday, weathersit,
temp, hum, windspeed.

We exclude casual and registered to prevent target leakage.
We also exclude instant and use dteday only for splitting.

## Data versions

v1:
- Training: January-September 2011
- Validation: October-December 2011

v2:
- Training: January 2011-June 2012
- Validation: July-September 2012
- Final evaluation: October-December 2012

The final evaluation data must not be used for training
or hyperparameter tuning.

The champion and candidate will be compared on the same
final evaluation observations.

## Reproducibility

The datasets are generated using ml/prepare_data.py.

DVC storage: Azure Blob Storage (to be configured).

Dataset hashes and associated Git commits will be added
after DVC versioning.
