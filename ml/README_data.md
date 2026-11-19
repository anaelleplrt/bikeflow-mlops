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

DVC remote:
- Provider: Azure Blob Storage
- Storage account: bikeflowmlops01
- Container: bikeflow-dvc
- Remote: azure://bikeflow-dvc/datasets

### Dataset v1

Git tag: data-v1

Git commit:
d1dc7cd8a73ba35473a2c1c9ee2f3d80f70763c3

DVC hash:
d0d9e62736306395adff5c47b6043278.dir

Training data: January-September 2011.
Validation data: October-December 2011.

### Dataset v2

Git tag: data-v2

Git commit:
1ceb913364d1c74867469c81586612423b20f38e

DVC hash:
5c0dd375f863b5e603ab9a71a171490c.dir

Training data: January 2011-June 2012.
Validation data: July-September 2012.

### Final evaluation

Period: October-December 2012.

DVC hash:
389e6a13161da90fdd0719229fea4a80.dir

The final evaluation set remains unchanged between
dataset versions.

It must not be used for model training or tuning.

### Restoring data versions

To restore v1:

    git switch --detach data-v1
    dvc pull

To restore v2:

    git switch --detach data-v2
    dvc pull

After checking a dataset version, return to the
development branch before continuing work.

Azure credentials are configured locally and must
never be committed to GitHub.

GitHub Actions will need separate Azure credentials
for continuous training.
