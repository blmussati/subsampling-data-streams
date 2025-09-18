#!/bin/bash

python  cal.py  --multirun  rng.seed=range\(10\)  experiment_name=EmbSplitCIFAR10_RF_dinov2 data=splitcifar10/embedding_curated   model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=la_epig  acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)  experiment_name=EmbSplitCIFAR10_RF_dinov2 data=splitcifar10/embedding_curated   model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=random   acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)  experiment_name=EmbSplitCIFAR10_RF_dinov2 data=splitcifar10/embedding_curated   model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=mic      acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)  experiment_name=EmbSplitCIFAR10_RF_dinov2 data=splitcifar10/embedding_curated   model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=epig     acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
