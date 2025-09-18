#!/bin/bash

python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_RF data=splitmnist/embedding_curated   data.label_counts_main.target=0  model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=random  acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(3,10\)  experiment_name=EmbSplitMNIST_RF data=splitmnist/embedding_curated   data.label_counts_main.target=0  model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=mic  acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(3,10\)  experiment_name=EmbSplitMNIST_RF data=splitmnist/embedding_curated   model=cl_random_forest_classif  trainer=sklearn_random_forest_classif  acquisition.method=la_epig  acquisition.n_train_labels_end=20,50,100  eval_every_acq_step=True
