#!/bin/bash

# FC-net, 3 layers, MC dropout
# LA-EPIG
python  cal.py  --multirun  rng.seed=4    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=la_epig  acquisition.n_train_labels_end=20    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=4    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=la_epig  acquisition.n_train_labels_end=50    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(5\)    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=la_epig  acquisition.n_train_labels_end=100    eval_every_acq_step=True

# random
python  cal.py  --multirun  rng.seed=range\(5\)    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=random  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True

# MIC
python  cal.py  --multirun  rng.seed=range\(5\)    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=mic  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True

# EPIG
python  cal.py  --multirun  rng.seed=range\(5\)    experiment_name=EmbSplitMNIST_FC_dinov2    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=epig     acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True

