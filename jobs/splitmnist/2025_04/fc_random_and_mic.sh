#!/bin/bash

# FC-net, 3 layers, MC dropout
# random
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=random  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.001   acquisition.method=random  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
# MIC
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.01    acquisition.method=mic     acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_mcdo_3layer    trainer=pytorch_neural_net_classif_mcdo     trainer.optimizer.lr=0.001   acquisition.method=mic     acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True

# FC-net, 1 layer, Laplace approximation
# random
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_1layer         trainer=pytorch_neural_net_classif_la       trainer.optimizer.lr=0.01    acquisition.method=random  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_1layer         trainer=pytorch_neural_net_classif_la       trainer.optimizer.lr=0.001   acquisition.method=random  acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
# MIC
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_1layer         trainer=pytorch_neural_net_classif_la       trainer.optimizer.lr=0.01    acquisition.method=mic     acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
python  cal.py  --multirun  rng.seed=range\(10\)    experiment_name=EmbSplitMNIST_FC    data=splitmnist/embedding_curated   model=pytorch_fc_net_1layer         trainer=pytorch_neural_net_classif_la       trainer.optimizer.lr=0.001   acquisition.method=mic     acquisition.n_train_labels_end=20,50,100    eval_every_acq_step=True
