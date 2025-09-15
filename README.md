This repo contains code for two papers:


### [Prediction-oriented Subsampling from Data Streams (CoLLAs 2025)](https://www.arxiv.org/pdf/2508.03868)

*Benedetta Lavinia Mussati\*, Freddie Bickford Smith\*, Tom Rainforth, Stephen Roberts*

Data is often generated in streams, with new observations arriving over time. A key challenge for learning models from data streams is capturing relevant information while keeping computational costs manageable. We explore intelligent data subsampling for offline learning, and argue for an information-theoretic method centred on reducing uncertainty in downstream predictions of interest. Empirically, we demonstrate that this prediction-oriented approach performs better than a previously proposed information-theoretic technique on two widely studied problems. At the same time, we highlight that reliably achieving strong performance in practice requires careful model design.

## Getting set up

Clone the repo and move into it:

```bash
git clone https://github.com/blmussati/subsampling-data-streams.git && cd epig
```

Create an environment using [Mamba](https://mamba.readthedocs.io) (or [Conda](https://conda.io), replacing `mamba` with `conda` below) and activate it:

```bash
mamba env create --file environment_cuda.yaml && mamba activate epig
```


## Running active learning

Run active learning with the default config:

```bash
python main.py
```

See [`jobs/`](/jobs/) for the commands used to run the continual learning experiments in the papers.

For the models comprising an encoder and a prediction head, the encoder is fixed and deterministic. 
Thus, we can compute the encoders' embeddings of all our inputs once up front and then save them to storage.
These embeddings just need to be moved into `data/` within this repo, and can be obtained from [`msn-embeddings`](https://github.com/fbickfordsmith/msn-embeddings.git), [`simclr-embeddings`](https://github.com/fbickfordsmith/simclr-embeddings.git), [`ssl-embeddings`](https://github.com/fbickfordsmith/ssl-embeddings.git) and [`vae-embeddings`](https://github.com/fbickfordsmith/vae-embeddings.git).


## Getting in touch

Contact [Benedetta](https://github.com/blmussati) if you have any questions about this research or encounter any problems using the code.
This repo is a partial release of a bigger internal repo, and it's possible that errors were introduced when preparing this repo for release.


## Citing this work

```bibtex
@article{MussatiBSRR25,
  author       = {Benedetta Lavinia Mussati and
                  Freddie Bickford Smith and
                  Tom Rainforth and
                  Stephen Roberts},
  title        = {Prediction-Oriented Subsampling from Data Streams},
  journal      = {Conference on Lifelong Learning Agents},
  volume       = {abs/2508.03868},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2508.03868},
  doi          = {10.48550/ARXIV.2508.03868},
  eprinttype    = {arXiv},
  eprint       = {2508.03868},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2508-03868.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


## Contributors

[Freddie Bickford Smit](https://github.com/fbickfordsmith) wrote the original EPIG work in active learning and advised on the code for this project.

Credit for the unsupervised encoders we use in our semi-supervised models goes to the authors of [`disentangling-vae`](https://github.com/YannDubs/disentangling-vae), [`lightly`](https://github.com/lightly-ai/lightly), [`msn`](https://github.com/facebookresearch/msn) and [`solo-learn`](https://github.com/vturrisi/solo-learn), as well as the designers of the pretraining methods used.
