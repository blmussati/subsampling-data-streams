from typing import Any, List, Tuple
import logging
from datetime import timedelta
from time import time

import torch
from torch.distributions import Gumbel
from torch.utils.data import Subset, DataLoader, TensorDataset
import numpy as np
from numpy.random import Generator
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate, call
from pathlib import Path
import hydra
import os
import wandb


from src.device import get_device
from src.random import get_rng
from src.trainers.base import Trainer
from src.trainers.pytorch import PyTorchTrainer
from src.data.active_learning import ActiveLearningData
from src.logging import (
    Dictionary, 
    get_formatters, 
    prepend_to_keys, 
    set_up_wandb,
    save_repo_status,
    save_run_to_wandb
    )


config_dir = Path(__file__).parent / "config"


def get_pytorch_trainer(data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str) -> PyTorchTrainer:
    """
    Adaptation from EPIG/main.py. 
    Output_size relise on cfg.continual_learning.n_classes
    """
    input_shape = data.main_dataset.input_shape
    output_size = cfg.continual_learning.n_classes

    model = instantiate(cfg.model, input_shape=input_shape, output_size=output_size)
    model = model.to(device)

    seed = rng.choice(int(1e6))
    torch_rng = torch.Generator(device).manual_seed(seed)

    return instantiate(cfg.trainer, model=model, torch_rng=torch_rng)


def get_sklearn_trainer(cfg: DictConfig) -> Trainer:
    n_classes = cfg.continual_learning.n_classes

    model = instantiate(cfg.model, n_classes=n_classes)
    return instantiate(cfg.trainer, model=model)


def acquire_using_random(data: ActiveLearningData, cfg: DictConfig, rng: Generator) -> List[int]:
    n_pool = len(data.main_inds["pool"])
    return rng.choice(n_pool, size=cfg.acquisition.batch_size, replace=False).tolist()


def acquire_using_uncertainty(
    data: ActiveLearningData, cfg: DictConfig, rng: Generator, device: str, trainer: Trainer
) -> List[int]:
    """
    Adapted from EPIG/main.py to retain only relevant parts for continual learning experiments.
    """
    seed = rng.choice(int(1e6))
    acq_kwargs = dict(
        loader=data.get_loader("pool"), method=cfg.acquisition.method, seed=seed
    )

    use_epig_with_target_class_dist = (
        cfg.acquisition.epig.classification.target_class_dist is not None
    )

    if ("epig" in cfg.acquisition.method) and (not use_epig_with_target_class_dist):
        target_loader = data.get_loader("target")
        target_inputs, _ = next(iter(target_loader))
        acq_kwargs = dict(inputs_targ=target_inputs, **acq_kwargs)

    with torch.inference_mode():
        scores = trainer.estimate_uncertainty(**acq_kwargs)

    acquired_pool_inds = torch.argsort(scores, descending=True)[:cfg.acquisition.batch_size]
    acquired_pool_inds = acquired_pool_inds.tolist()

    return acquired_pool_inds


def acquire_using_mic(
    data: ActiveLearningData, cfg: DictConfig, trainer: Trainer, coresets_lst: List[Tuple[torch.Tensor, torch.Tensor]]
) -> List[int]:
    # We don't need a target set when acquiring using MIC, only the pool set and memory
    memory = data.get_memory(coresets=coresets_lst)    # memory = (train_inputs, train_labels)
    acq_kwargs = dict(
        loader=data.get_loader("pool"), 
        memory=memory,
        mic_kwargs=cfg.acquisition.mic.mic_kwargs
    )

    with torch.inference_mode():
        scores = trainer.compute_mic(**acq_kwargs)

    if cfg.acquisition.batch_size == 1:
        # acquired_pool_inds = [torch.argmax(scores).item()]
        acquired_pool_inds = [np.argmax(scores)]
    else:
        raise NotImplementedError

    return acquired_pool_inds


def split_classes_equally(cfg):
    """
    Splits a 1D numpy array into d equal groups.
    
    Parameters:
    - n_classes (int): The tot number of classes.
    - n_experiences (int): The number of experiences to split the classes into.
    
    Returns:
    - list of np.ndarray: A list containing d equally-sized numpy arrays.
    
    Raises:
    - ValueError: If n_experiences is not an exact divisor of n_classes.
    """
    n_classes = cfg.n_classes
    n_experiences = cfg.n_experiences
    shuffle = cfg.shuffle_classes

    if n_classes % n_experiences != 0:
        raise ValueError(f"The number of experiences (n_experiences={n_experiences}) must be an exact divisor of the array length (n_classes={n_classes}).")
    
    classes = range(n_classes)
    if shuffle:
        classes = np.random.permutation(classes)

    classes_per_exp = np.array_split(classes, n_experiences)    # list[np.ndarray]
    return classes_per_exp


def check_optim_steps_max(cfg: DictConfig):
    """
    The minimum number of optimisation steps required to use each training point at least once is 
    ((data.n_train_labels + (M*experience)) // bs)+1
    For this to hold at every step, we check 
    n_optim_steps_max > (M*n_experiences//bs_train)
    """
    bs_train = cfg.data.batch_sizes.train
    if bs_train > 0:
        n_experiences = cfg.continual_learning.n_experiences
        M = cfg.acquisition.n_train_labels_end
        n_optim_steps_max = cfg.trainer.n_optim_steps_max
        assert n_optim_steps_max > (M*n_experiences//bs_train), f"n_optim_steps_max should be at least {(M*n_experiences//bs_train)+1}"


@hydra.main(version_base=None, config_path=str(config_dir), config_name="main_cal")
def main(cfg: DictConfig) -> None:
    """
    Adapted from main.py to retain only relevant parts for CL experiments. 
    Modified to have the continual active learning loop.
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", default=None)  # None if not running in Slurm

    device = get_device(cfg.use_gpu)
    rng = call(cfg.rng)
    formatters = get_formatters()

    if cfg.use_gpu and (device not in {"cuda", "mps"}):
        logging.warning(f"Device {device}")
    else:
        logging.info(f"Device: {device}")
    
    logging.info(f"Seed: {cfg.rng.seed}")
    logging.info(f"Making results dirs at {cfg.directories.results_run}")

    if cfg.model_type != "sklearn":
        check_optim_steps_max(cfg)

    results_dir = Path(cfg.directories.results_run)

    for subdir in cfg.directories.results_subdirs[cfg.model_type]:
        Path(subdir).mkdir(parents=True, exist_ok=True)
    
    save_repo_status(results_dir / "git")

    if cfg.wandb.use:
        set_up_wandb(cfg, slurm_job_id)

    # ----------------------------------------------------------------------------------------------
    classes_per_experience = split_classes_equally(cfg.continual_learning)

    logging.info(f"Number of classes: {cfg.continual_learning.n_classes} \t Number of experiences: {cfg.continual_learning.n_experiences} \t Classes per experience: {classes_per_experience}")
    logging.info(f"Data configs {cfg.data}")

    coresets_lst = []
    coresets_inds_lst = []
    test_log = Dictionary()

    start_time = time()

    logging.info("Starting continual learning")
    logging.info(f"Acquiring {cfg.acquisition.n_train_labels_end} train points for each coreset using {cfg.acquisition.method}.")
    for experience, _classes in enumerate(classes_per_experience):
        logging.info(f"CL task {experience}, classes {_classes}")
        data = instantiate(cfg.data, rng=rng, device=device, dataset={"classes_per_exp": _classes})
        data.convert_datasets_to_torch()

        # ----------------------------------------------------------------------------------------------
        logging.info(f"{int(cfg.acquisition.n_train_labels_end / cfg.acquisition.batch_size)} active learning steps")
        logging.info("Starting active learning")
        is_first_al_step = True
        al_step = 0
        if cfg.eval_every_acq_step:
            if cfg.wandb.use:
                wandb.define_metric(f"task{experience}_train_labels", hidden=True)
                wandb.define_metric(f"task{experience}_*", step_metric="task{experience}_train_labels")
            task_test_log = Dictionary()

        while True:
            n_labels_str = f"{data.n_train_labels:04}_labels"
            is_last_al_step = data.n_train_labels >= cfg.acquisition.n_train_labels_end

            # ----------------------------------------------------------------------------------------------
            if cfg.model_type == "pytorch":
                trainer = get_pytorch_trainer(data, cfg, rng, device)

            elif cfg.model_type == "sklearn":
                trainer = get_sklearn_trainer(cfg=cfg)
            
            elif cfg.model_type == "gpytorch":
                raise NotImplementedError
            
            else:
                raise ValueError(f"Unrecognized model type: {cfg.model_type}")

            if data.n_train_labels > 0:
                # ----------------------------------------------------------------------------------------------
                if experience > 0:
                    train_inputs, train_labels = data.adapt_dataset(coresets=coresets_lst)
                    if data.batch_sizes["train"] == -1:
                        batch_size = len(train_inputs)
                    else:
                        # To handle StopIteration error in utils.get_next_batch() when the training batch_size is larger than the available datapoints
                        batch_size = min(data.batch_sizes["train"], len(train_inputs))
                    train_loader = DataLoader(
                        dataset=TensorDataset(train_inputs, train_labels),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                else:
                    train_loader = data.get_loader(subset="train")

                # Check that valset is non-empty
                if trainer.use_val_data and data.main_inds['val']:
                    val_loader = data.get_loader("val")
                    train_step, train_log = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader
                    )
                else:
                    train_step, train_log = trainer.train(
                        train_loader=train_loader
                    )
                
                if train_step is not None:
                    if train_step < cfg.trainer.n_optim_steps_max - 1:
                        logging.info(f"Training stopped early at step {train_step}")
                    else:
                        logging.warning(f"Training stopped before convergence at step {train_step}")
                
                if train_log is not None:
                    train_log.save_to_csv(results_dir / "training" / f"{n_labels_str}.csv", formatters)
            
            # Save pytorch model
            is_in_save_steps = data.n_train_labels in cfg.model_save_steps
            model_dir_exists = (results_dir / "models").exists()

            if (is_first_al_step or is_last_al_step or is_in_save_steps) and model_dir_exists:
                model_state = trainer.model.state_dict()
                torch.save(model_state, results_dir / "models" / f"{n_labels_str}.pth")

            if cfg.eval_every_acq_step:
                if cfg.adjust_test_predictions:
                    test_labels = data.test_dataset.targets[data.test_inds]
                    test_kwargs = dict(n_classes=len(torch.unique(test_labels)))
                else:
                    test_kwargs = {}
                with torch.inference_mode():
                    task_test_metrics = trainer.test(data.get_loader("test"), **test_kwargs)
                
                task_test_metrics_str = ", ".join(
                    f"{key} = {formatters[f'test_{key}'](value)}" for key, value in task_test_metrics.items()
                )

                logging.info(f"Task {experience}, classes {_classes}, {data.n_train_labels} train points. \nTest metrics: {task_test_metrics_str}")

                testing_task_dir = results_dir.joinpath("testing", f"task{experience}")
                testing_task_dir.mkdir(parents=True, exist_ok=True)

                task_test_log.append({"train_labels": data.n_train_labels, **prepend_to_keys(task_test_metrics, f"test")})
                task_test_log.save_to_csv(testing_task_dir / "testing.csv", formatters)

                if cfg.wandb.use:
                    wandb.log({f"task{experience}_{key}": values[-1] for key, values in task_test_log.items()})
            
            if is_last_al_step:
                logging.info("Stopping active learning")
                break
            
            # ----------------------------------------------------------------------------------------------
            uncertainty_types = {
                "epig",
                "la_epig",
                "mic",
            }

            if cfg.acquisition.method == "random":
                acquired_pool_inds = acquire_using_random(data, cfg, rng)

            elif cfg.acquisition.method in uncertainty_types:
                acquired_pool_inds = acquire_using_uncertainty(data, cfg, rng, device, trainer)
            
            elif cfg.acquisition.method == "mic_linear":
                acquired_pool_inds = acquire_using_mic(data, cfg, trainer, coresets_lst)

            else:
                raise ValueError(f"Unrecognized acquisition method: {cfg.acquisition.method}")

            data.move_from_pool_to_train(acquired_pool_inds)
            is_first_al_step = False
            al_step += 1

        # ----------------------------------------------------------------------------------------------
        # Save indices at the end of active learning so that indices that have been selected to be moved to train no longer show up in `pool`
        data_indices_task_dir = results_dir.joinpath("data_indices", f"task{experience}")
        data_indices_task_dir.mkdir(parents=True, exist_ok=True)
        for subset, inds in data.main_inds.items():
            np.savetxt(data_indices_task_dir / f"{subset}.txt", inds, fmt="%d")

        # Save the test indices only if we use a subset of them
        if not (isinstance(cfg.data.label_counts_test, int)) or (cfg.data.label_counts_test != -1):
            np.savetxt(data_indices_task_dir / "test.txt", data.test_inds, fmt="%d")

        # ----------------------------------------------------------------------------------------------
        # Compute performance on evaluation set at the end of each CL task 

        if cfg.adjust_test_predictions:
            test_labels = data.test_dataset.targets[data.test_inds]
            test_kwargs = dict(n_classes=len(torch.unique(test_labels)))
        else:
            test_kwargs = {}
        with torch.inference_mode():
            test_metrics = trainer.test(data.get_loader("test"), **test_kwargs)
        
        test_metrics_str = ", ".join(
            f"{key} = {formatters[f'test_{key}'](value)}" for key, value in test_metrics.items()
        )

        logging.info(f"Test metrics: {test_metrics_str}")

        test_log.append({"Task": experience, "Classes": _classes, **prepend_to_keys(test_metrics, "test")})
        test_log.save_to_csv(results_dir / "testing.csv", formatters)

        if cfg.wandb.use:
            wandb.log({key: values[-1] for key, values in test_log.items()
                       if key not in ['Classes']})

        # ----------------------------------------------------------------------------------------------
        train_inds = data.main_inds['train']
        # Move torch tensors to CPU before saving them
        coreset = (data.main_dataset.data[train_inds].cpu(), data.main_dataset.targets[train_inds].cpu())
        coresets_lst.append(coreset)
        coresets_inds_lst.append(data.main_dataset.dataset_inds[train_inds])
    
    # ----------------------------------------------------------------------------------------------
    if cfg.continual_learning.save_coresets:
        coresets_dir = results_dir.joinpath("coresets")
        coresets_fname = coresets_dir.joinpath(f"coresets_{n_labels_str}.pth")
        coresets_inds_fname = coresets_dir.joinpath(f"coresets_train_inds_{n_labels_str}.txt")

        logging.info(f"Saving coresets and train_inds at {coresets_dir}")
        # Save list of coresets, where coreset = (datapoints, labels) in a single file
        torch.save(coresets_lst, coresets_fname)

        with open(coresets_inds_fname, "w") as f:
            for coreset_ind_lst in coresets_inds_lst:
                f.write(" ".join(map(str, coreset_ind_lst)) + "\n")

    
    run_time = timedelta(seconds=(time() - start_time))
    np.savetxt(results_dir / "run_time.txt", [str(run_time)], fmt="%s")

    if cfg.wandb.use:
        save_run_to_wandb(results_dir, cfg.directories.results_subdirs[cfg.model_type])
        wandb.finish()  # Ensure each run in a Hydra multirun is logged separately


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()