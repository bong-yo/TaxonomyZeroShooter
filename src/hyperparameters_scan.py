import logging
import os
import torch
from src.utils import FileIO

from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

logger = logging.getLogger("victims")


def run_hyperparameter_scan(config: Config) -> None:
    lr_zstc = 1e-4
    lr_usp = 4
    freeze_zstc = True
    freeze_usp = False
    n_epochs=[2]
    n_shots=[10, 40, 100]
    hyper_space = {
        "lr_zstc": tune.loguniform(1e-6, 1e-3),
        "lr_usp": tune.uniform(1e-1, 20),
        "n_epochs": tune.randint([0, 10]),

        'trans_hidden_size': tune.randint(40, 140),
        'depth': tune.randint(3, 12),
        'dim_feedforward': tune.randint(400, 1200),
        'n_attn_heads': tune.randint(4, 12),
        'max_seq_len': tune.randint(15, 29),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
        "p_dropout": tune.choice([0.15]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.randint(1, 4),
    }

    gpus_per_trial = config.hypertune.gpus_per_trial
    max_num_epochs = config.hypertune.max_num_epochs
    min_num_epochs = config.hypertune.min_num_epochs
    num_samples = config.hypertune.num_samples

    ray.init(configure_logging=True, logging_level=logging.ERROR)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=min_num_epochs,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns={'loss': 'mae', "training_iteration": 'iter'},
        sort_by_metric=True,
        max_report_frequency=60,
        metric='loss',
        mode='min',
        parameter_columns={
            'batch_size': 'b_size', 'depth': 'depth', 'dim_feedforward': 'ff_size',
            'hidden_categorical_sizes': 'hc_sizes', 'hidden_numerical_size': 'hn_size', 
            'trans_hidden_size': 'emb_size', 'lr': 'lr', 'max_seq_len': 'seq_len',
            'n_attn_heads': 'n_heads', 'p_dropout': 'dropout',
            'weight_decay': 'w_decay'
        }
    )

    hyper_search = HyperOptSearch(hyper_space, metric="loss", mode='min')
    result = tune.run(
        partial(train_hypersearch, config_static=config),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=hyper_search,
        progress_reporter=reporter,
        checkpoint_at_end=True
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))


    # Save best model.
    config.trainer.weight_decay = best_trial.config["weight_decay"]
    config.trainer.lr = best_trial.config["lr"]
    config.trainer.p_dropout = best_trial.config["p_dropout"]
    config.trainer.batch_size = best_trial.config["batch_size"]
    config.model.vf.hidden_categorical_sizes = best_trial.config["hidden_categorical_sizes"]
    config.model.vf.hidden_numerical_size = best_trial.config["hidden_numerical_size"]
    config.model.vf.trans_hidden_size = best_trial.config["trans_hidden_size"]
    config.model.vf.depth = best_trial.config['depth']
    config.model.vf.dim_feedforward = best_trial.config['dim_feedforward']
    config.model.vf.n_attn_heads = best_trial.config['n_attn_heads']
    config.model.vf.max_seq_len = best_trial.config['max_seq_len']
    savefolder = set_savefolder(config)

    data = Dataset(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        preprocess=config.dataset.preprocess,
        build_geotemp_grids=config.dataset.build_grid,
        n_x=config.grid.nx,
        n_y=config.grid.ny,
        delta_t=config.grid.delta_t,
        topn=config.dataset.topn
    )

    best_trained_model = Model.get(
        model_type=config.model.type,
        input_numerical_size=data.size_numerical,
        input_categ_sizes=data.sizes_categorical,
        emb_numeric_size=config.model.vf.hidden_numerical_size,
        emb_categ_size=config.model.vf.hidden_categorical_sizes,
        trans_hidden_size=config.model.vf.trans_hidden_size,
        feedforward_size=config.model.vf.dim_feedforward,
        depth=config.model.vf.depth,
        n_heads=config.model.vf.n_attn_heads,
        max_seq_len=config.model.vf.max_seq_len,
        mlp_posembs_size=config.model.mlp.positional_embeddings_size,
        mlp_hidden_zises=config.model.mlp.hidden_sizes,
        out_size=data.n_crimes,
        p_dropout=config.trainer.p_dropout,
        device=config.device
    )
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    best_trained_model.save(savefolder)
    FileIO.write_json(config.dict(), f'{savefolder}/configs.json')


    # Run test on best model.
    test(config, savefolder)
