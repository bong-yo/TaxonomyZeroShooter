import copy
import pandas as pd
import logging
from src.zero_shooter import TaxZeroShot
from src.few_shot.modeling import FewShotTrainer
from src.few_shot.select_FS_examples import FewShotData, ExampleFewShot
from src.few_shot import visuals
from src.dataset import WebOfScience
from src.utils import seed_everything
from globals import Paths

logger = logging.getLogger('few-shot_TC')


def save_results(res: dict, filename: str) -> None:
    '''Append as line to pd.DataFrame if already thede otherwise create it.'''
    try:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([res])
    df.to_csv(filename, index=False)


def load_data_and_model(n_train: int, n_valid: int, use_precomputed: bool,
                        freeze_zstc: bool, freeze_usp: bool
                        ) -> tuple[TaxZeroShot, WebOfScience, WebOfScience]:
    # Get data.
    train_data = WebOfScience('train', topn=n_train, use_precomputed_embeddings=True)
    valid_data = WebOfScience('valid', topn=n_valid, use_precomputed_embeddings=use_precomputed)
    # Load model for Tax Zero-Shot.
    tax_zero_shooter = TaxZeroShot(
        taxonomy=train_data.tax_tree,
        compute_label_thresholds=False,
        label_thresholds_file=f'{Paths.SAVE_DIR}/label_alphas_WebOfScience.json',
        freeze_zstc=freeze_zstc,
        freeze_usp=freeze_usp
    )
    return tax_zero_shooter, train_data, valid_data


def select_fs_examples(n_shots: int, tax_zero_shooter: TaxZeroShot,
                       train_data: WebOfScience,
                       valid_data: WebOfScience,
                       min_entropy: float = 0.7, max_entropy: float = 0.95,
                       ) -> tuple[list[ExampleFewShot], list[ExampleFewShot]]:
    # Get examples for few-shot training.
    few_shot_data = FewShotData(tax_zero_shooter)
    examples_train = few_shot_data.select_examples(
        train_data, min_entropy=min_entropy, max_entropy=max_entropy,
        n_shots=n_shots
    )
    # Evaluation examples.
    examples_valid = [
        ExampleFewShot(
            text=valid_data.abstracts[i],
            labels=[valid_data.Y[j][i] for j in range(valid_data.tax_depth)]
        )
        for i in range(valid_data.n_data)
    ]
    return examples_train, examples_valid


def fewshots_finetuning(tax_zero_shooter: TaxZeroShot,
                        labels_to_consider: list[str],
                        examples_train: list[ExampleFewShot],
                        examples_valid: list[ExampleFewShot],
                        n_shots: int, lr_zstc: float, lr_usp: float,
                        n_epochs: int, freeze_zstc: bool, freeze_usp: bool
                        ) -> dict:
    logger.info(
        '\n------ Run Parameters -------\n'
        f'n_shots: {n_shots}, n_epochs: {n_epochs}\n'
        f'lr_zstc: {lr_zstc}, lr_usp: {lr_usp}\n'
        f'freeze_zstc: {freeze_zstc}, freeze_usp: {freeze_usp}'
    )
    labels_train = set([example.labels[0] for example in examples_train])
    fs_trainer = FewShotTrainer(
        labels_all=labels_to_consider,
        labels_train=labels_train
    )
    # Evaluate
    # fs_trainer.evaluate(tax_zero_shooter, examples_valid)
    tax_zero_shooter = fs_trainer.train(tax_zero_shooter, examples_train,
                                        examples_valid, lr_zstc=lr_zstc,
                                        lr_usp=lr_usp, n_epochs=n_epochs)
    res = fs_trainer.evaluate(tax_zero_shooter, examples_valid)
    res['n_shots'] = n_shots
    res['lr_zstc'] = lr_zstc
    res['lr_usp'] = lr_usp
    res['n_epochs'] = n_epochs
    res['freeze_zstc'] = freeze_zstc
    res['freeze_usp'] = freeze_usp
    return res


if __name__ == "__main__":
    seed_everything(111)
    FREEZE_ZTSC = True
    FREEZE_USP = False
    LRS_USP = [10, 1, 0.1, 0.01]
    LRS_ZSTC = [0.0]
    EPOCHS = [1, 2, 3, 5]
    SHOTS = [10, 40, 200]

    # Load data and model.
    tax_zero_shooter, train_data, valid_data = \
        load_data_and_model(n_train=None, n_valid=None, freeze_zstc=FREEZE_ZTSC,
                            freeze_usp=FREEZE_USP, use_precomputed=FREEZE_ZTSC)
    labels_to_consider = train_data.labels_levels[0]
    for n_shots in SHOTS:
        examples_train, examples_valid = \
            select_fs_examples(n_shots, tax_zero_shooter, train_data, valid_data)
        for lr_zstc in LRS_ZSTC:
            for lr_usp in LRS_USP:
                for n_epochs in EPOCHS:
                    model = copy.deepcopy(tax_zero_shooter)
                    res = fewshots_finetuning(
                        model, labels_to_consider, examples_train,
                        examples_valid, n_shots=n_shots, lr_zstc=lr_zstc,
                        lr_usp=lr_usp, n_epochs=n_epochs, freeze_zstc=FREEZE_ZTSC,
                        freeze_usp=FREEZE_USP
                    )
                    logger.info(res)
                    logger.info(f'{Paths.RESULTS_DIR}/fewshot_results_new.csv')
                    save_results(res, f'{Paths.RESULTS_DIR}/fewshot_results_new.csv')

    # Plot results.
    res = pd.read_csv(f'{Paths.RESULTS_DIR}/fewshot_results_new.csv')
    visuals.plot_usp_results(res)
    visuals.plot_zstc_results(res)
    visuals.plot_usp_plus_zstc_results(res)
