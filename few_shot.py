import pandas as pd
from src.zero_shooter import TaxZeroShot
from src.few_shot.modeling import FewShotTrainer
from src.few_shot.select_FS_examples import FewShotData, ExampleFewShot
from src.dataset import WebOfScience
from src.utils import seed_everything
from globals import Paths


def fewshots_finetuning(n_shots: int, lr_zstc: float, lr_usp: float,
                        n_epochs: int, freeze_zstc: bool, freeze_usp: bool
                        ) -> dict:
    # Only precompute validation embeddings if zst stays the same throughout the training.
    use_precomputed_valid_embs = freeze_zstc
    train_data = WebOfScience('train', topn=200, embeddings_precomputed=True)
    valid_data = WebOfScience('valid', topn=30,
                              embeddings_precomputed=use_precomputed_valid_embs)

    tax_zero_shooter = TaxZeroShot(
        train_data.tax_tree,
        f'{Paths.SAVE_DIR}/label_alphas_WebOfScience.json',
        freeze_zstc=freeze_zstc,
        freeze_usp=freeze_usp
    )

    # Get examples for few-shot training.
    few_shot_data = FewShotData(tax_zero_shooter)
    examples_train = few_shot_data.select_examples(
        train_data, min_entropy=0.7, max_entropy=0.95, n_shots=n_shots
    )
    labels_train = set([example.labels[0] for example in examples_train])

    # Evaluation examples.
    examples_valid = [
        ExampleFewShot(
            text=valid_data.abstracts[i],
            labels=[valid_data.Y[j][i] for j in range(valid_data.tax_depth)]
        )
        for i in range(valid_data.n_data)
    ]

    fs_trainer = FewShotTrainer(
        labels_all=train_data.labels_levels[0],
        labels_train=labels_train
    )
    # Evaluate
    # fs_trainer.evaluate(tax_zero_shooter, examples_valid)
    tax_zero_shooter = fs_trainer.train(tax_zero_shooter, examples_train,
                                        examples_valid, lr_zstc=lr_zstc,
                                        lr_usp=lr_usp, n_epochs=n_epochs)
    res = fs_trainer.evaluate(tax_zero_shooter, examples_valid)
    return res


if __name__ == "__main__":
    seed_everything(111)

    n_shots = 4
    lr_zstc = 1e-4
    lr_usp = 1e-4
    n_epochs = 1
    freeze_zstc = True
    freeze_usp = False

    res = fewshots_finetuning(n_shots=n_shots, lr_zstc=lr_zstc, lr_usp=lr_usp,
                              n_epochs=n_epochs, freeze_zstc=freeze_zstc,
                              freeze_usp=freeze_usp)
    res['n_shots'] = n_shots
    res['lr_zstc'] = lr_zstc
    res['lr_usp'] = lr_usp
    res['n_epochs'] = n_epochs
    res['freeze_zstc'] = freeze_zstc
    res['freeze_usp'] = freeze_usp
    # Append as line to pd.DataFrame if already thede otherwise create it.
    try:
        df = pd.read_csv(f'{Paths.RESULTS_DIR}/few_shot_results.csv')
        df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([res])
    df.to_csv(f'{Paths.RESULTS_DIR}/few_shot_results.csv', index=False)
    print('Done!')
