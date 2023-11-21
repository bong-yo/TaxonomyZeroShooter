from src.zero_shooter import TaxZeroShot
from src.few_shot.modeling import FewShotTrainer
from src.few_shot.select_FS_examples import FewShotData, ExampleFewShot
from src.dataset import WebOfScience
from globals import Paths


if __name__ == "__main__":

    # tax_tree = {
    #     'Computer Science': {'Machine Learning': {}, 'Quantum Computing': {}},
    #     'Art': {'Renaissance': {}, 'Cubism': {}, 'Impressionism': {}},
    #     'Sport': {'Athletics': {}, 'Football': {}, 'Tennis': {}}
    # }

    # docs = [
    #     'OpenAI released DALL-E: an amazing neural network that leverages Transformers \
    #         architecture and Diffusion model training to generate images starting from text',
    #     'Usain Bolt was arguably the fastest sprinter that has ever run, and it currently\
    #         holds the world record for both 100 meters and 200 meters'
    # ]

    # labels_supervised = [
    #     ['Computer Science', 'Art', 'Machine Learning'],
    #     ['Sport', 'Athletics']
    # ]

    train_data = WebOfScience('train', topn=100)
    valid_data = WebOfScience('valid', topn=30)
    # test_data = WebOfScience('test', topn=10)

    tax_zero_shooter = TaxZeroShot(
        train_data.tax_tree,
        f'{Paths.SAVE_DIR}/label_alphas_WebOfScience.json',
        no_grad_zstc=False,
        no_grad_usp=True
    )

    # Get examples for few-shot training.
    few_shot_data = FewShotData(tax_zero_shooter)
    examples_train = few_shot_data.select_examples(
        train_data, min_entropy=0.7, max_entropy=0.95, n_shots=10
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
    fs_trainer.evaluate(tax_zero_shooter, examples_valid)
    fs_trainer.train(tax_zero_shooter, examples_train, examples_valid, lr=1e-2,
                     n_epochs=4)
