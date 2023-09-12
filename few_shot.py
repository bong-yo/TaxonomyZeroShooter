from src.zero_shooter import TaxZeroShot
from src.modeling import Trainer
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

    train = WebOfScience('train', topn=None)
    valid = WebOfScience('valid', topn=None)
    test= WebOfScience('test', topn=None)

    tax_zero_shooter = TaxZeroShot(tax_tree, f'{Paths.SAVE_DIR}/label_alphas_prova.json')
    trainer = Trainer()

    lab2id = tax_zero_shooter.label2id
    id2lab = {i: lab for lab, i in lab2id.items()}
    lab2alpha = tax_zero_shooter.label2alpha
    trainer.train(tax_zero_shooter, docs, labels_supervised, lr=0.1, n_epochs=4)
    print()
    for i in range(len(lab2id)):
        print(id2lab[i])
        print('\t', lab2alpha[id2lab[i]], tax_zero_shooter.UPS.sigmoid_gate_model.a[i].item())
