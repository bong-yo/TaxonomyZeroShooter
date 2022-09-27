from typing import Union, Dict, List
from copy import deepcopy
from src.scoring_functions import PosteriorScoresPropagation
from src.dataset import BaseData
from src.encoders import ZeroShooterZSTC


class ZeroShootTaxonomyMapper:
    @staticmethod
    def run(taxonomy: Union[Dict, str], documents: Union[str, List[str]]):
        """
        Apply Zero-Shoot Taxonomy Mapper on a list of texts given a custom taxonomy.

        Parameters
        ----------
        taxonomy: Union[Dict, str] - either 1) the Taxonomy tree itself, in the form of dict of dicts or 2) path to the json where the taxonomy tree is stored
        documents: Union[str, List[str]] - text or list of texts to be classified

        Returns
        -------
        top_labels: List[List[str]] - List of top label for each level.
        posterior_scores_trees:   List[Dict] - Taxonomy Tree with posterior score for each label.
        """
        data = BaseData(taxonomy, documents)
        encoder = ZeroShooterZSTC('all-mpnet-base-v2')
        scorer = PosteriorScoresPropagation(data, encoder)
        prior_scores_trees = scorer.compute_prior_trees()
        label2alpha = scorer.compute_labels_alpha()
        posterior_scores_trees = \
            [tree for tree in scorer.apply_USP(prior_scores_trees, label2alpha)]
        top_labels = [
            scorer.get_levels_top_label(tree)
            for tree in deepcopy(posterior_scores_trees)
            ]
        return top_labels, posterior_scores_trees


if __name__ == "__main__":

    tax_tree = {
        'Computer Science': {'Machine Learning': {}, 'Quantum Computing': {}},
        'Art': {'Renaissance': {}, 'Cubism': {}, 'Impressionism': {}},
        'Sport': {'Athletics': {}, 'Football': {}, 'Tennis': {}}
    }

    docs = [
        'OpenAI released DALL-E: an amazing neural network that leverages Transformers \
            architecture and Diffusion model training to generate images starting from text',
        'Usain Bolt was arguably the fastest sprinter that has ever run, and it currently\
            holds the world record for both 100 meters and 200 meters'
    ]

    top_labels, usp_scores = ZeroShootTaxonomyMapper.run(tax_tree, docs)
    results = list(zip(docs, usp_scores, top_labels))