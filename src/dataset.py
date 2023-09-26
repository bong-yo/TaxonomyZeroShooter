import logging
import re
from typing import Dict, List, Tuple, Union
from copy import copy
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, deque
from src.utils import FileIO
from globals import Paths

logger = logging.getLogger('zeroshot-logger')


class TaxonomyBase:
    """Base class to handle a Taxonomy tree in the form of nested Dictionaries.
    """
    def __init__(self, label2rename: Dict[str, str] = {}) -> None:
        # Dictionary for mapping labela name, into new names that better
        # represent its semantic.
        self.label2rename = label2rename

    def load_from_file(self, filename: str) -> List[str]:
        return FileIO.read_json(filename)

    def get_taxonomy_levels(self, tree: Dict) -> Tuple[List[List[str]], List[str]]:
        '''
        Return list of unique labels for each level, given the taxonomy tree.

        Parameters
        ----------
        tree: Dict[Dict[...]] - Taxonomy tree in the form of nested Dictionaries

        Returns
        -------
        labels_levels: List[List[str]] - List of lists of labels at each level.
        labels_flat:   List[str] - Flat list of all labels (ignoring tree structure).
        '''
        labels_levels = [list(set(level)) for level in self.traverse_bfs(tree)]
        labels_flat = [label for lev in labels_levels for label in lev]
        return labels_levels, labels_flat

    def traverse_bfs(self, tree: Dict) -> List[str]:
        """
        BFS traverse of the Taxonomy tree.
        - Perform relabelling during traverse if a non-null relabel map
            'self.label2rename' is provided.
        - Save labels in a separate list for each level.
        """
        labels_levels = []
        queue = deque([(tree, 0)])
        # BFS.
        while queue:
            root, level = queue.popleft()
            # Update label levels.
            if len(labels_levels) == level:
                labels_levels.append([])
            children = copy(list(root.keys()))
            for child in children:
                # Rename label if appropriate.
                if child.strip() in self.label2rename:
                    new_child_name = self.label2rename[child.strip()]
                    root[new_child_name] = root.pop(child)
                    child = new_child_name
                # Save label in its own level.
                labels_levels[-1].append(child)
                # Append subtree to BFS.
                if root[child]:
                    queue.append((root[child], level + 1))
        return labels_levels


class BaseData(TaxonomyBase):
    """
    Base class to handle the common format the data should have to be fed to the
    ML algos

    Parameters
    ----------
    taxonomy: Union[Dict, str] - can be two things: 1) the Taxonomy tree itself, in the form of dict of dicts
                                2) path to the json where the taxonomy tree is stored
    documents: Union[str, List[str]] - text or list of texts to be classified
    labels_remapping: Dict[str, str] - a dictionary in case one wants to change name
                                       to the taxonomy labels.

    Returns
    -------
    labels_levels: List[List[str]] - List of lists of labels at each level.
    labels_flat:   List[str] - Flat list of all labels (ignoring tree structure).
    """
    def __init__(self,
                 taxonomy: Union[Dict, str],
                 documents: Union[str, List[str]] = [],
                 labels_remapping: Dict[str, str] = {}) -> None:
        super(BaseData, self).__init__(labels_remapping)
        if isinstance(documents, str):
            documents = [documents]
        self.abstracts: List[str] = documents
        self.tax_tree: Dict = self.load_taxonomy_tree(taxonomy)
        self.labels_levels, self.labels_flat = self.get_taxonomy_levels(self.tax_tree)

    def load_taxonomy_tree(self, tree: Union[Dict, str]):
        if isinstance(tree, str):
            return self.load_from_file(tree)
        elif isinstance(tree, dict):
            return tree


class WebOfScience(BaseData):
    def __init__(self, datasplit: str, topn: int = None) -> None:
        remap_level1 = {
            'CS': 'Computer Science',
            'Civil': 'Civil Engineering',
            'ECE': 'Electrical Engineering',
            'Psychology': 'Psychology',
            'MAE': 'Mechanical Engineering',
            'Medical': 'Medical Science',
            'biochemistry': 'Biochemistry'
        }
        super(WebOfScience, self).__init__({}, [], remap_level1)
        logger.debug('Loading WebOfScience data')

        # Get taxonomy.
        self.tax_tree = self.load_taxonomy_tree(f'{Paths.WOS_DIR}/tax_tree.json')
        self.labels_levels, self.labels_flat = self.get_taxonomy_levels(self.tax_tree)

        # Load documents.
        assert datasplit in {'train', 'valid', 'test', 'all'}, 'Wrong datasplit name.'
        filename = f'{Paths.WOS_DIR}/train.xlsx' if datasplit == 'train' \
            else f'{Paths.WOS_DIR}/valid.xlsx' if datasplit == 'valid' \
            else f'{Paths.WOS_DIR}/test.xlsx' if datasplit == 'test' \
            else f'{Paths.WOS_DIR}/Data.xlsx'
        data = pd.read_excel(filename).head(topn)
        logger.debug(f'n docs: {len(data)}')

        self.abstracts = [x.strip() for x in data['Abstract'].values]
        self.Y = [
            [remap_level1[x.strip()] for x in data['Domain'].values],  # Labels level 1.
            [x.strip() for x in data['area'].values]  # Labels level 2.
        ]
        self.tax_depth = len(self.Y)
        self.n_data = len(self.abstracts)


class DBPedia(BaseData):
    def __init__(self, datasplit: str, topn: int = None, build_tree: bool = False) -> None:
        logger.debug('Loading DBPedia data')
        super(DBPedia, self).__init__({}, [])
        self.train_file = f'{Paths.DBP_DIR}/DBPEDIA_train.csv'
        self.valid_file = f'{Paths.DBP_DIR}/DBPEDIA_val.csv'
        self.test_file = f'{Paths.DBP_DIR}/DBPEDIA_test.csv'
        self.tax_tree_file = f'{Paths.DBP_DIR}/tax_tree.json'
        if build_tree:
            self.build_tax_tree()

        # Get taxonomy.
        self.tax_tree = self.load_taxonomy_tree(f'{Paths.DBP_DIR}/tax_tree.json')
        self.labels_levels, self.labels_flat = self.get_taxonomy_levels(self.tax_tree)

        assert datasplit in {'train', 'valid', 'test'}, 'Wrong datasplit name.'
        filename = self.train_file if datasplit == 'train' \
            else self.valid_file if datasplit == 'valid' \
            else self.test_file
        data = pd.read_csv(filename).head(topn).fillna('')

        self.abstracts = [x.strip() for x in data['text'].values]
        self.Y = [
            [self.norm_label(x.strip()) for x in data['l1'].values],
            [self.norm_label(x.strip()) for x in data['l2'].values],
            [self.norm_label(x.strip()) for x in data['l3'].values]
        ]
        self.tax_depth = len(self.Y)
        self.n_data = len(self.abstracts)
        logger.debug('Done opening file')

    def build_tax_tree(self) -> None:
        '''Build Taxonomy tree document by document.'''
        data = pd.read_csv(self.train_file).fillna('')
        logger.debug('n docs: ', len(data))
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for idx, row in tqdm(data.iterrows(), desc="building taxonomy tree"):
            # Separate Keywords as they are written as 'ThisIsAKeyword' -> 'This Is A Keyword'
            l1 = self.norm_label(row['l1'])
            l2 = self.norm_label(row['l2'])
            l3 = self.norm_label(row['l3'])
            tree[l1][l2][l3] = {}
        FileIO.write_json(tree, self.tax_tree_file)

    def norm_label(self, x: str) -> str:
        return ' '.join(re.findall('[A-Z][^A-Z]*', x))


class AmazonHTC(BaseData):
    def __init__(self, datasplit: str, topn: int = None, build_tree: bool = False) -> None:
        logger.debug('Loading AmazonHTC data')
        super(AmazonHTC, self).__init__({}, [])
        self.train_file = f'{Paths.AHTC_DIR}/train_40k.csv'
        self.test_file = f'{Paths.AHTC_DIR}/train_40k.csv'
        self.valid_file = f'{Paths.AHTC_DIR}/val_10k.csv'
        self.tax_tree_file = f'{Paths.AHTC_DIR}/tax_tree.json'
        if build_tree:
            self.build_tax_tree()

        # Get taxonomy.
        self.tax_tree = self.load_taxonomy_tree(f'{Paths.AHTC_DIR}/tax_tree.json')
        self.labels_levels, self.labels_flat = self.get_taxonomy_levels(self.tax_tree)

        assert datasplit in {'valid', 'test'}, 'Wrong datasplit name.'
        filename = self.test_file if datasplit == 'test' else self.valid_file
        data = pd.read_csv(filename).head(topn).fillna('')

        self.abstracts, Y1, Y2, Y3 = [], [], [], []
        for idx, row in data.iterrows():
            self.abstracts.append(row['Title'].strip() + ' . ' + row['Text'].strip())
            Y1.append(row['Cat1'].strip())
            Y2.append(row['Cat2'].strip())
            Y3.append(row['Cat3'].strip())
        self.Y = [Y1, Y2, Y3]
        self.tax_depth = len(self.Y)
        self.n_data = len(self.abstracts)
        logger.debug('Done opening file')

    def build_tax_tree(self) -> None:
        '''Build Taxonomy tree document by document.'''
        data = pd.read_csv(self.train_file).fillna('')
        logger.debug('n docs: ', len(data))
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for idx, row in tqdm(data.iterrows(), desc="building taxonomy tree"):
            l1 = row['Cat1'].strip()
            l2 = row['Cat2'].strip()
            l3 = row['Cat3'].strip()
            tree[l1][l2][l3] = {}
        FileIO.write_json(tree, self.tax_tree_file)
