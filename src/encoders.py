import logging
from typing import List, Union
from tqdm import tqdm
from nltk import sent_tokenize
import numpy as np
import torch
from transformers import pipeline
import flair
from flair.models import TARSClassifier
from flair.data import Sentence
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from src.utils import entropy
from globals import Globals

logger = logging.getLogger('zeroshot-logger')



class Batcher:
    @classmethod
    def create_batches(cls, data: List[str], size: int) -> List[List[str]]:
        n_batches = len(data) // size + int(len(data) % size != 0)
        return [data[i * size: (i + 1) * size] for i in range(n_batches)]


class ZeroShooter:
    def __init__(self) -> None:
        pass


class ZeroShooterZSTC():
    '''Class of text encoders that encode according to Z-STE
    (i.e. documents and labels separately)'''
    def __init__(self, model_name: str='all-mpnet-base-v2') -> None:
        super(ZeroShooterZSTC, self).__init__()
        self.encoder = SentenceTransformer(model_name).to(Globals.DEVICE)

    def encode_labels(self, labels: List[str]) -> np.array:
        logger.debug("Encoding labels")
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        with torch.no_grad():
            self.labels_embs = self.encoder.encode(labels, show_progress_bar=False)
        return self.labels_embs

    def compute_labels_scores(self,
                           texts: Union[str, List[str]],
                           labels: Union[str, List[str]]=None,
                           encoding_method: str="base") -> np.array:
        texts, labels = self.check_inputs(texts, labels)
        # Compute sts similarity by batch.
        if encoding_method == 'base':
            docs_embs = self.encode_base(texts)
        elif encoding_method == 'ess':
            docs_embs = self.encode_ess(texts)
        else:
            logger.error(f"Unsupported encoding method: {encoding_method}")
        return cos_sim(docs_embs, self.labels_embs).numpy() # Matrix N x M, N docs and M labels.

    def encode_base(self, docs: List[str], show_progress_bar: bool=False):
        """Naively encode the whole document."""
        with torch.no_grad():
            return self.encoder.encode(docs, show_progress_bar=show_progress_bar)
    
    def encode_ess(self, 
                   docs: List[str],
                   min_len: int = 20,
                   topn: int = 10) -> np.array:
        """Encode each document with Entropy-based Sentence Selection (ESS).
        The encoding of the original document is recover as the average of the 
        embeddings of all of its sentences, each one weighted with its own entropy.

        Parameters
        ----------
        docs: List[str] - Text of documents to encode.
        min_len: int - Min len of a sentence for it to be considered, otherwise it will be ignored.
        topn: int - Only Top N sentences, according to entropy, will be conisdered for the final embedding.

        Return
        ------
        2D np.array - Documents embedding, computed as wighted sum of their sentences.
        """
        texts_embeddings = []
        for doc in docs:
            sents = [s for s in sent_tokenize(doc) if len(s) >= min_len]
            # Embed each sentence separately.
            with torch.no_grad():
                sents_emb = self.encoder.encode(sents, show_progress_bar=False)
            # For each sentence, compute the probability of belonging to each label, as 
            # the absolute value of the cosine similarity between sent and labels.
            sents_labels_probs = abs(cos_sim(sents_emb, self.labels_embs).numpy())
            s = entropy(sents_labels_probs)
            # Select only topn sents with lower entropy.
            if topn is not None:
                topn_ids = np.argsort(s)
                sents_emb = sents_emb[topn_ids]
                s = s[topn_ids]
            s = s.reshape(-1, 1)
            # Recover doc embedding as the average of the sents weighted with their entropy.
            doc_emb = np.sum((1 - s) * sents_emb, axis=0) / np.sum((1 - s))
            texts_embeddings.append(doc_emb)
        return np.vstack(texts_embeddings)

    def check_inputs(self, texts: Union[str, List[str]], labels: Union[str, List[str]]):
        # Handle single label case.
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            self.encode_labels(labels)
        # Handle single text case.
        if isinstance(texts, str):
            texts = [texts]
        return texts, labels


    def classify(self,
                 texts: List[str],
                 labels: List[str] = None,
                 topn: int = None) -> List[str]:
        similarity = self.compute_similarity(texts, labels)
        topn_sims = np.argsort(similarity, axis=-1)[:, ::-1][:,: topn]
        return [
            [
                (self.id2label[label_id], similarity[row, label_id])
                for label_id in row_topn.tolist()
            ]
            for row, row_topn in enumerate(topn_sims)
        ]


class ZeroshooterTARS(Batcher):
    def __init__(self, batch_size) -> None:
        super(ZeroshooterTARS, self).__init__()
        flair.device = torch.device(f'cuda:4')
        self.batch_size = batch_size
        self.model = TARSClassifier.load('tars-base')
        self.model.tars_model.multi_label = True
        self.model.tars_model.multi_label_threshold = 0.0

    def classify(self,
                 texts: List[str],
                 labels: List[str] = None,
                 topn: int = None) -> List[str]:
        # self.model.add_and_switch_to_new_task('zero-shot', labels, '-'.join(labels))
        docs = [Sentence(x) for x in texts]
        self.model.predict_zero_shot(docs, labels)
        res = []
        for d in docs:
            if not d.labels:
                res.append([('none', 0.999)])
            else:
                res.append(sorted([(x.value, x.score) for x in d.labels], key=lambda x: x[1], reverse=True))
        return res


class ZeroshooterBART:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.model = pipeline(
            'zero-shot-classification',
            model='facebook/bart-large-mnli',
            batch_size=batch_size,
            device=Globals.DEVICE
        )

    def classify(self,
                 texts: List[str],
                 labels: List[str] = None,
                 topn: int = None) -> List[str]:
        hypothesis_template = 'This text is about {}.' # the template used in this demo
        results =  self.model(texts, labels, hypothesis_template=hypothesis_template,
                              multi_label=True)
        return [
            [(label, score) for label, score in zip(res['labels'], res['scores'])]
            for res in results
        ]


class ZeroshooterBART_2(Batcher):
    def __init__(self, batch_size) -> None:
        super(ZeroshooterBART_2, self).__init__()
        self.batch_size = batch_size
        self.model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(Globals.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    def classify(self,
                 texts: List[str],
                 labels: List[str] = None,
                 topn: int = None) -> List[str]:
        res = [{k: 0 for k in labels} for _ in texts]
        for label in tqdm(labels, desc='Encoding texts'):
            hypothesis = f'This example is {label}.'
            probs_label = []
            for batch in self.create_batches(texts, self.batch_size):
                with torch.no_grad():
                    x = torch.vstack([
                        self.tokenizer.encode(
                            premise, hypothesis, max_length=200, truncation=True,
                            padding='max_length', truncation_strategy='only_first',
                            return_tensors='pt'
                        )
                        for premise in batch
                    ])
                    logits = self.model(x.to(Globals.DEVICE))[0]
                    entail_contradiction_logits = logits[:,[0,2]]
                    probs = entail_contradiction_logits.softmax(dim=-1)
                    probs_label.extend(probs[:,1].tolist())
            for x, p in zip(res, probs_label):
                x[label] = p
        res = [sorted(doc.items(), key=lambda x: x[1], reverse=True) for doc in res]
        return res
