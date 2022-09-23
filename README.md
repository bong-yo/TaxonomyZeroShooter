# Zero-Shot Taxonomy Mapping for Document Classification
Classification of text according to a custom (hierarchical) taxonomy of categories. Does not need any labelled data or any fine-tuning of the model.


## Installation
* Python version 3.8.8
* App version: 1.3.0-beta
* Install modules: ```pip install -r requirements.txt```



## Reproduce Paper Results
1. Download benchmarck Datasets:
    * [Web Of Science](https://data.mendeley.com/datasets/9rw3vkcfy4/6) and save it as ```'/datasets/WebOfScience'```
    * [DBPedia Extract](https://www.kaggle.com/datasets/danofer/dbpedia-classes) and save it as ```'/datasets/DBPedia_Extract'```
    * [Amazon Reviews](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification) and save it as ```'/datasets/Amazon_Reviews'```

2. Run benchmarck scripts:
    * Zeros-Shot Text Classification (Z-STC):\
        classification of text according to all the categories, disregarding the structure of the Taxonomy and, therefore, regarding each category as independent from the others.
        ```bash
        python ZSTC_benchmarking.py --gpu_no --topn --dataset --savefile
        ```
    * Upwards Score Propagation mechanism (USP):\
        compute Posterior Relevance scores for each category in the Taxonomy by leveraging: prior Z-STC scores, Relevance Thresholds and the structure of the Taxonomy.
        ```bash
        python USP_benchmarking.py --gpu_no --topn --dataset --savefile
        ```
    * Entropy based Sentence Selection (ESS): 
        mechanism to focus the attention of the encoding model on the important sentences of the document. Importance here is represented by the value of the entropy of the sentence with respect to the categories of the Taxonomy.
        ```bash
        python ESS_benchmarking.py --gpu_no --level --topn --dataset 
        ```


### Author

* Repo owner or admin
* Other community or team contact