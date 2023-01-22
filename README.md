# NLP-PubMed-Abstracts ⚕️📋

## Paper
PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts | [Link](https://arxiv.org/abs/1710.06071)

## Dataset
PubMed 200k RCT dataset | [Link](https://github.com/Franck-Dernoncourt/pubmed-rct)

## How to run
```python
 python run.py -d <path to dataset>
```

Argumets: <br/>
*<path to dataset\>* - Path to Dataset directory with train, validation and test dataset.

## Experiments
Following are results with model trained on 10% of dataset, <br/>
```
Baseline model Results:
 {'accuracy': 72.1832384482987, 'precision': 0.7186466952323352, 'recall': 0.7218323844829869, 'f1': 0.6989250353450294}

1D Convolutional model (token embeddings) Results:
 {'accuracy': 80.05759300940024, 'precision': 0.8010363531029364, 'recall': 0.8005759300940024, 'f1': 0.7973543255468089}

Transfer learning model Results:
 {'accuracy': 73.37812789620018, 'precision': 0.7302941015295202, 'recall': 0.7337812789620018, 'f1': 0.7280725596495059}

1D Convolutional model (char embeddings) Results:
 {'accuracy': 67.91672183238448, 'precision': 0.6816523475097991, 'recall': 0.6791672183238449, 'f1': 0.6758935728995986}

Hybrid model (token and char embeddings) Results:
 {'accuracy': 79.8391367668476, 'precision': 0.803343397964016, 'recall': 0.7983913676684761, 'f1': 0.7969816102186146}

Hybrid model (token, char and position embeddings) Results:
 {'accuracy': 85.09201641731762, 'precision': 0.8530446920261443, 'recall': 0.8509201641731762, 'f1': 0.847385312945692}
```
<br/>

Following are results with model trained on 100% of dataset, <br/>
<TODO>

## References:
* [SkimLit](https://colab.research.google.com/github/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb#scrollTo=dDWUcMGOauy8)
