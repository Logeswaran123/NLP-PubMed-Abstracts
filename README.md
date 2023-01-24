# NLP-PubMed-Abstracts ⚕️📋

## Paper
PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts | [Link](https://arxiv.org/abs/1710.06071)

## Dataset
PubMed 200k RCT dataset | [Link](https://github.com/Franck-Dernoncourt/pubmed-rct)

## How to run
```python
 python run.py --data <path to dataset>
```

Argumets: <br/>
*<path to dataset\>* - Path to Dataset directory with train, validation and test dataset.

## Experiments
Following are test results with model trained on 10% of dataset, <br/>
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

Following are test results with model trained on 100% of dataset, <br/>
```
Baseline model Results:
 {'accuracy': 74.97580533665146, 'precision': 0.7431966265737352, 'recall': 0.7497580533665146, 'f1': 0.7389570175489825}

1D Convolutional model (token embeddings) Results:
 {'accuracy': 84.06262961426793, 'precision': 0.8396560920826044, 'recall': 0.8406262961426794, 'f1': 0.8392693996811378}

Transfer learning model Results:
 {'accuracy': 79.89769113784045, 'precision': 0.7987512364491505, 'recall': 0.7989769113784045, 'f1': 0.7968573552912587}

1D Convolutional model (char embeddings) Results:
 {'accuracy': 78.16604451818057, 'precision': 0.7849655144017162, 'recall': 0.7816604451818057, 'f1': 0.7803912162453682}

Hybrid model (token and char embeddings) Results:
 {'accuracy': 84.48776441310659, 'precision': 0.8542530409852241, 'recall': 0.8448776441310659, 'f1': 0.8431915858550552}

Hybrid model (token, char and position embeddings) Results:
 {'accuracy': 87.46025162449882, 'precision': 0.8751028356756292, 'recall': 0.8746025162449882, 'f1': 0.8732769016124527}
```

## References:
* [SkimLit](https://colab.research.google.com/github/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb#scrollTo=dDWUcMGOauy8)
