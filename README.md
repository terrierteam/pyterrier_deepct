# pyterrier-deepct

Advanced [PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [DeepCT](https://github.com/AdeDZY/DeepCT).

## Installation 
```
pip install --upgrade git+https://github.com/terrierteam/pyterrier_deepct.git
```

## Usage

```python
from pyterrier_deepct import DeepCT, Toks2Text
deepct = DeepCT() # loads macavaney/deepct, a version of the model weights converted to huggingface format by default
indexer = deepct >> Toks2Text() >> pt.IterDictIndexer("./deepct_index_path")
indexer.index(dataset.get_corpus_iter())
```

Options:
 - `device`: device to run the model on, defualt cuda if available (or cpu if not)
 - `batch_size`: batch size when encoding documents, defualt 64
 - `scale`: score multiplier that moves the model outputs to a reasonable integer range, default 100
 - `round`: round the scores to the nearest integer, default True

## Usage (legacy API)

The old API uses the `deepct` repository, which requires version 1 of tensorflow (not available everywhere, e.g., Colab).

Given an existing DeepCT checkpoint and original Google BERT files, an DeepCT transformer can be created as follows:

```python
from pyterrier_deepct import DeepCTTransformer
deepct = pyterrier_deepct.DeepCTTransformer("bert-base-uncased/bert_config.json", "marco/model.ckpt-65816")
indexer = deepct >> pt.IterDictIndexer("./deepct_index_path")
indexer.index(dataset.get_corpus_iter())
```

## Demos
 - vaswani.ipy - [[Github](blob/main/pyterrier_deepct_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_deepct/blob/main/pyterrier_deepct_vaswani.ipynb)] - demonstrates end-to-end indexing and retrieval on the Vaswani corpus (~11k documents)

## References

 - [Dai19]: Zhuyun Dai, Jamie Callan. Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval. https://arxiv.org/abs/1910.10687
 - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation in Information Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271


## Credits
 - Craig Macdonald, University of Glasgow
 - Sean MacAvaney, University of Glasgow
