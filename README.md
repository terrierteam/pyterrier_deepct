# pyterrier-deeptct

Advanced [PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [DeepCT](https://github.com/AdeDZY/DeepCT).


## Usage

Given an existing DeepCT checkpoint and original Google BERT files, an DeepCT transformer can be created as follows:

```python
from pyterrier_deepct import DeepCTTransformer
deepct = pyterrier_deepct.DeepCTTransformer("bert-base-uncased/bert_config.json", "marco/model.ckpt-65816")
indexer = deepct >> pt.IterDictIndexer("./deepct_index_path")
indexer.index(dataset.get_corpus_iter())
```

## Demos
 - vaswani.ipy - [[Github](blob/main/pyterrier_deepct_vaswani.ipynb
.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_deepct/blob/main/pyterrier_deepct_vaswani.ipynb)] - demonstrates end-to-end indexing and retrieval on the Vaswani corpus (~11k documents)

## References

 - [Dai19]: Zhuyun Dai, Jamie Callan. Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval. https://arxiv.org/abs/1910.10687
 - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation in Information Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271


## Credits
 - Craig Macdonald, University of Glasgow
