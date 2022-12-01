import shutil
from pyterrier.measures import *
import tempfile
import unittest
import pyterrier as pt
import pyterrier_deepct

class DeepCTRegressionTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    def test_deepct_vaswani_tok(self):
        dataset = pt.get_dataset('irds:vaswani')
        deepct = pyterrier_deepct.DeepCT()
        indexer = deepct >> pt.IterDictIndexer(self.test_dir, pretokenised=True)
        indexer.index(dataset.get_corpus_iter())
        bm25 = pt.TerrierRetrieve(self.test_dir, wmodel='BM25')
        res = pt.Experiment([bm25], dataset.get_topics(), dataset.get_qrels(), [MAP, P@10])
        self.assertAlmostEqual(res['AP'].iloc[0], 0.176284, places=5)
        self.assertAlmostEqual(res['P@10'].iloc[0], 0.265591, places=5)

    def test_deepct_vaswani_txt(self):
        dataset = pt.get_dataset('irds:vaswani')
        deepct = pyterrier_deepct.DeepCT()
        indexer = deepct >> pyterrier_deepct.Toks2Text() >> pt.IterDictIndexer(self.test_dir, fields=['text'])
        indexer.index(dataset.get_corpus_iter())
        bm25 = pt.TerrierRetrieve(self.test_dir, wmodel='BM25')
        res = pt.Experiment([bm25], dataset.get_topics(), dataset.get_qrels(), [MAP, P@10])
        self.assertAlmostEqual(res['AP'].iloc[0], 0.297491, places=5)
        self.assertAlmostEqual(res['P@10'].iloc[0], 0.356989, places=5)


if __name__ == '__main__':
    unittest.main()
