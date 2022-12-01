from collections import Counter
import transformers
import os
import torch
import numpy as np
import pyterrier as pt
if not pt.started():
    pt.init()
import pandas as pd
from more_itertools import chunked
import deepct

def _subword_weight_to_word_weight(tokens, logits, smoothing="none", m=100, keep_all_terms=False):
    import numpy as np
    fulltokens = []
    weights = []
    for token, weight in zip(tokens, logits):
        if token.startswith('##'):
            fulltokens[-1] += token[2:]
        else:
            fulltokens.append(token)
            weights.append(weight)
    fulltokens_filtered, weights_filtered = [], []
    selected_tokens = {}
    for token, w in zip(fulltokens, weights):
        if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
            continue

        if w < 0: w = 0
        if smoothing == "sqrt":
            tf = int(np.round(m * np.sqrt(w)))
        else:
            tf = int(np.round(m * w))

        if tf < 1: 
            if not keep_all_terms: continue
            else: tf = 1

        selected_tokens[token] = max(tf, selected_tokens.get(token, 0))
    return selected_tokens

def dict_tf2text(tfdict):
    rtr = ""
    for t in tfdict:
        for i in range(tfdict[t]):
            rtr += t + " " 
    return rtr

class DeepCTTransformer(pt.Transformer):
    
    #bert_config="/users/tr.craigm/projects/pyterrier/DeepCT/bert-base-uncased/bert_config.json"
    #checkpoint="/users/tr.craigm/projects/pyterrier/DeepCT/outputs/marco/model.ckpt-65816"
    def __init__(self, bert_config, checkpoint, vocab_file="bert-base-uncased/vocab.txt", max_seq_length=128):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        try:
            import deepct
        except ImportError:
            raise ImportError('deepct package not found\n - pip install git+https://github.com/cmacdonald/DeepCT.git@tf1#egg=DeepCT')
        from deepct import modeling
        from deepct import run_deepct
        model_fn = run_deepct.model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(bert_config),
            init_checkpoint=checkpoint,
            learning_rate=5e5,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=False,
            use_one_hot_embeddings=False, 
            use_all_layers=False,
        )
        from deepct import tokenization
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        
        import tensorflow as tf
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=None,
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,
                per_host_input_for_training=is_per_host))

        self.estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=False,
                model_fn=model_fn,
                config=run_config,
                predict_batch_size=16)

        self.max_seq_length = max_seq_length

    def transform(self, docs):
        def gen():
            from deepct.run_deepct import InputExample
            for row in docs.itertuples():
                yield InputExample(row.docno, row.text, {})
        from deepct import run_deepct
        features = run_deepct.convert_examples_to_features(gen(), None, self.max_seq_length, self.tokenizer)
        input_fn = run_deepct.input_fn_builder(features, self.max_seq_length, False, False)
        result = self.estimator.predict(input_fn=input_fn)
        newdocs = []
        for (i, prediction) in enumerate(result):
            targets = prediction["target_weights"]
            logits = prediction["logits"]
            tokens = self.tokenizer.convert_ids_to_tokens(prediction["token_ids"])
            term2tf = _subword_weight_to_word_weight(tokens, logits)
            newdocs.append(dict_tf2text(term2tf))
            if i >= len(docs):
                break
                
        rtr = pd.DataFrame()
        rtr["docno"] = docs["docno"]
        rtr["text"] = newdocs
        return rtr

def _subword_weight_to_dict(tokens, logits):
    fulltokens = []
    weights = []
    for token, weight in zip(tokens, logits.tolist()):
        if token.startswith('##'):
            fulltokens[-1] += token[2:]
        else:
            fulltokens.append(token)
            weights.append(weight)

    selected_tokens = {}
    for token, tf in zip(fulltokens, weights):
        if token in ('[CLS]', '[SEP]', '[PAD]') or tf <= 0:
            continue
        selected_tokens[token] = max(tf, selected_tokens.get(token, 0))
    return selected_tokens


class DeepCT(pt.Transformer):
    def __init__(self, model='macavaney/deepct', batch_size=64, scale=100, device=None, round=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(model).eval().to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.batch_size = batch_size
        self.scale = scale
        self.round = round

    def transform(self, inp):
        res = []
        with torch.no_grad():
            for texts in chunked(inp['text'], self.batch_size):
                texts = list(texts)
                toks = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                batch_tok_scores = self.model(**{k: v.to(self.device) for k, v in toks.items()})['logits']
                batch_tok_scores = batch_tok_scores.squeeze(2).cpu().numpy()
                batch_tok_scores = self.scale * batch_tok_scores
                if self.round:
                    batch_tok_scores = np.round(batch_tok_scores).astype(np.int32)
                for i in range(batch_tok_scores.shape[0]):
                    toks_txt = self.tokenizer.convert_ids_to_tokens(toks['input_ids'][i])
                    toks_dict = _subword_weight_to_dict(toks_txt, batch_tok_scores[i])
                    res.append(toks_dict)
        return inp.assign(toks=res)


class Toks2Text(pt.Transformer):
    def transform(self, inp):
        return inp.assign(text=inp['toks'].apply(self.toks2text))
    def toks2text(self, toks):
        return ' '.join(Counter(toks).elements())
