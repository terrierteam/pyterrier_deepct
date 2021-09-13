
import pandas as pd
from pyterrier.transformer import TransformerBase
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

class DeepCTTransformer(TransformerBase):
    
    #bert_config="/users/tr.craigm/projects/pyterrier/DeepCT/bert-base-uncased/bert_config.json"
    #checkpoint="/users/tr.craigm/projects/pyterrier/DeepCT/outputs/marco/model.ckpt-65816"
    def __init__(self, bert_config, checkpoint, vocab_file="bert-base-uncased/vocab.txt", max_seq_length=128):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        features = deepct.run_deepct.convert_examples_to_features(gen(), None, self.max_seq_length, self.tokenizer)
        input_fn = deepct.run_deepct.input_fn_builder(features, self.max_seq_length, False, False)
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
