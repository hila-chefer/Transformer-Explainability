import argparse
import json
import logging
import random
import os

from itertools import chain
from typing import Set

import numpy as np
import torch

from rationale_benchmark.utils import (
    write_jsonl,
    load_datasets,
    load_documents,
    intern_documents,
    intern_annotations
)
from rationale_benchmark.models.mlp import (
    AttentiveClassifier,
    BahadanauAttention,
    RNNEncoder,
    WordEmbedder
)
from rationale_benchmark.models.model_utils import extract_embeddings
from rationale_benchmark.models.pipeline.evidence_identifier import train_evidence_identifier
from rationale_benchmark.models.pipeline.evidence_classifier import train_evidence_classifier
from rationale_benchmark.models.pipeline.pipeline_utils import decode

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistant to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def initialize_models(params: dict, vocab: Set[str], batch_first: bool, unk_token='UNK'):
    # TODO this is obviously asking for some sort of dependency injection. implement if it saves me time.
    if 'embedding_file' in params['embeddings']:
        embeddings, word_interner, de_interner = extract_embeddings(vocab, params['embeddings']['embedding_file'], unk_token=unk_token)
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
    else:
        raise ValueError("No 'embedding_file' found in params!")
    word_embedder = WordEmbedder(embeddings, params['embeddings']['dropout'])
    query_encoder = RNNEncoder(word_embedder,
                               batch_first=batch_first,
                               condition=False,
                               attention_mechanism=BahadanauAttention(word_embedder.output_dimension))
    document_encoder = RNNEncoder(word_embedder,
                                  batch_first=batch_first,
                                  condition=True,
                                  attention_mechanism=BahadanauAttention(word_embedder.output_dimension,
                                                                         query_size=query_encoder.output_dimension))
    evidence_identifier = AttentiveClassifier(document_encoder,
                                              query_encoder,
                                              2,
                                              params['evidence_identifier']['mlp_size'],
                                              params['evidence_identifier']['dropout'])
    query_encoder = RNNEncoder(word_embedder,
                               batch_first=batch_first,
                               condition=False,
                               attention_mechanism=BahadanauAttention(word_embedder.output_dimension))
    document_encoder = RNNEncoder(word_embedder,
                                  batch_first=batch_first,
                                  condition=True,
                                  attention_mechanism=BahadanauAttention(word_embedder.output_dimension,
                                                                         query_size=query_encoder.output_dimension))
    evidence_classes = dict((y,x) for (x,y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = AttentiveClassifier(document_encoder,
                                              query_encoder,
                                              len(evidence_classes),
                                              params['evidence_classifier']['mlp_size'],
                                              params['evidence_classifier']['dropout'])
    return evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes


def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data (load, intern documents, load json)
    * convert data for evidence identification - in the case of training data we take all the positives and sample some negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a broader sampling of negative values.
    * train evidence identification
    * convert data for evidence classification - take all rationales + decisions and use this as input
    * train evidence classification
    * decode first the evidence, then run classification for each split
    
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    BATCH_FIRST = True

    with open(args.model_params, 'r') as fp:
        logging.debug(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
    train, val, test = load_datasets(args.data_dir)
    docids = set(e.docid for e in chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(args.data_dir, docids)
    document_vocab = set(chain.from_iterable(chain.from_iterable(documents.values())))
    annotation_vocab = set(chain.from_iterable(e.query.split() for e in chain(train, val, test)))
    logging.debug(f'Loaded {len(documents)} documents with {len(document_vocab)} unique words')
    # this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    vocab = document_vocab | annotation_vocab
    unk_token = 'UNK'
    evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes = \
        initialize_models(model_params, vocab, batch_first=BATCH_FIRST, unk_token=unk_token)
    logging.debug(f'Including annotations, we have {len(vocab)} total words in the data, with embeddings for {len(word_interner)}')
    interned_documents = intern_documents(documents, word_interner, unk_token)
    interned_train = intern_annotations(train, word_interner, unk_token)
    interned_val = intern_annotations(val, word_interner, unk_token)
    interned_test = intern_annotations(test, word_interner, unk_token)
    assert BATCH_FIRST # for correctness of the split  dimension for DataParallel
    evidence_identifier, evidence_ident_results = train_evidence_identifier(evidence_identifier.cuda(),
                                                                            args.output_dir, interned_train,
                                                                            interned_val,
                                                                            interned_documents,
                                                                            model_params,
                                                                            tensorize_model_inputs=True)
    evidence_classifier, evidence_class_results = train_evidence_classifier(evidence_classifier.cuda(),
                                                                            args.output_dir,
                                                                            interned_train,
                                                                            interned_val,
                                                                            interned_documents,
                                                                            model_params,
                                                                            class_interner=evidence_classes,
                                                                            tensorize_model_inputs=True)
    pipeline_batch_size = min([model_params['evidence_classifier']['batch_size'],
                               model_params['evidence_identifier']['batch_size']])
    pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier,
                                                                        evidence_classifier,
                                                                        interned_train,
                                                                        interned_val,
                                                                        interned_test,
                                                                        interned_documents,
                                                                        evidence_classes,
                                                                        pipeline_batch_size,
                                                                        tensorize_model_inputs=True)
    write_jsonl(train_decoded, os.path.join(args.output_dir, 'train_decoded.jsonl'))
    write_jsonl(val_decoded, os.path.join(args.output_dir, 'val_decoded.jsonl'))
    write_jsonl(test_decoded, os.path.join(args.output_dir, 'test_decoded.jsonl'))
    with open(os.path.join(args.output_dir, 'identifier_results.json'), 'w') as ident_output, \
            open(os.path.join(args.output_dir, 'classifier_results.json'), 'w') as class_output:
        ident_output.write(json.dumps(evidence_ident_results))
        class_output.write(json.dumps(evidence_class_results))
    for k, v in pipeline_results.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                logging.info(f'Pipeline results for {k}, {k1}={v1}')
        else:
            logging.info(f'Pipeline results {k}\t={v}')


if __name__ == '__main__':
    main()
