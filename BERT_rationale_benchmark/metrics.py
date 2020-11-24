import argparse
import json
import logging
import os
import pprint

from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import torch

from scipy.stats import entropy
from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, precision_recall_curve, roc_auc_score

from BERT_rationale_benchmark.utils import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    load_jsonl,
    load_documents,
    load_flattened_documents
 )

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

# start_token is inclusive, end_token is exclusive
@dataclass(eq=True, frozen=True)
class Rationale:
    ann_id: str
    docid: str
    start_token: int
    end_token: int

    def to_token_level(self) -> List['Rationale']:
        ret = []
        for t in range(self.start_token, self.end_token):
            ret.append(Rationale(self.ann_id, self.docid, t, t+1))
        return ret

    @classmethod
    def from_annotation(cls, ann: Annotation) -> List['Rationale']:
        ret = []
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.append(Rationale(ann.annotation_id, ev.docid, ev.start_token, ev.end_token))
        return ret

    @classmethod
    def from_instance(cls, inst: dict) -> List['Rationale']:
        ret = []
        for rat in inst['rationales']:
            for pred in rat.get('hard_rationale_predictions', []):
                ret.append(Rationale(inst['annotation_id'], rat['docid'], pred['start_token'], pred['end_token']))
        return ret

@dataclass(eq=True, frozen=True)
class PositionScoredDocument:
    ann_id: str
    docid: str
    scores: Tuple[float]
    truths: Tuple[bool]

    @classmethod
    def from_results(cls, instances: List[dict], annotations: List[Annotation], docs: Dict[str, List[Any]], use_tokens: bool=True) -> List['PositionScoredDocument']:
        """Creates a paired list of annotation ids/docids/predictions/truth values"""
        key_to_annotation = dict()
        for ann in annotations:
            for ev in chain.from_iterable(ann.evidences):
                key = (ann.annotation_id, ev.docid)
                if key not in key_to_annotation:
                    key_to_annotation[key] = [False for _ in docs[ev.docid]]
                if use_tokens:
                    start, end = ev.start_token, ev.end_token
                else:
                    start, end = ev.start_sentence, ev.end_sentence
                for t in range(start, end):
                    key_to_annotation[key][t] = True
        ret = []
        if use_tokens:
            field = 'soft_rationale_predictions'
        else:
            field = 'soft_sentence_predictions'
        for inst in instances:
            for rat in inst['rationales']:
                docid = rat['docid']
                scores = rat[field]
                key = (inst['annotation_id'], docid)
                assert len(scores) == len(docs[docid])
                if key in key_to_annotation :
                    assert len(scores) == len(key_to_annotation[key])
                else :
                    #In case model makes a prediction on docuemnt(s) for which ground truth evidence is not present
                    key_to_annotation[key] = [False for _ in docs[docid]]
                ret.append(PositionScoredDocument(inst['annotation_id'], docid, tuple(scores), tuple(key_to_annotation[key])))
        return ret

def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)

def _keyed_rationale_from_list(rats: List[Rationale]) -> Dict[Tuple[str, str], Rationale]:
    ret = defaultdict(set)
    for r in rats:
        ret[(r.ann_id, r.docid)].add(r)
    return ret

def partial_match_score(truth: List[Rationale], pred: List[Rationale], thresholds: List[float]) -> List[Dict[str, Any]]:
    """Computes a partial match F1

    Computes an instance-level (annotation) micro- and macro-averaged F1 score.
    True Positives are computed by using intersection-over-union and
    thresholding the resulting intersection-over-union fraction.

    Micro-average results are computed by ignoring instance level distinctions
    in the TP calculation (and recall, and precision, and finally the F1 of
    those numbers). Macro-average results are computed first by measuring
    instance (annotation + document) precisions and recalls, averaging those,
    and finally computing an F1 of the resulting average.
    """

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)

    num_classifications = {k:len(v) for k,v in pred_to_rat.items()}
    num_truth = {k:len(v) for k,v in ann_to_rat.items()}
    ious = defaultdict(dict)
    for k in set(ann_to_rat.keys()) | set(pred_to_rat.keys()):
        for p in pred_to_rat.get(k, []):
            best_iou = 0.0
            for t in ann_to_rat.get(k, []):
                num = len(set(range(p.start_token, p.end_token)) & set(range(t.start_token, t.end_token)))
                denom = len(set(range(p.start_token, p.end_token)) | set(range(t.start_token, t.end_token)))
                iou = 0 if denom == 0 else num / denom
                if iou > best_iou:
                    best_iou = iou
            ious[k][p] = best_iou
    scores = []
    for threshold in thresholds:
        threshold_tps = dict()
        for k, vs in ious.items():
            threshold_tps[k] = sum(int(x >= threshold) for x in vs.values())
        micro_r = sum(threshold_tps.values()) / sum(num_truth.values()) if sum(num_truth.values()) > 0 else 0
        micro_p = sum(threshold_tps.values()) / sum(num_classifications.values()) if sum(num_classifications.values()) > 0 else 0
        micro_f1 = _f1(micro_r, micro_p)
        macro_rs = list(threshold_tps.get(k, 0.0) / n if n > 0 else 0 for k, n in num_truth.items())
        macro_ps = list(threshold_tps.get(k, 0.0) / n if n > 0 else 0 for k, n in num_classifications.items())
        macro_r = sum(macro_rs) / len(macro_rs) if len(macro_rs) > 0 else 0
        macro_p = sum(macro_ps) / len(macro_ps) if len(macro_ps) > 0 else 0
        macro_f1 = _f1(macro_r, macro_p)
        scores.append({'threshold': threshold,
                       'micro': {
                            'p': micro_p,
                            'r': micro_r,
                            'f1': micro_f1
                       },
                       'macro': {
                            'p': macro_p,
                            'r': macro_r,
                            'f1': macro_f1
                       },
                       })
    return scores

def score_hard_rationale_predictions(truth: List[Rationale], pred: List[Rationale]) -> Dict[str, Dict[str, float]]:
    """Computes instance (annotation)-level micro/macro averaged F1s"""
    scores = dict()
    truth = set(truth)
    pred = set(pred)
    micro_prec = len(truth & pred) / len(pred)
    micro_rec = len(truth & pred) / len(truth)
    micro_f1 = _f1(micro_prec, micro_rec)
    scores['instance_micro'] = {
                                'p': micro_prec,
                                'r': micro_rec,
                                'f1': micro_f1,
                               }

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    instances_to_scores = dict()
    for k in set(ann_to_rat.keys()) | (pred_to_rat.keys()):
        if len(pred_to_rat.get(k, set())) > 0:
            instance_prec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(pred_to_rat[k])
        else:
            instance_prec = 0
        if len(ann_to_rat.get(k, set())) > 0:
            instance_rec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(ann_to_rat[k])
        else:
            instance_rec = 0
        instance_f1 = _f1(instance_prec, instance_rec)
        instances_to_scores[k] = {
                                    'p': instance_prec,
                                    'r': instance_rec,
                                    'f1': instance_f1,
                                 }
    # these are calculated as sklearn would
    macro_prec = sum(instance['p'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_rec = sum(instance['r'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_f1 = sum(instance['f1'] for instance in instances_to_scores.values()) / len(instances_to_scores)

    f1_scores = [instance['f1'] for instance in instances_to_scores.values()]
    print(macro_f1,  np.argsort(f1_scores)[::-1])



    scores['instance_macro'] = {
                                'p': macro_prec,
                                'r': macro_rec,
                                'f1': macro_f1,
                               }
    return scores

def _auprc(truth: Dict[Any, List[bool]], preds: Dict[Any, List[float]]) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    aucs = []
    for k, true in truth.items():
        pred = preds[k]
        true = [int(t) for t in true]
        precision, recall, _ = precision_recall_curve(true, pred)
        aucs.append(auc(recall, precision))
    return np.average(aucs)

def _score_aggregator(truth: Dict[Any, List[bool]], preds: Dict[Any, List[float]], score_function: Callable[[List[float], List[float]], float ], discard_single_class_answers: bool) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    scores = []
    for k, true in truth.items():
        pred = preds[k]
        if (all(true) or all(not x for x in true)) and discard_single_class_answers:
            continue
        true = [int(t) for t in true]
        scores.append(score_function(true, pred))
    return np.average(scores)

def score_soft_tokens(paired_scores: List[PositionScoredDocument]) -> Dict[str, float]:
    truth = {(ps.ann_id, ps.docid): ps.truths for ps in paired_scores}
    pred = {(ps.ann_id, ps.docid): ps.scores for ps in paired_scores}
    auprc_score = _auprc(truth, pred)
    ap = _score_aggregator(truth, pred, average_precision_score, True)
    roc_auc = _score_aggregator(truth, pred, roc_auc_score, True)

    return {
             'auprc': auprc_score,
             'average_precision': ap,
             'roc_auc_score': roc_auc,
           }

def _instances_aopc(instances: List[dict], thresholds: List[float], key: str) -> Tuple[float, List[float]]:
    dataset_scores = []
    for inst in instances:
        kls = inst['classification']
        beta_0 = inst['classification_scores'][kls]
        instance_scores = []
        for score in filter(lambda x : x['threshold'] in thresholds, sorted(inst['thresholded_scores'], key=lambda x: x['threshold'])):
            beta_k = score[key][kls]
            delta = beta_0 - beta_k
            instance_scores.append(delta)
        assert len(instance_scores) == len(thresholds)
        dataset_scores.append(instance_scores)
    dataset_scores = np.array(dataset_scores)
    # a careful reading of Samek, et al. "Evaluating the Visualization of What a Deep Neural Network Has Learned"
    # and some algebra will show the reader that we can average in any of several ways and get the same result:
    # over a flattened array, within an instance and then between instances, or over instances (by position) an
    # then across them.
    final_score = np.average(dataset_scores)
    position_scores = np.average(dataset_scores, axis=0).tolist()

    return final_score, position_scores

def compute_aopc_scores(instances: List[dict], aopc_thresholds: List[float]):
    if aopc_thresholds is None :
        aopc_thresholds = sorted(set(chain.from_iterable([x['threshold'] for x in y['thresholded_scores']] for y in instances)))
    aopc_comprehensiveness_score, aopc_comprehensiveness_points = _instances_aopc(instances, aopc_thresholds, 'comprehensiveness_classification_scores')
    aopc_sufficiency_score, aopc_sufficiency_points = _instances_aopc(instances, aopc_thresholds, 'sufficiency_classification_scores')
    return aopc_thresholds, aopc_comprehensiveness_score, aopc_comprehensiveness_points, aopc_sufficiency_score, aopc_sufficiency_points

def score_classifications(instances: List[dict], annotations: List[Annotation], docs: Dict[str, List[str]], aopc_thresholds: List[float]) -> Dict[str, float]:
    def compute_kl(cls_scores_, faith_scores_):
        keys = list(cls_scores_.keys())
        cls_scores_ = [cls_scores_[k] for k in keys]
        faith_scores_ = [faith_scores_[k] for k in keys]
        return entropy(faith_scores_, cls_scores_)
    labels = list(set(x.classification for x in annotations))
    label_to_int = {l:i for i,l in enumerate(labels)}
    key_to_instances = {inst['annotation_id']:inst for inst in instances}
    truth = []
    predicted = []
    for ann in annotations:
        truth.append(label_to_int[ann.classification])
        inst = key_to_instances[ann.annotation_id]
        predicted.append(label_to_int[inst['classification']])
    classification_scores = classification_report(truth, predicted, output_dict=True, target_names=labels, digits=3)
    accuracy = accuracy_score(truth, predicted)
    if 'comprehensiveness_classification_scores' in instances[0]:
        comprehensiveness_scores = [x['classification_scores'][x['classification']] - x['comprehensiveness_classification_scores'][x['classification']] for x in instances]
        comprehensiveness_score = np.average(comprehensiveness_scores)
    else :
        comprehensiveness_score = None
        comprehensiveness_scores = None

    if 'sufficiency_classification_scores' in instances[0]:
        sufficiency_scores = [x['classification_scores'][x['classification']] - x['sufficiency_classification_scores'][x['classification']] for x in instances]
        sufficiency_score = np.average(sufficiency_scores)
    else :
        sufficiency_score = None
        sufficiency_scores = None
    
    if 'comprehensiveness_classification_scores' in instances[0]:
        comprehensiveness_entropies = [entropy(list(x['classification_scores'].values())) - entropy(list(x['comprehensiveness_classification_scores'].values())) for x in instances]
        comprehensiveness_entropy = np.average(comprehensiveness_entropies)
        comprehensiveness_kl = np.average(list(compute_kl(x['classification_scores'], x['comprehensiveness_classification_scores']) for x in instances))
    else:
        comprehensiveness_entropies = None
        comprehensiveness_kl = None
        comprehensiveness_entropy = None

    if 'sufficiency_classification_scores' in instances[0]:
        sufficiency_entropies = [entropy(list(x['classification_scores'].values())) - entropy(list(x['sufficiency_classification_scores'].values())) for x in instances]
        sufficiency_entropy = np.average(sufficiency_entropies)
        sufficiency_kl = np.average(list(compute_kl(x['classification_scores'], x['sufficiency_classification_scores']) for x in instances))
    else:
        sufficiency_entropies = None
        sufficiency_kl = None
        sufficiency_entropy = None

    if 'thresholded_scores' in instances[0]:
        aopc_thresholds, aopc_comprehensiveness_score, aopc_comprehensiveness_points, aopc_sufficiency_score, aopc_sufficiency_points = compute_aopc_scores(instances, aopc_thresholds)
    else:
        aopc_thresholds, aopc_comprehensiveness_score, aopc_comprehensiveness_points, aopc_sufficiency_score, aopc_sufficiency_points = None, None, None, None, None
    if 'tokens_to_flip' in instances[0]:
        token_percentages = []
        for ann in annotations:
            # in practice, this is of size 1 for everything except e-snli
            docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
            inst = key_to_instances[ann.annotation_id]
            tokens = inst['tokens_to_flip']
            doc_lengths = sum(len(docs[d]) for d in docids)
            token_percentages.append(tokens / doc_lengths)
        token_percentages = np.average(token_percentages)
    else:
        token_percentages = None

    return {
              'accuracy': accuracy,
              'prf': classification_scores,
              'comprehensiveness': comprehensiveness_score,
              'sufficiency': sufficiency_score,
              'comprehensiveness_entropy': comprehensiveness_entropy,
              'comprehensiveness_kl': comprehensiveness_kl,
              'sufficiency_entropy': sufficiency_entropy,
              'sufficiency_kl': sufficiency_kl,
              'aopc_thresholds': aopc_thresholds,
              'comprehensiveness_aopc': aopc_comprehensiveness_score,
              'comprehensiveness_aopc_points': aopc_comprehensiveness_points,
              'sufficiency_aopc': aopc_sufficiency_score,
              'sufficiency_aopc_points': aopc_sufficiency_points,
           }

def verify_instance(instance: dict, docs: Dict[str, list], thresholds: Set[float]):
    error = False
    docids = []
    # verify the internal structure of these instances is correct:
    # * hard predictions are present
    # * start and end tokens are valid
    # * soft rationale predictions, if present, must have the same document length

    for rat in instance['rationales']:
        docid = rat['docid']
        if docid not in docid:
            error = True
            logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} could not be found as a preprocessed document! Gave up on additional processing.')
            continue
        doc_length = len(docs[docid])
        for h1 in rat.get('hard_rationale_predictions', []):
            # verify that each token is valid
            # verify that no annotations overlap
            for h2 in rat.get('hard_rationale_predictions', []):
                if h1 == h2:
                    continue
                if len(set(range(h1['start_token'], h1['end_token'])) & set(range(h2['start_token'], h2['end_token']))) > 0:
                    logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} {h1} and {h2} overlap!')
                    error = True
            if h1['start_token'] > doc_length:
                logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
            if h1['end_token'] > doc_length:
                logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
        # length check for soft rationale
        # note that either flattened_documents or sentence-broken documents must be passed in depending on result
        soft_rationale_predictions = rat.get('soft_rationale_predictions', [])
        if len(soft_rationale_predictions) > 0 and len(soft_rationale_predictions) != doc_length:
            logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} expected classifications for {doc_length} tokens but have them for {len(soft_rationale_predictions)} tokens instead!')
            error = True

    # count that one appears per-document
    docids = Counter(docids)
    for docid, count in docids.items():
        if count > 1:
            error = True
            logging.info('Error! For instance annotation={instance["annotation_id"]}, docid={docid} appear {count} times, may only appear once!')

    classification = instance.get('classification', '')
    if not isinstance(classification, str):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, classification field {classification} is not a string!')
        error = True
    classification_scores = instance.get('classification_scores', dict())
    if not isinstance(classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, classification_scores field {classification_scores} is not a dict!')
        error = True
    comprehensiveness_classification_scores = instance.get('comprehensiveness_classification_scores', dict())
    if not isinstance(comprehensiveness_classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, comprehensiveness_classification_scores field {comprehensiveness_classification_scores} is not a dict!')
        error = True
    sufficiency_classification_scores = instance.get('sufficiency_classification_scores', dict())
    if not isinstance(sufficiency_classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, sufficiency_classification_scores field {sufficiency_classification_scores} is not a dict!')
        error = True
    if ('classification' in instance) != ('classification_scores' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide classification scores!')
        error = True
    if ('comprehensiveness_classification_scores' in instance) and not ('classification' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide a comprehensiveness_classification_score')
        error = True
    if ('sufficiency_classification_scores' in instance) and not ('classification_scores' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a sufficiency_classification_score, you must also provide a classification score!')
        error = True
    if 'thresholded_scores' in instance:
        instance_thresholds = set(x['threshold'] for x in instance['thresholded_scores'])
        if instance_thresholds != thresholds:
            error = True
            logging.info('Error: {instance["thresholded_scores"]} has thresholds that differ from previous thresholds: {thresholds}')
        if 'comprehensiveness_classification_scores' not in instance\
            or 'sufficiency_classification_scores' not in instance\
            or 'classification' not in instance\
            or 'classification_scores' not in instance:
            error = True
            logging.info('Error: {instance} must have comprehensiveness_classification_scores, sufficiency_classification_scores, classification, and classification_scores defined when including thresholded scores')
        if not all('sufficiency_classification_scores' in x for x in instance['thresholded_scores']):
            error = True
            logging.info('Error: {instance} must have sufficiency_classification_scores for every threshold')
        if not all('comprehensiveness_classification_scores' in x for x in instance['thresholded_scores']):
            error = True
            logging.info('Error: {instance} must have comprehensiveness_classification_scores for every threshold')
    return error

def verify_instances(instances: List[dict], docs: Dict[str, list]):
    annotation_ids = list(x['annotation_id'] for x in instances)
    key_counter = Counter(annotation_ids)
    multi_occurrence_annotation_ids = list(filter(lambda kv: kv[1] > 1, key_counter.items()))
    error = False
    if len(multi_occurrence_annotation_ids) > 0:
        error = True
        logging.info(f'Error in instances: {len(multi_occurrence_annotation_ids)} appear multiple times in the annotations file: {multi_occurrence_annotation_ids}')
    failed_validation = set()
    instances_with_classification = list()
    instances_with_soft_rationale_predictions = list()
    instances_with_soft_sentence_predictions = list()
    instances_with_comprehensiveness_classifications = list()
    instances_with_sufficiency_classifications = list()
    instances_with_thresholded_scores = list()
    if 'thresholded_scores' in instances[0]:
        thresholds = set(x['threshold'] for x in instances[0]['thresholded_scores'])
    else:
        thresholds = None
    for instance in instances:
        instance_error = verify_instance(instance, docs, thresholds)
        if instance_error:
            error = True
            failed_validation.add(instance['annotation_id'])
        if instance.get('classification', None) != None:
            instances_with_classification.append(instance)
        if instance.get('comprehensiveness_classification_scores', None) != None:
            instances_with_comprehensiveness_classifications.append(instance)
        if instance.get('sufficiency_classification_scores', None) != None:
            instances_with_sufficiency_classifications.append(instance)
        has_soft_rationales = []
        has_soft_sentences = []
        for rat in instance['rationales']:
            if rat.get('soft_rationale_predictions', None) != None:
                has_soft_rationales.append(rat)
            if rat.get('soft_sentence_predictions', None) != None:
                has_soft_sentences.append(rat)
        if len(has_soft_rationales) > 0:
            instances_with_soft_rationale_predictions.append(instance)
            if len(has_soft_rationales) != len(instance['rationales']):
                error = True
                logging.info(f'Error: instance {instance["annotation"]} has soft rationales for some but not all reported documents!')
        if len(has_soft_sentences) > 0:
            instances_with_soft_sentence_predictions.append(instance)
            if len(has_soft_sentences) != len(instance['rationales']):
                error = True
                logging.info(f'Error: instance {instance["annotation"]} has soft sentences for some but not all reported documents!')
        if 'thresholded_scores' in instance:
            instances_with_thresholded_scores.append(instance)
    logging.info(f'Error in instances: {len(failed_validation)} instances fail validation: {failed_validation}')
    if len(instances_with_classification) != 0 and len(instances_with_classification) != len(instances):
        logging.info(f'Either all {len(instances)} must have a classification or none may, instead {len(instances_with_classification)} do!')
        error = True
    if len(instances_with_soft_sentence_predictions) != 0 and len(instances_with_soft_sentence_predictions) != len(instances):
        logging.info(f'Either all {len(instances)} must have a sentence prediction or none may, instead {len(instances_with_soft_sentence_predictions)} do!')
        error = True
    if len(instances_with_soft_rationale_predictions) != 0 and len(instances_with_soft_rationale_predictions) != len(instances):
        logging.info(f'Either all {len(instances)} must have a soft rationale prediction or none may, instead {len(instances_with_soft_rationale_predictions)} do!')
        error = True
    if len(instances_with_comprehensiveness_classifications) != 0 and len(instances_with_comprehensiveness_classifications) != len(instances):
        error = True
        logging.info(f'Either all {len(instances)} must have a comprehensiveness classification or none may, instead {len(instances_with_comprehensiveness_classifications)} do!')
    if len(instances_with_sufficiency_classifications) != 0 and len(instances_with_sufficiency_classifications) != len(instances):
        error = True
        logging.info(f'Either all {len(instances)} must have a sufficiency classification or none may, instead {len(instances_with_sufficiency_classifications)} do!')
    if len(instances_with_thresholded_scores) != 0 and len(instances_with_thresholded_scores) != len(instances):
        error = True
        logging.info(f'Either all {len(instances)} must have thresholded scores or none may, instead {len(instances_with_thresholded_scores)} do!')
    if error:
        raise ValueError('Some instances are invalid, please fix your formatting and try again')

def _has_hard_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0]\
        and len(results[0]['rationales']) > 0\
        and 'hard_rationale_predictions' in results[0]['rationales'][0]\
        and results[0]['rationales'][0]['hard_rationale_predictions'] is not None\
        and len(results[0]['rationales'][0]['hard_rationale_predictions']) > 0

def _has_soft_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_rationale_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_rationale_predictions'] is not None

def _has_soft_sentence_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_sentence_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_sentence_predictions'] is not None

def _has_classifications(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'classification' in results[0] and results[0]['classification'] is not None

def main():
    parser = argparse.ArgumentParser(description="""Computes rationale and final class classification scores""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True, help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--split', dest='split', required=True, help='Which of {train,val,test} are we scoring on?')
    parser.add_argument('--strict', dest='strict', required=False, action='store_true', default=False, help='Do we perform strict scoring?')
    parser.add_argument('--results', dest='results', required=True, help="""Results File
    Contents are expected to be jsonl of:
    {
        "annotation_id": str, required
        # these classifications *must not* overlap
        "rationales": List[
            {
                "docid": str, required
                "hard_rationale_predictions": List[{
                    "start_token": int, inclusive, required
                    "end_token": int, exclusive, required
                }], optional,
                # token level classifications, a value must be provided per-token
                # in an ideal world, these correspond to the hard-decoding above.
                "soft_rationale_predictions": List[float], optional.
                # sentence level classifications, a value must be provided for every
                # sentence in each document, or not at all
                "soft_sentence_predictions": List[float], optional.
            }
        ],
        # the classification the model made for the overall classification task
        "classification": str, optional
        # A probability distribution output by the model. We require this to be normalized.
        "classification_scores": Dict[str, float], optional
        # The next two fields are measures for how faithful your model is (the
        # rationales it predicts are in some sense causal of the prediction), and
        # how sufficient they are. We approximate a measure for comprehensiveness by
        # asking that you remove the top k%% of tokens from your documents,
        # running your models again, and reporting the score distribution in the
        # "comprehensiveness_classification_scores" field.
        # We approximate a measure of sufficiency by asking exactly the converse
        # - that you provide model distributions on the removed k%% tokens.
        # 'k' is determined by human rationales, and is documented in our paper.
        # You should determine which of these tokens to remove based on some kind
        # of information about your model: gradient based, attention based, other
        # interpretability measures, etc.
        # scores per class having removed k%% of the data, where k is determined by human comprehensive rationales
        "comprehensiveness_classification_scores": Dict[str, float], optional
        # scores per class having access to only k%% of the data, where k is determined by human comprehensive rationales
        "sufficiency_classification_scores": Dict[str, float], optional
        # the number of tokens required to flip the prediction - see "Is Attention Interpretable" by Serrano and Smith.
        "tokens_to_flip": int, optional
        "thresholded_scores": List[{
            "threshold": float, required,
            "comprehensiveness_classification_scores": like "classification_scores"
            "sufficiency_classification_scores": like "classification_scores"
        }], optional. if present, then "classification" and "classification_scores" must be present
    }
    When providing one of the optional fields, it must be provided for *every* instance.
    The classification, classification_score, and comprehensiveness_classification_scores
    must together be present for every instance or absent for every instance.
    """)
    parser.add_argument('--iou_thresholds', dest='iou_thresholds', required=False, nargs='+', type=float, default=[0.5], help='''Thresholds for IOU scoring.

    These are used for "soft" or partial match scoring of rationale spans.
    A span is considered a match if the size of the intersection of the prediction
    and the annotation, divided by the union of the two spans, is larger than
    the IOU threshold. This score can be computed for arbitrary thresholds.
    ''')
    parser.add_argument('--score_file', dest='score_file', required=False, default=None, help='Where to write results?')
    parser.add_argument('--aopc_thresholds', nargs='+', required=False, type=float, default=[0.01, 0.05, 0.1, 0.2, 0.5], help='Thresholds for AOPC Thresholds')
    args = parser.parse_args()
    results = load_jsonl(args.results)
    docids = set(chain.from_iterable([rat['docid'] for rat in res['rationales']] for res in results))
    docs = load_flattened_documents(args.data_dir, docids)
    verify_instances(results, docs)
    # load truth
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, args.split + '.jsonl'))
    docids |= set(chain.from_iterable((ev.docid for ev in chain.from_iterable(ann.evidences)) for ann in annotations))

    has_final_predictions = _has_classifications(results)
    scores = dict()
    if args.strict:
        if not args.iou_thresholds:
            raise ValueError("--iou_thresholds must be provided when running strict scoring")
        if not has_final_predictions:
            raise ValueError("We must have a 'classification', 'classification_score', and 'comprehensiveness_classification_score' field in order to perform scoring!")
    # TODO think about offering a sentence level version of these scores.
    if _has_hard_predictions(results):
        truth = list(chain.from_iterable(Rationale.from_annotation(ann) for ann in annotations))
        pred = list(chain.from_iterable(Rationale.from_instance(inst) for inst in results))
        if args.iou_thresholds is not None:
            iou_scores = partial_match_score(truth, pred, args.iou_thresholds)
            scores['iou_scores'] = iou_scores
        # NER style scoring
        rationale_level_prf = score_hard_rationale_predictions(truth, pred)
        scores['rationale_prf'] = rationale_level_prf
        token_level_truth = list(chain.from_iterable(rat.to_token_level() for rat in truth))
        token_level_pred = list(chain.from_iterable(rat.to_token_level() for rat in pred))
        token_level_prf = score_hard_rationale_predictions(token_level_truth, token_level_pred)
        scores['token_prf'] = token_level_prf
    else:
        logging.info("No hard predictions detected, skipping rationale scoring")

    if _has_soft_predictions(results):
        flattened_documents = load_flattened_documents(args.data_dir, docids)
        paired_scoring = PositionScoredDocument.from_results(results, annotations, flattened_documents, use_tokens=True)
        token_scores = score_soft_tokens(paired_scoring)
        scores['token_soft_metrics'] = token_scores
    else:
        logging.info("No soft predictions detected, skipping rationale scoring")

    if _has_soft_sentence_predictions(results):
        documents = load_documents(args.data_dir, docids)
        paired_scoring = PositionScoredDocument.from_results(results, annotations, documents, use_tokens=False)
        sentence_scores = score_soft_tokens(paired_scoring)
        scores['sentence_soft_metrics'] = sentence_scores
    else:
        logging.info("No sentence level predictions detected, skipping sentence-level diagnostic")

    if has_final_predictions:
        flattened_documents = load_flattened_documents(args.data_dir, docids)
        class_results = score_classifications(results, annotations, flattened_documents, args.aopc_thresholds)
        scores['classification_scores'] = class_results
    else:
        logging.info("No classification scores detected, skipping classification")

    pprint.pprint(scores)

    if args.score_file:
        with open(args.score_file, 'w') as of:
            json.dump(scores, of, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()
