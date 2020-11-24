# TODO consider if this can be collapsed back down into the pipeline_train.py
import argparse
import json
import logging
import random
import os

from sklearn.metrics import accuracy_score

from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

from BERT_rationale_benchmark.utils import (
    Annotation,
    Evidence,
    write_jsonl,
    load_datasets,
    load_documents,
)
from BERT_explainability.modules.BERT.BertForSequenceClassification import \
    BertForSequenceClassification as BertForSequenceClassificationTest
from BERT_explainability.modules.BERT.BERT_cls_lrp import \
    BertForSequenceClassification as BertForClsOrigLrp

from transformers import BertForSequenceClassification

from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    attention_list = attention_list[:len(text_list)]
    if attention_list.max() == attention_list.min():
        attention_list = torch.zeros_like(attention_list)
    else:
        attention_list = 100 * (attention_list - attention_list.min()) / (attention_list.max() - attention_list.min())
    attention_list[attention_list < 1] = 0
    attention_list = attention_list.tolist()
    text_list = [text_list[i].replace('$', '') for i in range(len(text_list))]
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth=150mm]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            # string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
            # print(text_list[idx])
            if '\#\#' in text_list[idx]:
                token = text_list[idx].replace('\#\#', '')
                string += "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + token + "}"
            else:
                string += " " + "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + text_list[idx] + "}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list


def scores_per_word_from_scores_per_token(input, tokenizer, input_ids, scores_per_id):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]
    score_per_char = []

    # TODO: DELETE
    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)
    # TODO: DELETE

    for i in range(len(scores_per_id)):
        if words[i] in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        score_per_char += [scores_per_id[i]] * len(words[i])


    score_per_word = []
    start_idx = 0
    end_idx = 0
    # TODO: DELETE
    words_from_chars = []
    for inp in input:
        if start_idx >= len(score_per_char):
            break
        end_idx = end_idx + len(inp)
        score_per_word.append(np.max(score_per_char[start_idx:end_idx]))

        # TODO: DELETE
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))

        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False

    return torch.tensor(score_per_word)

def get_input_words(input, tokenizer, input_ids):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]

    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)

    start_idx = 0
    end_idx = 0
    words_from_chars = []
    for inp in input:
        if start_idx >= len(input_ids_chars):
            break
        end_idx = end_idx + len(inp)
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))
        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False
    return words_from_chars

def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
    sents = []
    sent_token_spans = []
    for sent in doc:
        tokens = []
        spans = []
        start = 0
        for w in sent:
            if w in special_token_map:
                tokens.append(w)
            else:
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans

def initialize_models(params: dict, batch_first: bool, use_half_precision=False):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bert_dir = params['bert_dir']
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=len(evidence_classes))
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer


BATCH_FIRST = True


def extract_docid_from_dataset_element(element):
    return next(iter(element.evidences))[0].docid

def extract_evidence_from_dataset_element(element):
    return next(iter(element.evidences))


def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task
     (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data (load, intern documents, load json)
    * convert data for evidence identification - in the case of training data we take all the positives and sample some
      negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a
          broader sampling of negative values.
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
    assert BATCH_FIRST
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
        logger.info(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    train, val, test = load_datasets(args.data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer = \
        initialize_models(model_params, batch_first=BATCH_FIRST)
    logger.info(f'We have {len(word_interner)} wordpieces')
    cache = os.path.join(args.output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents) = torch.load(cache)
    else:
        logger.info(f'Interning documents')
        interned_documents = {}
        for d, doc in documents.items():
            encoding = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                max_length=model_params['max_length'],
                return_token_type_ids=False,
                pad_to_max_length=False,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            interned_documents[d] = encoding
        torch.save((interned_documents), cache)

    evidence_classifier = evidence_classifier.cuda()
    optimizer = None
    scheduler = None

    save_dir = args.output_dir

    logging.info(f'Beginning training classifier')
    evidence_classifier_output_dir = os.path.join(save_dir, 'classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'classifier_epoch_data.pt')

    device = next(evidence_classifier.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_classifier.parameters(), lr=model_params['evidence_classifier']['lr'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    batch_size = model_params['evidence_classifier']['batch_size']
    epochs = model_params['evidence_classifier']['epochs']
    patience = model_params['evidence_classifier']['patience']
    max_grad_norm = model_params['evidence_classifier'].get('max_grad_norm', None)

    class_labels = [k for k, v in sorted(evidence_classes.items())]

    results = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_epoch = -1
    best_val_acc = 0
    best_val_loss = float('inf')
    best_model_state_dict = None
    start_epoch = 0
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        evidence_classifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
        logging.info(f'Restoring training from epoch {start_epoch}')
    logging.info(f'Training evidence classifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = random.sample(train, k=len(train))
        epoch_train_loss = 0
        epoch_training_acc = 0
        evidence_classifier.train()
        logging.info(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            targets = [evidence_classes[s.classification] for s in batch_elements]
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
            input_ids = torch.stack([samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(
                1).to(device)
            attention_masks = torch.stack(
                [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
            preds = evidence_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]
            epoch_training_acc += accuracy_score(preds.argmax(dim=1).cpu(), targets.cpu(), normalize=False)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            epoch_train_loss += loss.item()
            loss.backward()
            assert loss == loss  # for nans
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(evidence_classifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        epoch_train_loss /= len(epoch_train_data)
        epoch_training_acc /= len(epoch_train_data)
        assert epoch_train_loss == epoch_train_loss  # for nans
        results['train_loss'].append(epoch_train_loss)
        logging.info(f'Epoch {epoch} training loss {epoch_train_loss}')
        logging.info(f'Epoch {epoch} training accuracy {epoch_training_acc}')

        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_data = random.sample(val, k=len(val))
            evidence_classifier.eval()
            val_batch_size = 32
            logging.info(
                f'Validating with {len(epoch_val_data) // val_batch_size} batches with {len(epoch_val_data)} examples')
            for batch_start in range(0, len(epoch_val_data), val_batch_size):
                batch_elements = epoch_val_data[batch_start:min(batch_start + val_batch_size, len(epoch_val_data))]
                targets = [evidence_classes[s.classification] for s in batch_elements]
                targets = torch.tensor(targets, dtype=torch.long, device=device)
                samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
                input_ids = torch.stack(
                    [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
                attention_masks = torch.stack(
                    [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
                    device)
                preds = evidence_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]
                epoch_val_acc += accuracy_score(preds.argmax(dim=1).cpu(), targets.cpu(), normalize=False)
                loss = criterion(preds, targets.to(device=preds.device)).sum()
                epoch_val_loss += loss.item()

            epoch_val_loss /= len(val)
            epoch_val_acc /= len(val)
            results["val_acc"].append(epoch_val_acc)
            results["val_loss"] = epoch_val_loss

            logging.info(f'Epoch {epoch} val loss {epoch_val_loss}')
            logging.info(f'Epoch {epoch} val acc {epoch_val_acc}')

            if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
                best_epoch = epoch
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_acc': best_val_acc,
                    'done': 0,
                }
                torch.save(evidence_classifier.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.debug(f'Epoch {epoch} new best model with val accuracy {epoch_val_acc}')
        if epoch - best_epoch > patience:
            logging.info(f'Exiting after epoch {epoch} due to no improvement')
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    evidence_classifier.load_state_dict(best_model_state_dict)
    evidence_classifier = evidence_classifier.to(device=device)
    evidence_classifier.eval()

    # test

    test_classifier = BertForSequenceClassificationTest.from_pretrained(model_params['bert_dir'],
                                                                        num_labels=len(evidence_classes)).to(device)
    orig_lrp_classifier = BertForClsOrigLrp.from_pretrained(model_params['bert_dir'],
                                                            num_labels=len(evidence_classes)).to(device)
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        test_classifier.load_state_dict(torch.load(model_save_file))
        orig_lrp_classifier.load_state_dict(torch.load(model_save_file))
        test_classifier.eval()
        orig_lrp_classifier.eval()
        test_batch_size = 1
        logging.info(
            f'Testing with {len(test) // test_batch_size} batches with {len(test)} examples')

        # explainability
        explanations = Generator(test_classifier)
        explanations_orig_lrp = Generator(orig_lrp_classifier)
        method = "transformer_attribution"
        method_folder = {"transformer_attribution": "ours", "partial_lrp": "partial_lrp", "last_attn": "last_attn",
                         "attn_gradcam": "attn_gradcam", "lrp": "lrp", "rollout": "rollout",
                         "ground_truth": "ground_truth", "generate_all": "generate_all"}
        method_expl = {"transformer_attribution": explanations.generate_LRP,
                       "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                       "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                       "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                       "lrp": explanations_orig_lrp.generate_full_lrp,
                       "rollout": explanations_orig_lrp.generate_rollout}

        os.makedirs(os.path.join(args.output_dir, method_folder[method]), exist_ok=True)

        result_files = []
        for i in range(5,85,5):
            result_files.append(open(os.path.join(args.output_dir, '{0}/identifier_results_{1}.json').format(method_folder[method], i), 'w'))

        j = 0
        for batch_start in range(0, len(test), test_batch_size):
            batch_elements = test[batch_start:min(batch_start + test_batch_size, len(test))]
            targets = [evidence_classes[s.classification] for s in batch_elements]
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            samples_encoding = [interned_documents[extract_docid_from_dataset_element(s)] for s in batch_elements]
            input_ids = torch.stack(
                [samples_encoding[i]['input_ids'] for i in range(len(samples_encoding))]).squeeze(1).to(device)
            attention_masks = torch.stack(
                [samples_encoding[i]['attention_mask'] for i in range(len(samples_encoding))]).squeeze(1).to(
                device)
            preds = test_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]

            for s in batch_elements:
                doc_name = extract_docid_from_dataset_element(s)
                inp = documents[doc_name].split()
                classification = "neg" if targets.item() == 0 else "pos"
                is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
                if method == "generate_all":
                    file_name ="{0}_{1}_{2}.tex".format(j, classification, is_classification_correct)
                    GT_global = os.path.join(args.output_dir, '{0}/visual_results_{1}.pdf').format(
                             method_folder["ground_truth"], j)
                    GT_ours = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                             method_folder["transformer_attribution"], j, classification, is_classification_correct)
                    CF_ours = os.path.join(args.output_dir, '{0}/{1}_CF.pdf').format(
                             method_folder["transformer_attribution"], j)
                    GT_partial = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                        method_folder["partial_lrp"], j, classification, is_classification_correct)
                    CF_partial = os.path.join(args.output_dir, '{0}/{1}_CF.pdf').format(
                        method_folder["partial_lrp"], j)
                    GT_gradcam = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                        method_folder["attn_gradcam"], j, classification, is_classification_correct)
                    CF_gradcam = os.path.join(args.output_dir, '{0}/{1}_CF.pdf').format(
                        method_folder["attn_gradcam"], j)
                    GT_lrp = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                        method_folder["lrp"], j, classification, is_classification_correct)
                    CF_lrp = os.path.join(args.output_dir, '{0}/{1}_CF.pdf').format(
                        method_folder["lrp"], j)
                    GT_lastattn = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                        method_folder["last_attn"], j, classification, is_classification_correct)
                    GT_rollout = os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.pdf').format(
                        method_folder["rollout"], j, classification, is_classification_correct)
                    with open(file_name, 'w') as f:
                        f.write(r'''\documentclass[varwidth]{standalone}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}
{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{
    \setlength{\tabcolsep}{2pt} % Default value: 6pt
    \begin{tabular}{ccc}
        \includegraphics[width=0.32\linewidth]{''' + GT_global + '''}&
        \includegraphics[width=0.32\linewidth]{''' + GT_ours + '''}&
        \includegraphics[width=0.32\linewidth]{''' + CF_ours + '''}\\\\
        (a) & (b) & (c)\\\\
        \includegraphics[width=0.32\linewidth]{''' + GT_partial + '''}&
        \includegraphics[width=0.32\linewidth]{''' + CF_partial + '''}&
        \includegraphics[width=0.32\linewidth]{''' + GT_gradcam + '''}\\\\
        (d) & (e) & (f)\\\\
        \includegraphics[width=0.32\linewidth]{''' + CF_gradcam + '''}&
        \includegraphics[width=0.32\linewidth]{''' + GT_lrp + '''}&
        \includegraphics[width=0.32\linewidth]{''' + CF_lrp + '''}\\\\
        (g) & (h) & (i)\\\\
        \includegraphics[width=0.32\linewidth]{''' + GT_lastattn + '''}&
        \includegraphics[width=0.32\linewidth]{''' + GT_rollout + '''}&\\\\
        (j) & (k)&\\\\
    \end{tabular}
}}}
\end{CJK*}
\end{document}
)''')
                    j += 1
                    break


                if method == "ground_truth":
                    inp_cropped = get_input_words(inp, tokenizer, input_ids[0])
                    cam = torch.zeros(len(inp_cropped))
                    for evidence in extract_evidence_from_dataset_element(s):
                        start_idx = evidence.start_token
                        if start_idx >= len(cam):
                            break
                        end_idx = evidence.end_token
                        cam[start_idx:end_idx] = 1
                    generate(inp_cropped, cam,
                             (os.path.join(args.output_dir, '{0}/visual_results_{1}.tex').format(method_folder[method],
                                                                                                 j)), color="green")
                    j = j + 1
                    break
                text = tokenizer.convert_ids_to_tokens(input_ids[0])
                classification = "neg" if targets.item() == 0 else "pos"
                is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
                target_idx = targets.item()
                cam_target = method_expl[method](input_ids=input_ids, attention_mask=attention_masks, index=target_idx)[0]
                cam_target = cam_target.clamp(min=0)
                generate(text, cam_target,
                         (os.path.join(args.output_dir, '{0}/{1}_GT_{2}_{3}.tex').format(
                             method_folder[method], j, classification, is_classification_correct)))
                if method in ["transformer_attribution", "partial_lrp", "attn_gradcam", "lrp"]:
                    cam_false_class = method_expl[method](input_ids=input_ids, attention_mask=attention_masks, index=1-target_idx)[0]
                    cam_false_class = cam_false_class.clamp(min=0)
                    generate(text, cam_false_class,
                         (os.path.join(args.output_dir, '{0}/{1}_CF.tex').format(
                             method_folder[method], j)))
                cam = cam_target
                cam = scores_per_word_from_scores_per_token(inp, tokenizer,input_ids[0], cam)
                j = j + 1
                doc_name = extract_docid_from_dataset_element(s)
                hard_rationales = []
                for res, i in enumerate(range(5, 85, 5)):
                    print("calculating top ", i)
                    _, indices = cam.topk(k=i)
                    for index in indices.tolist():
                        hard_rationales.append({
                            "start_token": index,
                            "end_token": index+1
                        })
                    result_dict = {
                        "annotation_id": doc_name,
                        "rationales": [{
                            "docid": doc_name,
                            "hard_rationale_predictions": hard_rationales
                        }],
                    }
                    result_files[res].write(json.dumps(result_dict) + "\n")

        for i in range(len(result_files)):
            result_files[i].close()


if __name__ == '__main__':
    main()
