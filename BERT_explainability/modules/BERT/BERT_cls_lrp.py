from transformers import BertPreTrainedModel
from transformers.utils import logging
from BERT_explainability.modules.layers_lrp import *
from BERT_explainability.modules.BERT.BERT_orig_lrp import BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from typing import List, Any
import torch
from BERT_rationale_benchmark.models.model_utils import PaddedSequence


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        return cam


# this is the actual classifier we will be using
class BertClassifier(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""

    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 max_length: int = 512,
                 use_half_precision=True):
        super(BertClassifier, self).__init__()
        bert = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels)
        if use_half_precision:
            import apex
            bert = bert.half()
        self.bert = bert
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        print(query)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id,
                                            device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        (classes,) = self.bert(bert_input.data,
                               attention_mask=bert_input.mask(on=0.0, off=float('-inf'), device=target_device),
                               position_ids=positions.data)
        assert torch.all(classes == classes)  # for nans

        print(input_tensors[0])
        print(self.relprop()[0])

        return classes

    def relprop(self, cam=None, **kwargs):
        return self.bert.relprop(cam, **kwargs)


if __name__ == '__main__':
    from transformers import BertTokenizer
    import os

    class Config:
        def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, num_labels,
                     hidden_dropout_prob):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.num_labels = num_labels
            self.hidden_dropout_prob = hidden_dropout_prob


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    x = tokenizer.encode_plus("In this movie the acting is great. The movie is perfect! [sep]",
                         add_special_tokens=True,
                         max_length=512,
                         return_token_type_ids=False,
                         return_attention_mask=True,
                         pad_to_max_length=True,
                         return_tensors='pt',
                         truncation=True)

    print(x['input_ids'])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model_save_file = os.path.join('./BERT_explainability/output_bert/movies/classifier/', 'classifier.pt')
    model.load_state_dict(torch.load(model_save_file))

    # x = torch.randint(100, (2, 20))
    # x = torch.tensor([[101, 2054, 2003, 1996, 15792, 1997, 2023, 3319, 1029, 102,
    #                    101, 4079, 102, 101, 6732, 102, 101, 2643, 102, 101,
    #                    2038, 102, 101, 1037, 102, 101, 2933, 102, 101, 2005,
    #                    102, 101, 2032, 102, 101, 1010, 102, 101, 1037, 102,
    #                    101, 3800, 102, 101, 2005, 102, 101, 2010, 102, 101,
    #                    2166, 102, 101, 1010, 102, 101, 1998, 102, 101, 2010,
    #                    102, 101, 4650, 102, 101, 1010, 102, 101, 2002, 102,
    #                    101, 2074, 102, 101, 2515, 102, 101, 1050, 102, 101,
    #                    1005, 102, 101, 1056, 102, 101, 2113, 102, 101, 2054,
    #                    102, 101, 1012, 102]])
    # x.requires_grad_()

    model.eval()

    y = model(x['input_ids'], x['attention_mask'])
    print(y)

    cam, _ = model.relprop()

    #print(cam.shape)

    cam = cam.sum(-1)
    #print(cam)
