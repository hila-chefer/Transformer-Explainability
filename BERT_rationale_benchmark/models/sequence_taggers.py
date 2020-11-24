import torch
import torch.nn as nn

from typing import List, Tuple, Any

from transformers import BertModel

from rationale_benchmark.models.model_utils import PaddedSequence


class BertTagger(nn.Module):
    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertTagger, self).__init__()
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        bert = BertModel.from_pretrained(bert_dir)
        if use_half_precision:
            import apex
            bert = bert.half()
        self.bert = bert
        self.relevance_tagger = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor],
                aggregate_spans: List[Tuple[int, int]]):
        assert len(query) == len(document_batch)
        # note about device management: since distributed training is enabled, the inputs to this module can be on
        # *any* device (preferably cpu, since we wrap and unwrap the module) we want to keep these params on the
        # input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        #cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        query_lengths = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 1 > self.max_length:
                d = d[:(self.max_length - len(q) - 1)]
            input_tensors.append(torch.cat([q, sep_token, d]))
            query_lengths.append(q.size()[0])
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        outputs = self.bert(bert_input.data, attention_mask=bert_input.mask(on=0.0, off=float('-inf'), dtype=torch.float, device=target_device))
        hidden = outputs[0]
        classes = self.relevance_tagger(hidden)
        ret = []
        for ql, cls, doc in zip(query_lengths, classes, document_batch):
            start = ql + 1
            end = start + len(doc)
            ret.append(cls[ql + 1:end])
        return PaddedSequence.autopad(ret, batch_first=True, padding_value=0, device=target_device).data.squeeze(dim=-1)
