from transformers import AutoTokenizer

from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

import torch


if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

    explanations = Generator(model)
    classifications = ["NEGATIVE", "POSITIVE"]
    
    text_batch = ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great."]
    encoding = tokenizer(text_batch, return_tensors='pt')
    input_ids = encoding['input_ids'].to("cuda")
    attention_mask = encoding['attention_mask'].to("cuda")
    
    true_class = 1
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    expl = (expl - expl.min()) / (expl.max() - expl.min()) # normalize
    
    print(f"explanable weights : {expl}\n\n")
    
    output = torch.nn.functional.softmax(
        model(input_ids=input_ids, attention_mask=attention_mask)[0],
        dim=-1
    )
    classification = output.argmax(dim=-1).item()
    