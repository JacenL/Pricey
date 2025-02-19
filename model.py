from transformers import BertModel, PreTrainedModel
import torch.nn as nn

class PriceEstimator(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :] 
        price_prediction = self.regressor(last_hidden_state)
        return price_prediction
