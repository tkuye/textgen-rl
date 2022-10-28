from transformers import T5ForConditionalGeneration, BertForSequenceClassification
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, file_path="t5-small"):
        self.model = T5ForConditionalGeneration.from_pretrained(file_path)
        super(Policy, self).__init__()

    def sample(self, state):
        outputs = self.model(state)
        logits = outputs.logits
        logits_softmax = F.softmax(logits, dim=-1)
        action = logits_softmax.argmax()
        action_logprob = logits_softmax.max()
        return action.detach().cpu(), action_logprob.detach().cpu()

    def forward(self, state):
        raise NotImplementedError


class QNetwork(nn.Module):
    def __init__(self, file_path="t5-small"):
        self.q1 = BertForSequenceClassification.from_pretrained(file_path, num_labels=1)
        self.q2 = BertForSequenceClassification.from_pretrained(file_path, num_labels=1)
        super(QNetwork, self).__init__()

    def forward(self, actions, states):
        state_actions = torch.cat([states, actions], 1)
        q1 = self.q1(state_actions)
        q2 = self.q2(state_actions)
        return q1, q2
