import torch

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, x: torch.Tensor) -> str:
        indices = torch.argmax(x, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1) # remove consecutive duplicates
        indices = [i for i in indices if i != self.blank] # remove blanks
        return "".join([self.labels[i] for i in indices]) # indices -> characters
    

class CTCBeamSearchDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0, beam_width=10):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.beam_width = beam_width

    def forward(self, x: torch.Tensor) -> str:
        pass