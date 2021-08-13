import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class model_encdec(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        channel_in = 2


        # encoder-decoder
        self.d_model = d_model = 512
        dropout = 0.1

        self.srcEmbed = nn.Sequential(LinearEmbedding(channel_in,d_model), PositionalEncoding(d_model, dropout))
        self.tgtEmbed = nn.Sequential(LinearEmbedding(channel_in,d_model), PositionalEncoding(d_model, dropout))

        self.transformer_model = nn.Transformer(d_model=self.d_model, nhead=16, num_encoder_layers=12)
        self.FC_output = torch.nn.Linear(d_model, 2)

        # activation function
        self.relu = nn.ReLU()

        # weight initialization: kaiming
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)
        # nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        # nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        # nn.init.kaiming_normal_(self.encoder_fut.weight_ih_l0)
        # nn.init.kaiming_normal_(self.encoder_fut.weight_hh_l0)
        # nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        # nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)
        # nn.init.zeros_(self.encoder_past.bias_ih_l0)
        # nn.init.zeros_(self.encoder_past.bias_hh_l0)
        # nn.init.zeros_(self.encoder_fut.bias_ih_l0)
        # nn.init.zeros_(self.encoder_fut.bias_hh_l0)
        # nn.init.zeros_(self.decoder.bias_ih_l0)
        # nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)
        raise

    def forward(self, past, future, src_mask, tgt_mask):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """
        src = torch.cat((past, future), 1)
        src = self.srcEmbed(src)
        src = src.permute(1,0,2)

        tgt = future
        tgt = self.tgtEmbed(tgt)
        tgt = tgt.permute(1,0,2)
        # print(src.size(), tgt.size(), src_mask.size(),tgt_mask.size())
        # raise
        output = self.transformer_model(src, tgt, src_mask, tgt_mask)
        output = output.permute(1, 0, 2)
        return self.FC_output(output)

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
