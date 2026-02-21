import torch
import torch.nn as nn
import torch_dct
import math


# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# define model
class PosePredictModel(nn.Module):
    def __init__(self, input_len, output_len, lstm=False, src_features=34, tgt_features=34, d_model=512, nhead=8,
                 fusion=None):
        super(PosePredictModel, self).__init__()

        # model param
        self.input_len = input_len
        self.output_len = output_len
        self.src_features = src_features
        self.tgt_features = tgt_features
        self.d_model = d_model
        self.nhead = nhead
        self.fusion = fusion

        # encoder embedding layer
        self.encoder_norm = nn.LayerNorm(self.src_features)
        self.encoder_linear = nn.Linear(self.src_features, self.d_model)
        self.encoder_position = PositionalEncoding(self.d_model, dropout=0.1, max_len=self.input_len)
        if lstm:
            self.lstm = nn.LSTM(input_size=self.src_features, hidden_size=64, num_layers=2, dropout=0.2,
                                batch_first=True)
        # decoder embedding layer
        self.decoder_norm = nn.LayerNorm(self.tgt_features)
        self.decoder_linear = nn.Linear(self.tgt_features, self.d_model)
        self.decoder_position = PositionalEncoding(self.d_model, dropout=0.1, max_len=self.output_len)
        # transformer model
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, num_encoder_layers=2,
                                          num_decoder_layers=2, batch_first=True)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.output_len)
        self.out_linear = nn.Linear(self.d_model, self.tgt_features)

        if fusion == 'MLP':
            self.fusion_linear = nn.Linear(self.src_features, self.d_model)
        elif fusion == 'DCT':
            self.fusion_linear = nn.Linear(self.src_features, self.d_model)

    def forward(self, src, tgt):
        # encoder input x
        src = self.encoder_norm(src)
        if self.fusion == 'DCT':
            src = src.view(self.input_len, self.src_features//2, 2)
            src = torch_dct.dct_2d(src, norm='ortho')
            src = src.view(self.input_len, self.src_features)
        x = self.encoder_linear(src)  # src:(input_len, src_features) -> x:(input_len, d_model)
        x = self.encoder_position(x)
        # decoder input y
        # tgt = src[-output_len:]
        tgt = self.decoder_norm(tgt)
        if self.fusion == 'DCT':
            tgt = tgt.view(self.output_len, self.tgt_features//2, 2)
            tgt = torch_dct.dct_2d(tgt, norm='ortho')
            tgt = tgt.view(self.output_len, self.tgt_features)
        y = self.decoder_linear(tgt)  # tgt:(output_len, tgt_features) -> y:(output_len, d_model)
        y = self.decoder_position(y)
        output = self.out_linear(self.transformer(x, y, tgt_mask=self.tgt_mask))
        return output
