import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        pe = torch.zeros(1, max_length, embed_dim)
        i = torch.arange(max_length).unsqueeze(1)
        exponents = torch.pow(1e4, -torch.arange(0, embed_dim, 2) / embed_dim)
        pe[0,:,torch.arange(0,embed_dim,2)] = torch.sin(i * exponents)
        pe[0,:,torch.arange(1,embed_dim,2)] = torch.cos(i * exponents)

    def forward(self, x):
        N, S, D = x.shape
        output = torch.empty((N, S, D))
        output = x + self.pe[:,:S]
        output = self.dropout(output)
        return output

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length=5000):
        super(ImageCaptioningModel, self).__init__()

        self.image_encoder = models.resnet50(pretrained=True)
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=2048, nhead=8),
            num_layers=6
        )
        self.embedding = nn.Embedding(vocab_size, 2048)
        self.positional_encoding = PositionalEncoding(max_length, 2048)

    def forward(self, image, caption):
        image_encoding = self.image_encoder(image)
        positional_encoding = self.positional_encoding(torch.arange(0, caption.shape[1]))
        caption_embedding = self.embedding(caption) + positional_encoding.unsqueeze(0)
        decoder_input = caption_embedding.permute(1, 0, 2)
        output = self.text_decoder(decoder_input, image_encoding.unsqueeze(0))
        return output