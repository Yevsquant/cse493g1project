import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image
import copy

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
        self.register_buffer('pe', pe)

    def forward(self, x):
        N, S, D = x.shape
        output = torch.empty((N, S, D))
        output = x + self.pe[:,:S]
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        N, S, E = query.shape
        N, T, E = value.shape
        output = torch.empty((N, S, E))
        q = self.query(query).reshape(N, S, self.n_head, self.emd_dim // self.n_head)
        k = self.key(key).reshape(N, T, self.n_head, self.emd_dim // self.n_head)
        v = self.value(value).reshape(N, T, self.n_head, self.emd_dim // self.n_head)
        q = torch.permute(q, (0, 2, 1, 3))
        k = torch.permute(k, (0, 2, 3, 1))
        v = torch.permute(v, (0, 2, 1, 3))
        y = torch.matmul(q, k) / torch.sqrt(torch.Tensor([self.head_dim]))
        if attn_mask != None:
          y = y.masked_fill(attn_mask == 0, -math.inf)
        y = F.softmax(y, dim=3)
        y = self.attn_drop(y)
        y = torch.matmul(y, v)
        y = torch.permute(y, (0, 2, 1, 3)).reshape(output.shape)
        output = self.proj(y)
        return output


class GraphCaptioningModel(nn.Module):
    def __init__(self, word_to_idx, wordvec_dim, num_heads=8,
                 num_layers=6, max_length=5000):
        super().__init__()
        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        self.visual_projection = nn.Linear(2048, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)
        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)
        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image, captions):
        N, T = captions.shape
        image_encoding = self.image_encoder(image)
        scores = torch.empty((N, T, self.vocab_size))
        embeds = self.positional_encoding(self.embedding(captions))
        proj_feat = self.visual_projection(image_encoding).reshape(N, -1, embeds.shape[-1])
        tgt_mask = torch.ones(T, T)
        tgt_mask = torch.tril(tgt_mask)
        scores = self.transformer(embeds, proj_feat, tgt_mask)
        scores = self.output(scores)
        return scores

    def sample(self, features, max_length=30):
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]
            captions = self._null * np.ones((N, max_length), dtype=np.int32)
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            partial_caption = partial_caption.unsqueeze(1)
            for t in range(max_length):
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]
                word = torch.argmax(output_logits, axis=1)
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)
            return captions


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)
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
