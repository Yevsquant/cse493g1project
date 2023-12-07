import numpy as np

def create_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def encode_captions(captions, word_to_idx):
    N = len(captions)
    T = 2000
    encoded = np.zeros((N, T))
    encoded[:,0] = 1
    for n in range(N):
        for t in range(T):
            if t < len(captions[n]):
                encoded[i][t+1] = word_to_idx[captions[n][t]]
            elif t == len(captions[n]):
                encoded[n][t+1] = word_to_idx["<END>"]
            else:
                break
    return encoded
