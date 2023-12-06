
# this is the vocab used in captions, its size is 108
vocab = [i for i in range(100)]
vocab.append('{') # start
vocab.append('}') # end
vocab.append('[')
vocab.append(']')
vocab.append('(')
vocab.append(')')
vocab.append(',')
vocab.append(':')

#image: 512 * 512 * 3