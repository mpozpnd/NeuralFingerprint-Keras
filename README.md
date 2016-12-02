# NeuralFingerprint-Keras
Implementation of NeuralFingerprint(https://arxiv.org/pdf/1509.09292.pdf)

# requirement
* keras (with theano as backend)
* rdkit

# Usage
```
from nfp_keras import NFPLayer
from nfp_keras.features import gen_feature_vector_from_sdf

from keras.models import Sequential
from keras.layers import Dense

data_x = gen_feature_vector_from_sdf('hogepiyo.sdf')

model = Sequential()
# you should pass "batch_input_shape".Since NFPLayer must know batch_size before model.compile.
model.add(NFPLayer(2048, 62, 5, batch_input_shape=(batch_size, 24586,)))
model.add(Dense(1))
....
```

# TODO
* support batch learning
* visualize

some codes in features.py are borrow from original implementation (https://github.com/HIPS/neural-fingerprint) 
