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
model.add(NFPLayer(2048, 62, 5, input_shape=(24586, )))
model.add(Dense(1))
....
```

# TODO
* support batch learning
* visualize
