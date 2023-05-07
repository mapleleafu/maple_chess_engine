import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inp = Input((21,))
l1 = Dense(128, activation='relu')(inp)
l2 = Dense(128, activation='relu')(l1)
l3 = Dense(128, activation='relu')(l2)
l4 = Dense(128, activation='relu')(l3)
l5 = Dense(128, activation='relu')(l4)

policyOut = Dense(28, name='policyHead', activation='softmax')(l5)
valueOut = Dense(1, name='valueHead', activation='tanh')(l5)

bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
model = Model(inp, [policyOut, valueOut])
model.compile(optimizer='SGD', loss={'valueHead': 'mean_squared_error', 'policyHead': bce})

model.save('random_model.keras')
