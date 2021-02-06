# -*- coding: utf-8 -*-
# ******************************************************
# kerasnet: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/dsblank/kerasnet
#
# ******************************************************

from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model, Sequential

from kerasnet import Network


def test_squential():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    network = Network(model)

    network.picture()


def test_graph():

    d1a = Input(8)
    d1b = Input(8)
    d1 = Concatenate()([d1a, d1b])
    d2 = Dense(12, input_dim=8, activation="relu")(d1)
    d3 = Dense(8, activation="relu")(d2)
    output = Dense(1, activation="sigmoid")(d3)

    model = Model(inputs=[d1a, d1b], outputs=[output])

    network = Network(model)

    network.picture()
