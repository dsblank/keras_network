# -*- coding: utf-8 -*-
# ******************************************************
# kerasnet: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/dsblank/kerasnet
#
# ******************************************************

from tensorflow.keras.callbacks import Callback


class PlotCallback(Callback):
    def __init__(self, network, report_rate):
        super().__init__()
        self._network = network
        self._figure = None
        self._history = []

    def on_epoch_end(self, epoch, logs=None):
        self._history.append((epoch, logs))
        self._network.plot_results(self)
