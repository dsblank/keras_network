# -*- coding: utf-8 -*-
# ******************************************************
# kerasnet: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/dsblank/kerasnet
#
# ******************************************************

import html
import itertools
import math
import operator
from functools import reduce

import numpy as np
import tensorflow.keras.backend as K
from matplotlib import cm
from PIL import Image, ImageDraw

from .utils import (
    get_error_colormap,
    get_templates,
    image_to_uri,
    render_curve,
    scale_output_for_image,
    svg_to_image,
    topological_sort,
)


class Network:
    """
    Wrapper around a keras.Model.
    """

    def __init__(self, model):
        self._model = model
        # Get all of the layers, even implicit ones, in order:
        self._layers = topological_sort(self._model.layers)
        # Make a mapping of names to layers:
        self._layers_map = {layer.name: layer for layer in self._layers}
        # Get the input bank names, in order:
        self.input_bank_order = self._get_input_layers()
        self.num_input_layers = len(self.input_bank_order)
        # Get the best (shortest path) between layers:
        self._level_ordering = self._get_level_ordering()
        self._svg_counter = 0
        self.minmax = (0, 0)
        self.max_draw_units = 20
        self.config = {
            "name": "Keras Network",  # for svg title
            "class_id": "keras-network",  # for svg network classid
            "svg_id": "keras-network",  # for svg id
            "font_size": 12,  # for svg
            "font_family": "monospace",  # for svg
            "border_top": 25,  # for svg
            "border_bottom": 25,  # for svg
            "hspace": 150,  # for svg
            "vspace": 30,  # for svg, arrows
            "image_maxdim": 200,  # for svg
            "image_pixels_per_unit": 50,  # for svg
            "activation": "linear",  # Dense default, if none specified
            "arrow_color": "black",
            "arrow_width": "2",
            "border_width": "2",
            "border_color": "black",
            "show_targets": False,
            "show_error": False,
            "pixels_per_unit": 1,
            "precision": 2,
            "svg_scale": None,  # for svg, 0 - 1, or None for optimal
            "svg_rotate": False,  # for rotating SVG
            "svg_smoothing": 0.02,  # smoothing curves
            "svg_preferred_size": 400,  # in pixels
            "svg_max_width": 800,  # in pixels
            "dashboard.dataset": "Train",
            "dashboard.features.bank": "",
            "dashboard.features.columns": 3,
            "dashboard.features.scale": 1.0,
            "layers": {},
        }

    def __getattr__(self, attr):
        return getattr(self._model, attr)

    def __getitem__(self, layer_name):
        return self._layers_map.get(layer_name, None)

    def make_dummy_vector(self, layer_name, default_value=0.0):
        """
        This is in the easy to use human format (list of lists ...)
        """
        # layer = self[layer_name]
        shape = self._get_output_shape(layer_name)
        # FIXME: for pictures give a vector
        if (shape is None) or (isinstance(shape, (list, tuple)) and None in shape):
            v = np.ones(100) * default_value
        else:
            v = np.ones(shape) * default_value
        lo, hi = self._get_act_minmax(layer_name)
        v *= (lo + hi) / 2.0
        return v

    def make_image(self, layer_name, vector, colormap=None):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        # FIXME:
        # self is a layer from here down:

        # if self.vshape and self.vshape != self.shape:
        #    vector = vector.reshape(self.vshape)
        if len(vector.shape) > 2:
            # Drop dimensions of vector:
            s = slice(None, None)
            args = []
            # The data is in the same format as Keras
            # so we can ask Keras what that format is:
            # ASSUMES: that the network that loaded the
            # dataset has the same image_data_format as
            # now:
            if K.image_data_format() == "channels_last":
                for d in range(len(vector.shape)):
                    if d in [0, 1]:
                        args.append(s)  # keep the first two
                    else:
                        args.append(self._get_feature(layer_name))  # pick which to use
            else:  # 'channels_first'
                count = 0
                for d in range(len(vector.shape)):
                    if d in [0]:
                        args.append(self._get_feature(layer_name))  # pick which to use
                    else:
                        if count < 2:
                            args.append(s)
                            count += 1
            vector = vector[tuple(args)]
        vector = scale_output_for_image(
            vector, self._get_act_minmax(layer_name), truncate=True
        )
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        size = self.config.get("pixels_per_unit", 1)
        new_width = vector.shape[0] * size  # in, pixels
        new_height = vector.shape[1] * size  # in, pixels
        if colormap is None:
            colormap = self._get_colormap(layer_name)
        if colormap is not None:
            try:
                cm_hot = cm.get_cmap(colormap)
            except Exception:
                cm_hot = cm.get_cmap("RdGy")
            vector = cm_hot(vector)
        vector = np.uint8(vector * 255)
        if max(vector.shape) <= self.max_draw_units:
            # Need to make it bigger, to draw circles:
            # Make this value too small, and borders are blocky;
            # too big and borders are too thin
            scale = int(250 / max(vector.shape))
            size = size * scale
            image = Image.new(
                "RGBA", (new_height * scale, new_width * scale), color="white"
            )
            draw = ImageDraw.Draw(image)
            for row in range(vector.shape[1]):
                for col in range(vector.shape[0]):
                    # upper-left, lower-right:
                    draw.rectangle(
                        (
                            row * size,
                            col * size,
                            (row + 1) * size - 1,
                            (col + 1) * size - 1,
                        ),
                        fill=tuple(vector[col][row]),
                        outline="black",
                    )
        else:
            image = Image.fromarray(vector)
            image = image.resize((new_height, new_width))
        # If rotated, and has features, rotate it:
        if self.config.get("svg_rotate", False):
            output_shape = self._get_output_shape(layer_name)
            if (isinstance(output_shape, tuple) and len(output_shape) >= 3) or (
                self.vshape is not None and len(self.vshape) == 2
            ):
                image = image.rotate(90, expand=1)
        return image

    def _get_input_layers(self):
        return [x.name for x in self._layers if self._get_layer_type(x.name) == "input"]

    def _get_output_shape(self, layer_name):
        layer = self[layer_name]
        if isinstance(layer.output_shape, list):
            return layer.output_shape[0][1:]
        else:
            return layer.output_shape[1:]

    def _get_feature(self, layer_name):
        """
        Which feature plane is selected to show? Defaults to 0
        """
        return 0

    def _get_keep_aspect_ratio(self, layer_name):
        return False

    def _get_tooltip(self, layer_name):
        """
        String (with newlines) for describing layer."
        """

        def format_range(minmax):
            minv, maxv = minmax
            if minv <= -2:
                minv = "-Infinity"
            if maxv >= +2:
                maxv = "+Infinity"
            return "(%s, %s)" % (minv, maxv)

        layer = self[layer_name]
        kind = self._get_layer_type(layer_name)
        activation = self._get_activation_name(layer)
        retval = "Layer: %r" % html.escape(self[layer_name].name)
        retval += "\nType: %s" % kind
        if activation:
            retval += "\nActivation: %s" % activation
        retval += "\nOutput range: %s" % (
            format_range(self._get_act_minmax(layer_name),)
        )
        # if self.shape:
        #    retval += "\n shape = %s" % (self.shape, )
        # if self.dropout:
        #    retval += "\n dropout = %s" % self.dropout
        #    if self.dropout_dim > 0:
        #        retval += "\n dropout dimension = %s" % self.dropout_dim
        # if self.bidirectional:
        #    retval += "\n bidirectional = %s" % self.bidirectional
        # if kind == "input":
        #    retval += "\nClass = Input"
        # else:
        retval += "\nClass = %s" % layer.__class__.__name__
        # for key in self.params:
        #    if key in ["name"] or self.params[key] is None:
        #        continue
        #    retval += "\n %s = %s" % (key, html.escape(str(self.params[key])))
        return retval

    def _get_visible(self, layer_name):
        return True

    def _get_colormap(self, layer_name):
        return cm.get_cmap("RdGy")

    def _get_activation_name(self, layer):
        if hasattr(layer, "activation"):
            # names = layer.activation._keras_api_names
            # if len(names) > 0 and "." in names[0]:
            #    names[0].split(".")[-1]
            return layer.activation.__name__

    def _get_act_minmax(self, layer_name):
        """
        Get the activation (output) min/max for a layer.

        Note: +/- 2 represents infinity
        """
        # if self.minmax is not None: # allow override
        #    return self.minmax
        # else:
        if True:
            layer = self[layer_name]
            if layer.__class__.__name__ == "Flatten":
                in_layer = self.incoming_layers(layer_name)[0]
                return self._get_act_minmax(in_layer.name)
            elif self._get_layer_type(layer_name) == "input":
                # try to get from dataset
                # if self.network and len(self.network.dataset) > 0:
                #    bank_idx = self.network.input_bank_order.index(self.name)
                #    return self.network.dataset._inputs_range[bank_idx]
                # else:
                return (-2, +2)
            else:  # try to get from activation function
                activation = self._get_activation_name(layer)
                if activation in ["tanh", "softsign"]:
                    return (-1, +1)
                elif activation in ["sigmoid", "softmax", "hard_sigmoid"]:
                    return (0, +1)
                elif activation in ["relu", "elu", "softplus"]:
                    return (0, +2)
                elif activation in ["selu", "linear"]:
                    return (-2, +2)
                else:  # default, or unknown activation function
                    # Enhancement:
                    # Someday could sample the unknown activation function
                    # and provide reasonable values
                    return (-2, +2)

    def _get_border_color(self, layer_name):
        if self.config.get("highlights") and layer_name in self.config.get(
            "highlights"
        ):
            return self.config["highlights"][layer_name]["border_color"]
        else:
            return self.config["border_color"]

    def _get_border_width(self, layer_name):
        if self.config.get("highlights") and layer_name in self.config.get(
            "highlights"
        ):
            return self.config["highlights"][layer_name]["border_width"]
        else:
            return self.config["border_width"]

    def _find_spacing(self, row, ordering, max_width):
        """
        Find the spacing for a row number
        """
        return max_width / (len(ordering[row]) + 1)

    def describe_connection_to(self, layer1, layer2):
        """
        Returns a textual description of the weights for the SVG tooltip.
        """
        retval = "Weights from %s to %s" % (layer1.name, layer2.name)
        for klayer in self._layers:
            if klayer.name == layer2.name:
                weights = klayer.get_weights()
                for w in range(len(klayer.weights)):
                    retval += "\n %s has shape %s" % (
                        klayer.weights[w].name,
                        weights[w].shape,
                    )
        return retval

    def picture(
        self,
        inputs=None,
        targets=None,
        show_error=False,
        show_targets=False,
        format="svg",
    ):
        """
        Create an SVG of the network given some inputs (optional).

        Arguments:
            inputs: input values to propagate
            rotate (bool): rotate picture to horizontal
            scale (float): scale the picture
            show_error (bool): show the output error in resulting picture
            show_targets (bool): show the targets in resulting picture
            format (str): "html", "image", or "svg"
            minmax (tuple): provide override for input range (layer 0 only)

        Examples:
            >>> net = Network("Picture", 2, 2, 1)
            >>> net.compile(error="mse", optimizer="adam")
            >>> net.picture()
            <IPython.core.display.HTML object>
            >>> net.picture([.5, .5])
            <IPython.core.display.HTML object>
            >>> net.picture([.5, .5])
            <IPython.core.display.HTML object>
        """
        svg = self.to_svg(
            inputs=inputs, class_id=self.config["class_id"], targets=targets
        )
        if format == "svg":
            return svg
        elif format == "pil":
            return svg_to_image(svg)

    def to_svg(self, inputs=None, class_id=None, targets=None):
        """
        """
        struct = self.build_struct(inputs, class_id, targets)
        templates = get_templates(self.config)
        # get the header:
        svg = None
        for (template_name, dict) in struct:
            if template_name == "svg_head":
                svg = templates["svg_head"].format(**dict)
        # build the rest:
        for index in range(len(struct)):
            (template_name, dict) = struct[index]
            if template_name != "svg_head" and not template_name.startswith("_"):
                rotate = dict.get("rotate", self.config["svg_rotate"])
                if template_name == "label_svg" and rotate:
                    dict["x"] += 8
                    dict["text_anchor"] = "middle"
                    dict["transform"] = (
                        """ transform="rotate(-90 %s %s) translate(%s)" """
                        % (dict["x"], dict["y"], 2)
                    )
                else:
                    dict["transform"] = ""
                if template_name == "curve":
                    if not dict["drawn"]:
                        curve_svg = render_curve(
                            dict,
                            struct[(index + 1) :],  # noqa: E203
                            templates[template_name],
                            self.config,
                        )
                        svg += curve_svg
                else:
                    t = templates[template_name]
                    svg += t.format(**dict)
        svg += """</svg></g></svg>"""
        return svg

    def build_struct(self, inputs, class_id, targets):
        ordering = list(
            reversed(self._level_ordering)
        )  # list of names per level, input to output
        (
            max_width,
            max_height,
            row_heights,
            images,
            image_dims,
        ) = self._pre_process_struct(inputs, ordering, targets)
        # Now that we know the dimensions:
        struct = []
        cheight = self.config["border_top"]  # top border
        # Display targets?
        if self.config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of targets:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]:  # no anchors in output
                if layer_name + "_targets" not in images:
                    continue
                image = images[layer_name + "_targets"]
                (width, height) = image_dims[layer_name]
                cwidth += spacing - width / 2
                struct.append(
                    [
                        "image_svg",
                        {
                            "name": layer_name + "_targets",
                            "svg_counter": self._svg_counter,
                            "x": cwidth,
                            "y": cheight,
                            "image": image_to_uri(image),
                            "width": width,
                            "height": height,
                            "tooltip": self._get_tooltip(layer_name),
                            "border_color": self._get_border_color(layer_name),
                            "border_width": self._get_border_width(layer_name),
                            "rx": cwidth - 1,  # based on arrow width
                            "ry": cheight - 1,
                            "rh": height + 2,
                            "rw": width + 2,
                        },
                    ]
                )
                # show a label
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": cwidth + width + 5,
                            "y": cheight + height / 2 + 2,
                            "label": "targets",
                            "font_size": self.config["font_size"],
                            "font_color": "black",
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                cwidth += width / 2
            # Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10  # max height of row, plus some
        # Display error
        if self.config["show_error"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of errores:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]:  # no anchors in output
                if layer_name + "_errors" not in images:
                    continue
                image = images[layer_name + "_errors"]
                (width, height) = image_dims[layer_name]
                cwidth += spacing - (width / 2)
                struct.append(
                    [
                        "image_svg",
                        {
                            "name": layer_name + "_errors",
                            "svg_counter": self._svg_counter,
                            "x": cwidth,
                            "y": cheight,
                            "image": image_to_uri(image),
                            "width": width,
                            "height": height,
                            "tooltip": self._get_tooltip(layer_name),
                            "border_color": self._get_border_color(layer_name),
                            "border_width": self._get_border_width(layer_name),
                            "rx": cwidth - 1,  # based on arrow width
                            "ry": cheight - 1,
                            "rh": height + 2,
                            "rw": width + 2,
                        },
                    ]
                )
                # show a label
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": cwidth + width + 5,
                            "y": cheight + height / 2 + 2,
                            "label": "error",
                            "font_size": self.config["font_size"],
                            "font_color": "black",
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                cwidth += width / 2
            # Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10  # max height of row, plus some
        # Show a separator that takes no space between output and targets/errors
        if self.config["show_error"] or self.config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # Draw a line for each column in putput:
            cwidth = spacing / 2 + spacing / 2  # border + middle of first column
            # number of columns:
            for level_tups in ordering[0]:
                struct.append(
                    [
                        "line_svg",
                        {
                            "x1": cwidth - spacing / 2,
                            "y1": cheight - 5,  # half the space between them
                            "x2": cwidth + spacing / 2,
                            "y2": cheight - 5,
                            "arrow_color": "green",
                            "tooltip": "",
                        },
                    ]
                )
                cwidth += spacing
        # Now we go through again and build SVG:
        positioning = {}
        level_num = 0
        # For each level:
        hiding = {}
        for row in range(len(ordering)):
            level_tups = ordering[row]
            # how many space at this level for this column?
            spacing = self._find_spacing(row, ordering, max_width)
            cwidth = 0
            # See if there are any connections up:
            any_connections_up = False
            for (layer_name, anchor, fname) in level_tups:
                if not self._get_visible(layer_name):
                    continue
                elif anchor:
                    continue
                for out in self.outgoing_layers(layer_name):
                    if (
                        out.name not in positioning
                    ):  # is it drawn yet? if not, continue,
                        # if yes, we need vertical space for arrows
                        continue
                    any_connections_up = True
            if any_connections_up:
                cheight += self.config["vspace"]  # for arrows
            else:  # give a bit of room:
                # FIXME: determine if there were spaces drawn last layer
                # Right now, just skip any space at all
                # cheight += 5
                pass
            row_height = 0  # for row of images
            # Draw each column:
            for column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self._get_visible(layer_name):
                    if not hiding.get(
                        column, False
                    ):  # not already hiding, add some space:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": cwidth + spacing - 80,  # center the text
                                    "y": cheight + 15,
                                    "label": "[layer(s) not visible]",
                                    "font_size": self.config["font_size"],
                                    "font_color": "green",
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                    "rotate": False,
                                },
                            ]
                        )
                        row_height = max(row_height, self.config["vspace"])
                    hiding[column] = True
                    cwidth += spacing  # leave full column width
                    continue
                # end run of hiding
                hiding[column] = False
                # Anchor
                if anchor:
                    anchor_name = "%s-%s-anchor%s" % (layer_name, fname, level_num)
                    cwidth += spacing
                    positioning[anchor_name] = {
                        "x": cwidth,
                        "y": cheight + row_heights[row],
                    }
                    x1 = cwidth
                    # now we are at an anchor. Is the thing that it anchors in the
                    # lower row? level_num is increasing
                    prev = [
                        (oname, oanchor, lfname)
                        for (oname, oanchor, lfname) in ordering[level_num - 1]
                        if (
                            ((layer_name == oname) and (oanchor is False))
                            or (
                                (layer_name == oname)
                                and (oanchor is True)
                                and (fname == lfname)
                            )
                        )
                    ]
                    if prev:
                        tooltip = html.escape(
                            self.describe_connection_to(self[fname], self[layer_name])
                        )
                        if prev[0][1]:  # anchor
                            anchor_name2 = "%s-%s-anchor%s" % (
                                layer_name,
                                fname,
                                level_num - 1,
                            )
                            # draw a line to this anchor point
                            x2 = positioning[anchor_name2]["x"]
                            y2 = positioning[anchor_name2]["y"]
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": anchor_name2,
                                        "x1": cwidth,
                                        "y1": cheight,
                                        "x2": x2,
                                        "y2": y2,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": anchor_name2,
                                        "x1": cwidth,
                                        "y1": cheight + row_heights[row],
                                        "x2": cwidth,
                                        "y2": cheight,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                        else:
                            # draw a line to this bank
                            x2 = (
                                positioning[layer_name]["x"]
                                + positioning[layer_name]["width"] / 2
                            )
                            y2 = (
                                positioning[layer_name]["y"]
                                + positioning[layer_name]["height"]
                            )
                            tooltip = "TODO"
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": True,
                                        "drawn": False,
                                        "name": layer_name,
                                        "x1": cwidth,
                                        "y1": cheight,
                                        "x2": x2,
                                        "y2": y2,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                            struct.append(
                                [
                                    "curve",
                                    {
                                        "endpoint": False,
                                        "drawn": False,
                                        "name": layer_name,
                                        "x1": cwidth,
                                        "y1": cheight + row_heights[row],
                                        "x2": cwidth,
                                        "y2": cheight,
                                        "arrow_color": self.config["arrow_color"],
                                        "tooltip": tooltip,
                                    },
                                ]
                            )
                    else:
                        print("that's weird!", layer_name, "is not in", prev)
                    continue
                else:
                    # Bank positioning
                    image = images[layer_name]
                    (width, height) = image_dims[layer_name]
                    cwidth += spacing - (width / 2)
                    positioning[layer_name] = {
                        "name": layer_name
                        + ("-rotated" if self.config["svg_rotate"] else ""),
                        "svg_counter": self._svg_counter,
                        "x": cwidth,
                        "y": cheight,
                        "image": image_to_uri(image),
                        "width": width,
                        "height": height,
                        "tooltip": self._get_tooltip(layer_name),
                        "border_color": self._get_border_color(layer_name),
                        "border_width": self._get_border_width(layer_name),
                        "rx": cwidth - 1,  # based on arrow width
                        "ry": cheight - 1,
                        "rh": height + 2,
                        "rw": width + 2,
                    }
                    x1 = cwidth + width / 2
                y1 = cheight - 1
                # Arrows going up
                for out in self.outgoing_layers(layer_name):
                    if out.name not in positioning:
                        continue
                    # draw an arrow between layers:
                    anchor_name = "%s-%s-anchor%s" % (
                        out.name,
                        layer_name,
                        level_num - 1,
                    )
                    # Don't draw this error, if there is an anchor in the next level
                    if anchor_name in positioning:
                        tooltip = html.escape(
                            self.describe_connection_to(self[layer_name], out)
                        )
                        x2 = positioning[anchor_name]["x"]
                        y2 = positioning[anchor_name]["y"]
                        struct.append(
                            [
                                "curve",
                                {
                                    "endpoint": False,
                                    "drawn": False,
                                    "name": anchor_name,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "arrow_color": self.config["arrow_color"],
                                    "tooltip": tooltip,
                                },
                            ]
                        )
                        continue
                    else:
                        tooltip = html.escape(
                            self.describe_connection_to(self[layer_name], out)
                        )
                        x2 = (
                            positioning[out.name]["x"]
                            + positioning[out.name]["width"] / 2
                        )
                        y2 = (
                            positioning[out.name]["y"] + positioning[out.name]["height"]
                        )
                        struct.append(
                            [
                                "curve",
                                {
                                    "endpoint": True,
                                    "drawn": False,
                                    "name": out.name,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2 + 2,
                                    "arrow_color": self.config["arrow_color"],
                                    "tooltip": tooltip,
                                },
                            ]
                        )
                # Bank images
                struct.append(["image_svg", positioning[layer_name]])
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": positioning[layer_name]["x"]
                            + positioning[layer_name]["width"]
                            + 5,
                            "y": positioning[layer_name]["y"]
                            + positioning[layer_name]["height"] / 2
                            + 2,
                            "label": layer_name,
                            "font_size": self.config["font_size"],
                            "font_color": "black",
                            "font_family": self.config["font_family"],
                            "text_anchor": "start",
                        },
                    ]
                )
                output_shape = self._get_output_shape(layer_name)
                # FIXME: how to determine a layer that has images as input?
                if (
                    isinstance(output_shape, tuple)
                    and len(output_shape) == 4
                    and self[layer_name].__class__.__name__ != "ImageLayer"
                ):
                    features = str(output_shape[3])
                    # FIXME:
                    feature = str(self._get_feature(layer_name))
                    if self.config["svg_rotate"]:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"] + 5,
                                    "y": positioning[layer_name]["y"] - 10 - 5,
                                    "label": features,
                                    "font_size": self.config["font_size"],
                                    "font_color": "black",
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    + positioning[layer_name]["width"]
                                    - 10,
                                    "y": positioning[layer_name]["y"]
                                    + positioning[layer_name]["height"]
                                    + 10
                                    + 5,
                                    "label": feature,
                                    "font_size": self.config["font_size"],
                                    "font_color": "black",
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                    else:
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    + positioning[layer_name]["width"]
                                    + 5,
                                    "y": positioning[layer_name]["y"] + 5,
                                    "label": features,
                                    "font_size": self.config["font_size"],
                                    "font_color": "black",
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                        struct.append(
                            [
                                "label_svg",
                                {
                                    "x": positioning[layer_name]["x"]
                                    - (len(feature) * 7)
                                    - 5
                                    - 5,
                                    "y": positioning[layer_name]["y"]
                                    + positioning[layer_name]["height"]
                                    - 5,
                                    "label": feature,
                                    "font_size": self.config["font_size"],
                                    "font_color": "black",
                                    "font_family": self.config["font_family"],
                                    "text_anchor": "start",
                                },
                            ]
                        )
                if False:  # (self[layer_name].dropout > 0): FIXME:
                    struct.append(
                        [
                            "label_svg",
                            {
                                "x": positioning[layer_name]["x"]
                                - 1 * 2.0
                                - 18,  # length of chars * 2.0
                                "y": positioning[layer_name]["y"] + 4,
                                "label": "o",  # "&#10683;"
                                "font_size": self.config["font_size"] * 2.0,
                                "font_color": "black",
                                "font_family": self.config["font_family"],
                                "text_anchor": "start",
                            },
                        ]
                    )
                    struct.append(
                        [
                            "label_svg",
                            {
                                "x": positioning[layer_name]["x"]
                                - 1 * 2.0
                                - 15
                                + (
                                    -3 if self.config["svg_rotate"] else 0
                                ),  # length of chars * 2.0
                                "y": positioning[layer_name]["y"]
                                + 5
                                + (-1 if self.config["svg_rotate"] else 0),
                                "label": "x",  # "&#10683;"
                                "font_size": self.config["font_size"] * 1.3,
                                "font_color": "black",
                                "font_family": self.config["font_family"],
                                "text_anchor": "start",
                            },
                        ]
                    )
                cwidth += width / 2
                row_height = max(row_height, height)
                self._svg_counter += 1
            cheight += row_height
            level_num += 1
        cheight += self.config["border_bottom"]
        # DONE!
        # Draw live/static sign
        if False:  # FIXME
            # dynamic image:
            label = "*"
            if self.config["svg_rotate"]:
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": 10,
                            "y": cheight - 10,
                            "label": label,
                            "font_size": self.config["font_size"] * 2.0,
                            "font_color": "red",
                            "font_family": self.config["font_family"],
                            "text_anchor": "middle",
                        },
                    ]
                )
            else:
                struct.append(
                    [
                        "label_svg",
                        {
                            "x": 10,
                            "y": 10,
                            "label": label,
                            "font_size": self.config["font_size"] * 2.0,
                            "font_color": "red",
                            "font_family": self.config["font_family"],
                            "text_anchor": "middle",
                        },
                    ]
                )
        # Draw the title:
        if self.config["svg_rotate"]:
            struct.append(
                [
                    "label_svg",
                    {
                        "x": 10,  # really border_left
                        "y": cheight / 2,
                        "label": self.config["name"],
                        "font_size": self.config["font_size"] + 3,
                        "font_color": "black",
                        "font_family": self.config["font_family"],
                        "text_anchor": "middle",
                    },
                ]
            )
        else:
            struct.append(
                [
                    "label_svg",
                    {
                        "x": max_width / 2,
                        "y": self.config["border_top"] / 2,
                        "label": self.config["name"],
                        "font_size": self.config["font_size"] + 3,
                        "font_color": "black",
                        "font_family": self.config["font_family"],
                        "text_anchor": "middle",
                    },
                ]
            )
        # figure out scale optimal, if scale is None
        # the fraction:
        if self.config["svg_scale"] is not None:  # scale is given:
            if self.config["svg_rotate"]:
                scale_value = (self.config["svg_max_width"] / cheight) * self.config[
                    "svg_scale"
                ]
            else:
                scale_value = (self.config["svg_max_width"] / max_width) * self.config[
                    "svg_scale"
                ]
        else:
            if self.config["svg_rotate"]:
                scale_value = self.config["svg_max_width"] / max(cheight, max_width)
            else:
                scale_value = self.config["svg_preferred_size"] / max(
                    cheight, max_width
                )
        # svg_scale = "%s%%" % int(scale_value * 100)
        scaled_width = max_width * scale_value
        scaled_height = cheight * scale_value
        # Need a top-level width, height because Jupyter peeks at it
        if self.config["svg_rotate"]:
            svg_transform = (
                """ transform="rotate(90) translate(0 -%s)" """ % scaled_height
            )
            # Swap them:
            top_width = scaled_height
            top_height = scaled_width
        else:
            svg_transform = ""
            top_width = scaled_width
            top_height = scaled_height
        struct.append(
            [
                "svg_head",
                {
                    "viewbox_width": int(max_width),  # view port width
                    "viewbox_height": int(cheight),  # view port height
                    "width": int(scaled_width),  # actual pixels of image in page
                    "height": int(scaled_height),  # actual pixels of image in page
                    "svg_id": self.config["svg_id"],
                    "top_width": int(top_width),
                    "top_height": int(top_height),
                    "arrow_color": self.config["arrow_color"],
                    "arrow_width": self.config["arrow_width"],
                    "svg_transform": svg_transform,
                },
            ]
        )
        return struct

    def incoming_layers(self, layer_name):
        layer = self[layer_name]
        layers = []
        for node in layer.inbound_nodes:
            if hasattr(node.inbound_layers, "__iter__"):
                for layer in node.inbound_layers:
                    if layer not in layers:
                        layers.append(layer)
            else:
                if node.inbound_layers not in layers:
                    layers.append(node.inbound_layers)
        return layers

    def outgoing_layers(self, layer_name):
        layer = self[layer_name]
        layers = []
        for node in layer.outbound_nodes:
            if node.outbound_layer not in layers:
                layers.append(node.outbound_layer)
        return layers

    def _get_layer_type(self, layer_name):
        """
        Determines whether a layer is a "input", "hidden", or "output"
        layer based on its connections. If no connections, then it is
        "unconnected".
        """
        incoming_connections = self.incoming_layers(layer_name)
        outgoing_connections = self.outgoing_layers(layer_name)
        if len(incoming_connections) == 0 and len(outgoing_connections) == 0:
            return "unconnected"
        elif len(incoming_connections) > 0 and len(outgoing_connections) > 0:
            return "hidden"
        elif len(incoming_connections) > 0:
            return "output"
        else:
            return "input"

    def _get_level_ordering(self):
        """
        Returns a list of lists of tuples from
        input to output of levels.

        Each tuple contains: (layer_name, anchor?, from_name/None)

        If anchor is True, this is just an anchor point.
        """
        # First, get a level for all layers:
        levels = {}
        for layer in self._layers:
            level = max(
                [levels[lay.name] for lay in self.incoming_layers(layer.name)] + [-1]
            )
            levels[layer.name] = level + 1
        max_level = max(levels.values())
        ordering = []
        for i in range(max_level + 1):  # input to output
            layer_names = [
                layer.name for layer in self._layers if levels[layer.name] == i
            ]
            ordering.append(
                [
                    (name, False, [x.name for x in self.incoming_layers(name)])
                    for name in layer_names
                ]
            )  # (going_to/layer_name, anchor, coming_from)
        # promote all output banks to last row:
        for level in range(len(ordering)):  # input to output
            tuples = ordering[level]
            index = 0
            for (name, anchor, none) in tuples[:]:
                if self._get_layer_type(name) == "output":
                    # move it to last row
                    # find it and remove
                    ordering[-1].append(tuples.pop(index))
                else:
                    index += 1
        # insert anchor points for any in next level
        # that doesn't go to a bank in this level
        # order_cache = {}
        for level in range(len(ordering)):  # input to output
            tuples = ordering[level]
            for (name, anchor, fname) in tuples:
                if anchor:
                    # is this in next? if not add it
                    next_level = [
                        (n, anchor) for (n, anchor, hfname) in ordering[level + 1]
                    ]
                    if (
                        name,
                        False,
                    ) not in next_level:  # actual layer not in next level
                        ordering[level + 1].append(
                            (name, True, fname)
                        )  # add anchor point
                    else:
                        pass  # finally!
                else:
                    # if next level doesn't contain an outgoing
                    # connection, add it to next level as anchor point
                    for layer in self.outgoing_layers(name):
                        next_level = [
                            (n, anchor) for (n, anchor, fname) in ordering[level + 1]
                        ]
                        if (layer.name, False) not in next_level:
                            ordering[level + 1].append(
                                (layer.name, True, name)
                            )  # add anchor point
        ordering = self._optimize_ordering(ordering)
        return ordering

    def _optimize_ordering(self, ordering):
        def perms(l):
            return list(itertools.permutations(l))

        def distance(xy1, xy2):
            return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

        def find_start(cend, canchor, name, plevel):
            """
            Return position and weight of link from cend/name to
            col in previous level.
            """
            col = 1
            for bank in plevel:
                pend, panchor, pstart_names = bank
                if name == pend:
                    if not panchor and not canchor:
                        weight = 10.0
                    else:
                        weight = 1.0
                    return col, weight
                elif cend == pend and name == pstart_names:
                    return col, 5.0
                col += 1
            raise Exception("connecting layer not found!")

        # First level needs to be in bank_order, and cannot permutate:
        first_level = [(bank_name, False, []) for bank_name in self.input_bank_order]
        perm_count = reduce(
            operator.mul, [math.factorial(len(level)) for level in ordering[1:]]
        )
        if perm_count < 70000:  # globally minimize
            permutations = itertools.product(*[perms(x) for x in ordering[1:]])
            # measure arrow distances for them all and find the shortest:
            best = (10000000, None, None)
            for ordering in permutations:
                ordering = (first_level,) + ordering
                sum = 0.0
                for level_num in range(1, len(ordering)):
                    level = ordering[level_num]
                    plevel = ordering[level_num - 1]
                    col1 = 1
                    for bank in level:  # starts at level 1
                        cend, canchor, cstart_names = bank
                        if canchor:
                            cstart_names = [cstart_names]  # put in list
                        for name in cstart_names:
                            col2, weight = find_start(cend, canchor, name, plevel)
                            dist = (
                                distance(
                                    (col1 / (len(level) + 1), 0),
                                    (col2 / (len(plevel) + 1), 0.1),
                                )
                                * weight
                            )
                            sum += dist
                        col1 += 1
                if sum < best[0]:
                    best = (sum, ordering)
            return best[1]
        else:  # locally minimize, between layers:
            ordering[0] = first_level  # replace first level with sorted
            for level_num in range(1, len(ordering)):
                best = (10000000, None, None)
                plevel = ordering[level_num - 1]
                for level in itertools.permutations(ordering[level_num]):
                    sum = 0.0
                    col1 = 1
                    for bank in level:  # starts at level 1
                        cend, canchor, cstart_names = bank
                        if canchor:
                            cstart_names = [cstart_names]  # put in list
                        for name in cstart_names:
                            col2, weight = find_start(cend, canchor, name, plevel)
                            dist = (
                                distance(
                                    (col1 / (len(level) + 1), 0),
                                    (col2 / (len(plevel) + 1), 0.1),
                                )
                                * weight
                            )
                            sum += dist
                        col1 += 1
                    if sum < best[0]:
                        best = (sum, level)
                ordering[level_num] = best[1]
            return ordering

    def vshape(self, layer_name):
        """
        Find the vshape of layer.
        """
        # layer = self[layer_name]
        # vshape = layer.vshape if layer.vshape else layer.shape if layer.shape else None
        # if vshape is None:
        vshape = self._get_output_shape(layer_name)
        return vshape

    def _pre_process_struct(self, inputs, ordering, targets):
        """
        Determine sizes and pre-compute images.
        """
        # find max_width, image_dims, and row_height
        # Go through and build images, compute max_width:
        row_heights = []
        max_width = 0
        max_height = 0
        images = {}
        image_dims = {}
        # if targets, then need to propagate for error:
        if targets is not None:
            outputs = self.predict(inputs)
            if len(self.output_bank_order) == 1:
                targets = [targets]
                errors = (np.array(outputs) - np.array(targets)).tolist()
            else:
                errors = []
                for bank in range(len(self.output_bank_order)):
                    errors.append(
                        (np.array(outputs[bank]) - np.array(targets[bank])).tolist()
                    )
        # For each level:
        hiding = {}
        for level_tups in ordering:  # output to input:
            # first make all images at this level
            row_width = 0  # for this row
            row_height = 0  # for this row
            # For each column:
            for column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self._get_visible(layer_name):
                    if not hiding.get(column, False):
                        row_height = max(
                            row_height, self.config["vspace"]
                        )  # space for hidden indicator
                    hiding[column] = True  # in the middle of hiding some layers
                    row_width += self.config["hspace"]  # space between
                    max_width = max(max_width, row_width)  # of all rows
                    continue
                elif anchor:
                    # No need to handle anchors here
                    # as they occupy no vertical space
                    hiding[column] = False
                    # give it some hspace for this column
                    # in case there is nothing else in this column
                    row_width += self.config["hspace"]
                    max_width = max(max_width, row_width)
                    continue
                hiding[column] = False
                # The rest of this for loop is handling image of bank
                if inputs is not None:
                    v = inputs
                # elif len(self.dataset.inputs) > 0 and not isinstance(self.dataset, VirtualDataset):
                #    # don't change cache if virtual... could take some time to rebuild cache
                #    v = self.dataset.inputs[0]
                # else:
                if True:  # FIXME
                    if self.num_input_layers > 1:
                        v = []
                        for in_name in self.input_bank_order:
                            v.append(self.make_dummy_vector(in_name))
                    else:
                        in_layer = [
                            layer
                            for layer in self._layers
                            if self._get_layer_type(layer.name) == "input"
                        ][0]
                        v = self.make_dummy_vector(in_layer.name)
                if True:  # self[layer_name].model: # FIXME
                    try:
                        image = self._propagate_to_image(layer_name, v)
                    except Exception:
                        image = self.make_image(
                            layer_name, np.array(self.make_dummy_vector(layer_name)),
                        )
                else:
                    self.warn_once(
                        "WARNING: network is uncompiled; activations cannot be visualized"
                    )
                    image = self.make_image(
                        layer_name, np.array(self.make_dummy_vector(layer_name)),
                    )
                (width, height) = image.size
                images[layer_name] = image  # little image
                if self._get_layer_type(layer_name) == "output":
                    if targets is not None:
                        # Target image, targets set above:
                        target_colormap = "grey"  # FIXME: self[layer_name].colormap
                        target_bank = targets[self.output_bank_order.index(layer_name)]
                        target_array = np.array(target_bank)
                        target_image = self.make_image(
                            layer_name, target_array, target_colormap
                        )
                        # Error image, error set above:
                        error_colormap = get_error_colormap()
                        error_bank = errors[self.output_bank_order.index(layer_name)]
                        error_array = np.array(error_bank)
                        error_image = self.make_image(
                            layer_name, error_array, error_colormap
                        )
                        images[layer_name + "_errors"] = error_image
                        images[layer_name + "_targets"] = target_image
                    else:
                        images[layer_name + "_errors"] = image
                        images[layer_name + "_targets"] = image
                # Layer settings:
                # FIXME:
                # if self[layer_name].image_maxdim:
                #    image_maxdim = self[layer_name].image_maxdim
                # else:
                image_maxdim = self.config["image_maxdim"]
                # FIXME:
                # if self[layer_name].image_pixels_per_unit:
                #    image_pixels_per_unit = self[layer_name].image_pixels_per_unit
                # else:
                image_pixels_per_unit = self.config["image_pixels_per_unit"]
                # First, try based on shape:
                # pwidth, pheight = np.array(image.size) * image_pixels_per_unit
                vshape = self.vshape(layer_name)
                if vshape is None or self._get_keep_aspect_ratio(layer_name):
                    pass  # let the image set the shape
                elif len(vshape) == 1:
                    if vshape[0] is not None:
                        width = vshape[0] * image_pixels_per_unit
                        height = image_pixels_per_unit
                elif len(vshape) >= 2:
                    if vshape[0] is not None:
                        height = vshape[0] * image_pixels_per_unit
                        if vshape[1] is not None:
                            width = vshape[1] * image_pixels_per_unit
                    else:
                        if len(vshape) > 2:
                            if vshape[1] is not None:
                                height = vshape[1] * image_pixels_per_unit
                                width = vshape[2] * image_pixels_per_unit
                        elif vshape[1] is not None:  # flatten
                            width = vshape[1] * image_pixels_per_unit
                            height = image_pixels_per_unit
                # keep aspect ratio:
                if self._get_keep_aspect_ratio(layer_name):
                    scale = image_maxdim / max(width, height)
                    image = image.resize((int(width * scale), int(height * scale)))
                    width, height = image.size
                else:
                    # Change aspect ratio if too big/small
                    if width < image_pixels_per_unit:
                        width = image_pixels_per_unit
                    if height < image_pixels_per_unit:
                        height = image_pixels_per_unit
                    # make sure not too big:
                    if height > image_maxdim:
                        height = image_maxdim
                    if width > image_maxdim:
                        width = image_maxdim
                image_dims[layer_name] = (width, height)
                row_width += width + self.config["hspace"]  # space between
                row_height = max(row_height, height)
            row_heights.append(row_height)
            max_width = max(max_width, row_width)  # of all rows
        return max_width, max_height, row_heights, images, image_dims
