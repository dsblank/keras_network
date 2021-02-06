# -*- coding: utf-8 -*-
# ******************************************************
# keras_network: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/dsblank/keras_network
#
# ******************************************************

import io

import numpy as np


def minimum(seq):
    """
    Find the minimum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The minimum value in list or matrix.

    >>> minimum([5, 2, 3, 1])
    1
    >>> minimum([[5, 2], [3, 1]])
    1
    >>> minimum([[[5], [2]], [[3], [1]]])
    1
    """
    try:
        seq[0][0]
        return min([minimum(v) for v in seq])
    except Exception:
        return np.array(seq).min()


def maximum(seq):
    """
    Find the maximum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The maximum value in list or matrix.

    >>> maximum([0.5, 0.2, 0.3, 0.1])
    0.5
    >>> maximum([[0.5, 0.2], [0.3, 0.1]])
    0.5
    >>> maximum([[[0.5], [0.2]], [[0.3], [0.1]]])
    0.5
    """
    try:
        seq[0][0]
        return max([maximum(v) for v in seq])
    except Exception:
        return np.array(seq).max()


def gather_nodes(layers):
    nodes = []
    for layer in layers:
        for node in layer.inbound_nodes:
            if node not in nodes:
                nodes.append(node)

        for node in layer.outbound_nodes:
            if node not in nodes:
                nodes.append(node)
    return nodes


def topological_sort(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    nodes = topological_sort_nodes(layers)
    layer_list = []
    for node in nodes:
        if hasattr(node.inbound_layers, "__iter__"):
            for layer in node.inbound_layers:
                if layer not in layer_list:
                    layer_list.append(layers)
        else:
            if node.inbound_layers not in layer_list:
                layer_list.append(node.inbound_layers)

        if node.outbound_layer not in layer_list:
            layer_list.append(node.outbound_layer)

    return layer_list


def topological_sort_nodes(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    # Initilize all:
    nodes = gather_nodes(layers)
    for node in nodes:
        node.visited = False
    stack = []
    for node in nodes:
        if not node.visited:
            visit_node(node, stack)
    stack.reverse()
    return stack


def visit_node(node, stack):
    """
    Utility function for topological_sort.
    """
    node.visited = True
    if node.outbound_layer:
        for subnode in node.outbound_layer.outbound_nodes:
            if not subnode.visited:
                visit_node(subnode, stack)
    stack.append(node)


def scale_output_for_image(vector, minmax, truncate=False):
    """
    Given an activation name (or something else) and an output
    vector, scale the vector.
    """
    return rescale_numpy_array(vector, minmax, (0, 255), "uint8", truncate=truncate,)


def rescale_numpy_array(a, old_range, new_range, new_dtype, truncate=False):
    """
    Given a numpy array, old min/max, a new min/max and a numpy type,
    create a new numpy array that scales the old values into the new_range.

    >>> import numpy as np
    >>> new_array = rescale_numpy_array(np.array([0.1, 0.2, 0.3]), (0, 1), (0.5, 1.), float)
    >>> ", ".join(["%.2f" % v for v in new_array])
    '0.55, 0.60, 0.65'
    """
    assert isinstance(old_range, (tuple, list)) and isinstance(new_range, (tuple, list))
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        if truncate:
            a = np.clip(a, old_min, old_max)
        else:
            raise Exception("array values are outside range %s" % (old_range,))
    new_min, new_max = new_range
    old_delta = float(old_max - old_min)
    new_delta = float(new_max - new_min)
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max) / 2).astype(new_dtype)
    else:
        return (new_min + (a - old_min) * new_delta / old_delta).astype(new_dtype)


def svg_to_image(svg, background=(255, 255, 255, 255)):
    import cairosvg
    from PIL import Image

    if isinstance(svg, bytes):
        pass
    elif isinstance(svg, str):
        svg = svg.encode("utf-8")
    else:
        raise Exception("svg_to_image takes a str, rather than %s" % type(svg))

    image_bytes = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image_bytes))
    if background is not None:
        # create a blank image, with background:
        canvas = Image.new("RGBA", image.size, background)
        try:
            canvas.paste(image, mask=image)
        except Exception:
            canvas = None  # fails on images that don't need backgrounds
        if canvas:
            return canvas
        else:
            return image
    else:
        return image


def get_templates(config):
    # Define the SVG strings:
    image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:{{border_color}};stroke-width:{{border_width}}"/><image id="{svg_id}_{{name}}_{{svg_counter}}" class="{svg_id}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" preserveAspectRatio="none" image-rendering="optimizeSpeed" xlink:href="{{image}}"><title>{{tooltip}}</title></image>""".format(
        **config
    )
    line_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}"><title>{{tooltip}}</title></line>""".format(
        **config
    )
    arrow_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)"><title>{{tooltip}}</title></line>""".format(
            **config
    )
    curve_svg = """" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)" fill="none" />""".format(
        **config
    )
    arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
    label_svg = """<text x="{x}" y="{y}" font-family="{font_family}" font-size="{font_size}" text-anchor="{text_anchor}" fill="{font_color}" alignment-baseline="central" {transform}>{label}</text>"""
    svg_head = """<svg id='{svg_id}' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' image-rendering="pixelated" width="{top_width}px" height="{top_height}px">
 <g {svg_transform}>
  <svg viewBox="0 0 {viewbox_width} {viewbox_height}" width="{width}px" height="{height}px">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="{arrow_color}" />
        </marker>
    </defs>"""
    templates = {
        "image_svg": image_svg,
        "line_svg": line_svg,
        "arrow_svg": arrow_svg,
        "arrow_rect": arrow_rect,
        "label_svg": label_svg,
        "svg_head": svg_head,
        "curve": curve_svg,
    }
    return templates
