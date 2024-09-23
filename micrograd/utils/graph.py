from micrograd.core import Tensor, Op, OpNode
from micrograd.ops import *
import numpy as np 
import graphviz

def build_graph(node: Tensor):
    nodes = []
    edges = []

    def _build(n: Tensor|OpNode):
        nodes.append(n)
        if type(n) == Tensor:
            if n.source is not None: 
                edges.append((n.source, n))
                _build(n.source)
        else:
            for c in n.inputs:
                edges.append((c, n))
                _build(c)

    _build(node)
    return nodes, edges

def draw_graph(node: Tensor):
    nodes, edges = build_graph(node)
    
    dot = graphviz.Digraph("", graph_attr={'rankdir': 'LR'})

    for n in nodes:
        if type(n) == Tensor:
            if n.data.squeeze().shape != ():
                label = f"{n.name}\n{n.data.shape} | {n.grad.shape}"
            else:
                label = f"{n.name}\n{n.data.item():.2f} | {n.grad.item():.2f}"
            color = "white"
            shape = "box"
        else:
            label = f"{n.name}"
            color = "white"
            shape = "ellipse"

        dot.node(str(id(n))+n.name, label=label,
                 fillcolor=color, style="filled", shape=shape, margin='0')
        
    for a, b in edges:
        dot.edge(str(id(a))+a.name, str(id(b))+b.name)

    return dot
