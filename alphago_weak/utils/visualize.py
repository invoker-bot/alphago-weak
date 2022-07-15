# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2022/7/14
@Version : 1.0
"""

from graphviz import Digraph

from ..model.mcts import MCTSNode


def plot_mcts(node: MCTSNode, filename: str = None, depth: int = None):
    g = Digraph(comment="visualization of MCTSNode", filename=filename)
    idx = 0
    g.node("0")

    def _plot_mcts_node(id_: int, node: MCTSNode):
        nonlocal idx
        if node.children is not None:
            for action, child_node in node.children.items():
                idx += 1
                g.node(str(idx))
                if id_ != 0:
                    g.edge(str(id_), str(idx), label=str(action))
                    _plot_mcts_node(idx, child_node)

    _plot_mcts_node(0, node)
    g.view()
