"""Tree attention engine — JetSpec-style causal masks and verification paths."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TreeNode:
    token: int
    parent: int
    children: list[int]


def build_tree(width: int, depth: int, tokens: list[int]) -> list[TreeNode]:
    """Build an explicit tree of draft tokens.
    `tokens` is the flattened BFS order; node 0 is the root.
    """
    if not tokens:
        return []
    nodes: list[TreeNode] = [TreeNode(token=tokens[0], parent=-1, children=[])]
    for i in range(1, len(tokens)):
        parent = max(0, (i - 1) // max(1, width))
        nodes.append(TreeNode(token=tokens[i], parent=parent, children=[]))
        nodes[parent].children.append(i)
    return nodes


def ancestor(node: int, ancestor: int, width: int) -> bool:
    cur = node
    while cur != -1:
        if cur == ancestor:
            return True
        cur = max(0, (cur - 1) // max(1, width))
    return False


class TreeAttentionEngine:
    """Tree-attention mask + attention score gatherer.

    Operates on integer token ids. Real implementation links to mlx_lm and
    builds an explicit block-diagonal mask of shape
    (total_tokens, total_tokens) with 1s in ancestor positions.
    """

    def __init__(self, tree_width: int = 4, tree_depth: int = 2):
        self.w = tree_width
        self.d = tree_depth

    def mask(self, total: int) -> list[list[int]]:
        m = [[0] * total for _ in range(total)]
        for r in range(total):
            for c in range(total):
                if c <= r:
                    m[r][c] = 1
                elif c < self.w ** self.d + 1:
                    m[r][c] = 1 if ancestor(c, r, self.w) else 0
        return m
