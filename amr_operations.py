"""
Manipulate AMR graphs, for example extracting subgraphs.
networkx doesn't seem to simplify things.
"""

from typing import *
from itertools import product

import networkx as nx

from amr_format import AMRNode, AMRVariable, AMRConstant, AMREdge, AMRRef


class AMRGraph:
    def __init__(self, root: AMRNode, invert_edges=False):
        self.root = root
        self.other_roots = []
        self.nodes: List[AMRNode] = []
        self.edges: List[AMREdge] = []
        self._node_map: Dict[str, AMRNode] = {}  # name -> node
        self._inv_edges: List[AMREdge] = []

        self._constant_cnt = 0
        self._collect_nodes(root)
        for node in self.nodes:
            self._collect_edges(node, invert_edges)
        for edge in self._inv_edges:
            edge.var1.edges.append(edge)

        # if discard_inv_edges:
        #    nodes = self.root.descendants
        #    self.nodes = nodes
        #    self.edges = [x for x in self.edges if x.var1 in nodes and x.var2 in nodes]

    def _collect_nodes(self, node: AMRNode):
        if isinstance(node, AMRRef):
            return
        if isinstance(node, AMRConstant):
            node.name = f"constant_{self._constant_cnt}"
            self._constant_cnt += 1
        self.nodes.append(node)
        self._node_map[node.name] = node
        if isinstance(node, AMRVariable):
            for edge in node.edges:
                self._collect_nodes(edge.var2)

    def _collect_edges(self, node: AMRNode, invert_edges=False):
        if not isinstance(node, AMRVariable):
            return
        node: AMRVariable
        fixed_edges = []
        for edge in node.edges:
            if isinstance(edge.var2, AMRRef):  # Resolve references
                # exclude edges to unresolved vars
                # do not include them in fixed_edges
                if edge.var2.name not in self._node_map:
                    continue
                edge.var2 = self._node_map[edge.var2.name]
            if (
                invert_edges
                and edge.relationship.endswith("-of")
                and edge.relationship != "consist-of"
                and len(edge.var2.descendants) > 3
            ):
                # reverse edge
                edge.relationship = edge.relationship[:-3]
                self.other_roots.append(edge.var2)
                edge.var1, edge.var2 = edge.var2, edge.var1
                self._inv_edges.append(edge)
            fixed_edges.append(edge)
            self.edges.append(edge)
        node.edges = fixed_edges

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node.name)
        for edge in self.edges:
            graph.add_edge(edge.var1.name, edge.var2.name)
        return graph

    def extract_subgraphs(self) -> List[AMRVariable]:
        all_subgraphs: List[AMRVariable] = []

        if self.root.concept in ("and", "multi-sentence"):
            queue = []
            for edge in self.root.outbound_edges:
                if edge.relationship.startswith("op") or edge.relationship.startswith(
                    "snt"
                ):
                    queue.append(edge.var2)
        else:
            queue = [self.root]
        queue.extend(self.other_roots)

        # perform a DFS copy.
        # if looking at a verb, stop and start a new graph copy with verb as root
        # if looking at a reference, only copy a simplified version of it
        while queue:
            next_node = queue.pop(0)
            node_copy, new_roots = self.dfs_copy_from_node(next_node, set())
            for new_root in new_roots:
                if new_root not in all_subgraphs:
                    queue.append(new_root)
            all_subgraphs.append(node_copy)
        
        return all_subgraphs

    def dfs_copy_from_node(
        self, root: AMRNode, copied_nodes, stack=None
    ) -> Tuple[AMRNode, List[AMRNode]]:
        # practically, loop must be avoided.
        if stack is None:
            stack = []
        if id(root) in stack:
            # loop detected, create a shallow copy
            if isinstance(root, AMRConstant):
                copied_nodes.add(id(root))
                return root.clone(), []
            assert isinstance(root, AMRVariable)
            return AMRVariable(root.name, root.concept, [], root.metadata), []
        stack.append(id(root))

        # return a copy of the graph rooted at `root`, and new roots to start copying
        new_roots = []
        if isinstance(root, AMRConstant):
            copied_nodes.add(id(root))
            return root.clone(), []
        assert isinstance(root, AMRVariable)

        reentrant = id(self) in copied_nodes

        # make a copy of the node
        copied_nodes.add(id(root))
        copied_root = AMRVariable(root.name, root.concept, [], root.metadata)
        for edge in root.outbound_edges:
            # if reentrant, only keep attributes
            if reentrant and edge.relationship.startswith("ARG"):
                continue

            # copy all edges
            if (
                isinstance(edge.var2, AMRVariable)
                and edge.var2.is_verb
                and (
                    not edge.relationship.endswith("-of")
                    and edge.relationship != "consist-of"
                )
                and len(edge.var2.edges) > 1
            ):
                # don't go deeper, and start a new copy
                if edge.var2 not in new_roots and id(edge.var2) not in stack:
                    new_roots.append(edge.var2)
                # make a shallow copy of the verb
                verb_copy = AMRVariable(
                    edge.var2.name, edge.var2.concept, [], edge.var2.metadata
                )
                copied_root.edges.append(
                    AMREdge(copied_root, verb_copy, edge.relationship)
                )
            else:
                # go deeper
                new_node, new_roots_from_node = self.dfs_copy_from_node(
                    edge.var2, copied_nodes, stack
                )
                copied_root.edges.append(
                    AMREdge(copied_root, new_node, edge.relationship)
                )
                for new_root in new_roots_from_node:
                    if new_root not in new_roots:
                        new_roots.append(new_root)
        stack.pop()
        return copied_root, new_roots


if __name__ == "__main__":
    penman = """
# ::id 0
# ::annotator bart-amr
# ::date 2023-01-30 14:48:34.480011
# ::snt She enjoyed her strenuous years at Westminster , with their comradeship , their common purpose , and their desperate overwork for the women members , who found themselves in exceptional demand .
(z0 / enjoy-01
    :ARG0 (z1 / she)
    :ARG1 (z2 / multiple
              :op1 (z3 / temporal-quantity
                       :quant 1
                       :unit (z4 / year))
              :mod (z5 / strenuous)
              :poss z1
              :location (z6 / government-organization
                            :wiki "Palace_of_Westminster"
                            :name (z7 / name
                                      :op1 "Westminster")))
    :accompanier (z8 / and
                     :op1 (z9 / comradehip
                              :poss (z10 / they))
                     :op2 (z11 / purpose
                               :mod (z12 / common)
                               :poss z10)
                     :op3 (z13 / overwork-01
                               :ARG0 z10
                               :ARG1-of (z14 / desperate-02
                                             :ARG0 (z15 / member
                                                        :mod (z16 / woman)
                                                        :ARG0-of (z17 / find-01
                                                                      :ARG1 (z18 / demand-01
                                                                                 :ARG1 z15
                                                                                 :mod (z19 / exceptional))))))))
    """

    from amr_format import parse_amrbart_output

    sent, amr_node, _ = next(parse_amrbart_output(penman))
    graph = AMRGraph(amr_node)
    """
    dig = graph.to_networkx()
    import matplotlib.pyplot as plt
    nx.draw_networkx(dig)
    plt.show()
    """
    subgraphs = graph.extract_subgraphs()
    for i, subgraph in enumerate(subgraphs):
        print("Subgraph", i)
        print(subgraph.to_penman())
        print()
