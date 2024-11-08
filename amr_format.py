"""
Parsing PENMAN format AMR graphs into a tree structure.
Convert the tree structure into SPRING format or PENMAN format.
"""

from typing import *
from dataclasses import dataclass, field
from collections import namedtuple
import re

from lark import Lark, Transformer

PENMANItem = namedtuple("PENMANItem", ["penman", "sent", "metadata"])

@dataclass
class AMRVariable:
    name: str
    concept: str
    edges: List["AMREdge"] = field(default_factory=list)
    metadata: List[str] = field(default_factory=list)

    def clone(self):
        new_var = AMRVariable(self.name, self.concept, [], self.metadata.copy())
        for edge in self.edges:
            new_edge = AMREdge(new_var, edge.var2.clone(), edge.relationship)
            new_var.edges.append(new_edge)
        return new_var

    def asdict(self):
        return {"name": self.name, "concept": self.concept}
    
    def __str__(self):
        return f"[{self.name} / {self.concept}]"

    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name

    def __getitem__(self, rel) -> "AMRNode":
        for edge in self.edges:
            if edge.var1 == self and edge.relationship == rel:
                return edge.var2
        raise KeyError

    def pretty_tree(self, indent=0) -> str:
        r = ""
        r += " " * indent + str(self) + "\n"
        for edge in self.outbound_edges:
            r += " " * indent + "--> " + edge.relationship + "\n"
            r += edge.var2.pretty_tree(indent + 8)
        return r

    def get(self, rel, fallback=None) -> Optional["AMRNode"]:
        try:
            return self[rel]
        except KeyError:
            return fallback

    @property
    def outbound_edges(self) -> List["AMREdge"]:
        edges = [edge for edge in self.edges if edge.var1 == self]
        return sorted(edges, key=lambda edge: edge.relationship)
    
    @property
    def is_verb(self):
        name_is_verb = re.match(r".*-\d+", self.concept) is not None
        return name_is_verb and any(x.relationship.startswith("ARG") for x in self.outbound_edges)
    
    @property
    def descendants(self) -> List["AMRNode"]:
        return self._get_descendants()
    
    def _get_descendants(self, stack=None):
        if stack is None:
            stack = set()
        if id(self) in stack:
            return []
        stack.add(id(self))
        nodes = [self]
        for edge in self.outbound_edges:
            if edge.var2 == self:
                continue
            var2_descendants = edge.var2._get_descendants(stack)
            if self in var2_descendants:
                var2_descendants.remove(self)
            nodes.extend(var2_descendants)
        stack.remove(id(self))
        return nodes

    def add_edge(self, var2, relationship):
        self.edges.append(AMREdge(self, var2, relationship))

    def to_spring(self, delim='\u0120', lit_begin="<lit>", lit_end="</lit>") -> str:
        s = f"{delim}( {delim}<pointer:{self.name[1:]}> {delim}{self.concept} "
        for edge in self.outbound_edges:
            s += f"{delim}:{edge.relationship} {edge.var2.to_spring(delim, lit_begin, lit_end)} "
        s += f"{delim})"
        return s
    
    def to_penman(self, indent=0):
        s = f"({self.name} / {self.concept}"
        indent += len(self.name) + 2 # align with /
        for edge in self.outbound_edges:
            s += f"\n{' ' * indent}:{edge.relationship} {edge.var2.to_penman(indent + len(edge.relationship) + 2)}"
        return s + ")"
    
    def to_json(self):
        return {
            "name": self.name,
            "type": "variable",
            "concept": self.concept,
            "edges": {
                edge.relationship: edge.var2.to_json()
                for edge in self.outbound_edges
            }
        }

    def remove_wiki_links(self):
        self.edges = [x for x in self.edges if x.relationship != "wiki"]
        for edge in self.outbound_edges:
            if isinstance(edge.var2, AMRVariable):
                edge.var2.remove_wiki_links()
    
    def mask_entities(self, entity_id=1) -> List[Tuple[str, str]]:
        # returns a list of (mask_name, entity_name) pairs
        results = []
        # first, remove wiki links
        self.edges = [x for x in self.edges if x.relationship != "wiki"]
        if self.concept == "name":
            # concatenate all name parts
            name_parts = []
            op_i = 1
            while True:
                try:
                    name_parts.append(self[f"op{op_i}"].value)
                except KeyError:
                    break
                except AttributeError:
                    pass
                op_i += 1
            entity_name = " ".join(map(str, name_parts))
            if entity_name: #and not entity_name.startswith("ENTITY"): # ignore empty names
                mask_name = f"ENTITY{entity_id}"
                results.append((mask_name, entity_name))

                # replace the name node with a constant
                new_node = AMRConstant(mask_name, name=None, literal=True)
                self.edges = [x for x in self.edges if not x.relationship.startswith("op")]
                self.edges.append(AMREdge(self, new_node, "op1"))
        for edge in self.outbound_edges:
            if isinstance(edge.var2, AMRVariable):
                results.extend(edge.var2.mask_entities(entity_id + len(results)))
        return results
    

    def extract_entities(self) -> List[str]:
        results = []
        if self.is_verb:
            for edge in self.outbound_edges:
                if isinstance(edge.var2, AMRVariable):
                    results.extend(edge.var2.extract_entities())
        else:
            concept = self.concept
            if concept in ('and', 'multi-sentence'):
                return []
            if '-' in concept:
                return []
            names = []
            for edge in self.outbound_edges:
                if edge.relationship == "name":
                    names = edge.var2.mask_entities()
                    break
            names = [x[1] for x in names]
            if names:
                concept = f"{concept} ({names[0]})"
            results.append(concept)
        return results
            

@dataclass
class AMRConstant:
    value: Union[int, str, float]
    name: str = None
    literal: bool = False

    @property
    def is_verb(self):
        return False

    def clone(self):
        return AMRConstant(self.value, self.name, self.literal)

    def __str__(self):
        return str(self.value)

    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self):
        return hash(("constant", self.value, id(self)))

    def pretty_tree(self, indent=0) -> str:
        return " " * indent + str(self) + "\n"

    def to_spring(self, delim="\u0120", lit_begin="<lit>\u0120", lit_end="</lit>\u0120") -> str:
        if self.literal:
            return f"{delim}{lit_begin}{self.value}{lit_end}{delim}"
        else:
            return f"{delim}{self.value} "
    
    def to_json(self):
        return {
            "name": self.name,
            "type": "constant",
            "value": self.value,
        }
    
    def to_penman(self, indent=0):
        if self.literal:
            return f'"{self.value}"'
        else:
            return str(self.value)
    
    @property
    def descendants(self) -> List["AMRNode"]:
        return [self]

    def _get_descendants(self, stack=None) -> List["AMRNode"]:
        return [self]


@dataclass
class AMRRef:
    name: str

    def to_spring(self, delim="\u0120", lit_begin=None, lit_end=None) -> str:
        return f"{delim}<pointer:{self.name[1:]}>"
    
    def to_penman(self, indent=0):
        return self.name

    def to_json(self):
        return {
            "name": self.name,
            "type": "reference",
        }
    def _get_descendants(self, stack=None) -> List["AMRNode"]:
        return [self]


AMRNode = Union[AMRVariable, AMRConstant, AMRRef]


@dataclass
class AMREdge:
    var1: AMRNode
    var2: AMRNode
    relationship: str

    def asdict(self):
        return {
            "var1": self.var1.name,
            "var2": self.var2.name,
            "relationship": self.relationship,
        }

    def __str__(self):
        return f"{self.var1} - {self.relationship} -> {self.var2}"

    def __repr__(self) -> str:
        return str(self)


class ParseToAMRTreeTransformer(Transformer):
    def attr(self, children):
        relation = children[0].value
        var2 = children[1]
        edge = AMREdge(None, var2, relation)
        return edge

    def arg(self, children):
        return self.attr(children)

    def literal(self, value):
        assert len(value) > 0
        value = value[0].value
        is_literal = False
        if value[0] == '"':
            value = value[1:-1]
            is_literal = True
        elif value in ("+", "-", "imperative", "expressive"):
            pass
        else:
            for transforms in (int, float):
                try:
                    value = transforms(value)
                    break
                except ValueError:
                    pass
        return AMRConstant(value, None, is_literal)

    def var_ref(self, children):
        return AMRRef(children[0].value)

    def var_def(self, children):
        name = children[0].value
        concept = children[1].value
        edges = children[2:]
        node = AMRVariable(name, concept, edges)
        for e in node.edges:
            e.var1 = node
        return node


with open("penman.ebnf") as f:
    _grammar = f.read()
_parser = Lark(_grammar, start="amr_tree")
_transformer = ParseToAMRTreeTransformer()


def read_penman(lines) -> Iterable[PENMANItem]:
    sent = None
    amr_output_lines = []
    metadata_lines = []
    for line in lines:
        if line.startswith("# ::snt "):
            metadata_lines.append(line)
            sent = line[8:].strip()
        elif line.startswith("#"):
            metadata_lines.append(line)
            continue
        elif not line.strip():
            amr_output = "\n".join(amr_output_lines)
            if not amr_output.strip():
                continue
            yield PENMANItem(amr_output, sent, metadata_lines)
            metadata_lines = []
            amr_output_lines = []
        else:
            amr_output_lines.append(line)
    if amr_output_lines:
        amr_output = "\n".join(amr_output_lines)
        yield PENMANItem(amr_output, sent, metadata_lines)


def parse_penman(item: PENMANItem, print_unparsed_item=False) -> Iterable[AMRNode]:
    try:
        tree = _parser.parse(item.penman)
    except Exception:
        print("can not be parsed")
        if print_unparsed_item:
            print(item.penman)
        yield None
        return
    graph = _transformer.transform(tree)
    graph.metadata = item.metadata
    yield graph


def parse_amrbart_output_file(filename, **kwargs) -> Iterable[AMRNode]:
    with open(filename) as f:
        for item in read_penman(f):
            for out in parse_penman(item):
                yield item.sent, out


def parse_amrbart_output(content) -> Iterable[Tuple[str, AMRNode]]:
    if isinstance(content, str):
        lines = content.split("\n") + [""]
    else:
        lines = content
    for item in read_penman(lines):
        for out in parse_penman(item):
            yield item.sent, out, item.penman

if __name__ == '__main__':
    """
    sents = list(parse_amrbart_output_file("examples/penman.txt"))
    with open("examples/spring.txt", "w") as f:
        for sent, graph in sents:
            f.write(graph.to_spring() + "\n")
    with open("examples/to_penman.txt", "w") as f:
        for sent, graph in sents:
            f.write(graph.to_penman() + "\n\n")
    with open("examples/mask_entity_penman.txt", "w") as f:
        for sent, graph in sents:
            pairs = graph.mask_entities()
            for mask_name, entity_name in pairs:
                f.write(f"# ::{mask_name} {entity_name}\n")
            f.write(graph.to_penman() + "\n\n")

    """

    import pandas as pd
    df = pd.read_json("examples/websplit.json", orient="records", lines=True)
    df['entities'] = None
    for idx, row in df.iterrows():
        amr = row['amrv1']
        _, node, _ = next(parse_amrbart_output(amr))
        if node is None:
            continue
        entities = node.extract_entities()
        df.at[idx, 'entities'] = entities
    df.to_json("examples/websplit_with_entities.json", orient="records", lines=True)