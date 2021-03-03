import time
import logging
import argparse
from treelib import Node, Tree
from typing import List
from stanza.server import CoreNLPClient

logging.getLogger().setLevel(logging.INFO)

leave_count = 0


def tree_builder(tree: Tree, subsyn, pnode: Node = None):
    global leave_count
    if pnode is None:
        node = tree.create_node(subsyn.value, None, data=[])
    else:
        node = Node(subsyn.value, None, data=[])
        tree.add_node(node, pnode)
    if len(subsyn.child) == 0:
        # leave node, mark idx in tokens
        node.data = [leave_count]
        leave_count += 1
        return
    else:
        for c in subsyn.child:
            tree_builder(tree, c, node)
    return


def update_node(tree: Tree, node: Node):
    # gather leaves from bottom to top
    if node.is_leaf():
        return
    for child in tree.children(node.identifier):
        update_node(tree, child)
        node.data += child.data


def H(tree: Tree, node: Node) -> int:
    # height of LCA
    level_leaves = [tree.level(n.identifier) for n in tree.leaves()]
    tree_level = max(level_leaves)
    node_level = tree.level(node.identifier)
    return tree_level - node_level


class SynTree:
    def __init__(self, text, tokens, constituency_parse):
        self.text = text
        self.tokens = tokens
        # building tree
        self.tree = Tree()
        s_time = time.time()
        # reset leave count before build
        global leave_count
        leave_count = 0
        tree_builder(self.tree, constituency_parse, None)
        update_node(self.tree, self.tree.get_node(self.tree.root))
        # idx of token in tokens -> node in tree
        self.idx2node: [int, Node] = {}
        for n in self.tree.leaves():
            assert len(n.data) == 1
            self.idx2node[n.data[0]] = n
        # nid -> token idx
        self.nid2idx: [str, int] = {}
        for idx, n in self.idx2node.items():
            self.nid2idx[n.identifier] = idx
        # order check
        for idx, k in enumerate(self.tokens):
            n = self.idx2node[idx]
            assert k == n.tag
        # log building time
        e_time = time.time()
        dur = (e_time - s_time) * 1000
        logging.info(f"syntax tree build duration: {dur} ms")

    def print(self):
        self.tree.show()

    def is_subsequent(self, widx: int, widxs: List[int]):
        return widx <= min(widxs)

    def all_preceding(self, widx: int, widxs: List[int]):
        return widx >= max(widxs)

    def hbcw(self, widx: int) -> str:
        node = self.pos(widx)
        nids_to_root = list(self.tree.rsearch(node.identifier))
        nodes_to_root = [self.tree.get_node(t) for t in nids_to_root]
        for i, n in enumerate(nodes_to_root):
            if not self.is_subsequent(widx, n.data):
                return nodes_to_root[i - 1].tag
        return nodes_to_root[-1].tag

    def hecw(self, widx: int) -> str:
        node = self.pos(widx)
        nids_to_root = list(self.tree.rsearch(node.identifier))
        nodes_to_root = [self.tree.get_node(t) for t in nids_to_root]
        for i, n in enumerate(nodes_to_root):
            if not self.all_preceding(widx, n.data):
                return nodes_to_root[i - 1].tag
        return nodes_to_root[-1].tag

    def preced_token(self, widx: int) -> int:
        # FIXME if widx==0, return 1; if only one elem in token, return 0
        if len(self.tokens) == 1:
            return 0
        elif widx == 0:
            return 1
        else:
            return widx - 1

    def hepw(self, widx: int) -> str:
        pwidx = self.preced_token(widx)
        return self.hecw(pwidx)

    def lca(self, widx: int) -> Node:
        w_node = self.pos(widx)
        pwidx = self.preced_token(widx)
        nids_to_root = self.tree.rsearch(w_node.identifier)
        for nid in nids_to_root:
            n = self.tree.get_node(nid)
            if pwidx in n.data:
                return n
        return self.tree.get_node(self.tree.root)

    def lca_str(self, widx: int) -> str:
        return self.lca(widx).tag

    def pos(self, widx: int) -> Node:
        node = self.idx2node[widx]
        return self.tree.parent(node.identifier)

    def pos_str(self, widx: int) -> str:
        return self.pos(widx).tag

    def hl(self, widx: int) -> int:
        lca_node = self.lca(widx)
        # NOTE minus the leave level
        return H(self.tree, lca_node) - 1

    def dcl(self, widx: int) -> int:
        lca_node = self.lca(widx)
        c_node = self.pos(widx)
        return H(self.tree, lca_node) - H(self.tree, c_node)

    def dpl(self, widx: int) -> int:
        lca_node = self.lca(widx)
        pwidx = self.preced_token(widx)
        p_node = self.pos(pwidx)
        return H(self.tree, lca_node) - H(self.tree, p_node)

    def dcp(self, widx: int) -> int:
        return self.dcl(widx) + self.dpl(widx)

    def compose(self) -> str:
        s_time = time.time()
        res = []
        for idx, w in enumerate(self.tokens):
            pos = self.pos_str(idx)
            hbcw = self.hbcw(idx)
            hepw = self.hepw(idx)
            lca = self.lca_str(idx)
            hl = self.hl(idx)
            dcl = self.dcl(idx)
            dpl = self.dpl(idx)
            dcp = self.dcp(idx)
            feat = f"{w}|{pos}|{hbcw}|{hepw}|{lca}|{hl}|{dcl}|{dpl}|{dcp}"
            res.append(feat)
        e_time = time.time()
        dur = (e_time - s_time) * 1000
        logging.info(f"syntax feature compose duration: {dur} ms")
        return ",".join(res)


class StanzaNLPAnal:
    def __init__(self):
        self.client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'], timeout=30000,
            memory="10G")

    def constituency_parse(self, text: str):
        s_time = time.time()
        ann = self.client.annotate(text)
        sentence = ann.sentence[0]
        constituency_parse = sentence.parseTree
        tokens = [t.value for t in sentence.token]
        e_time = time.time()
        dur = (e_time - s_time) * 1000
        logging.info(f"constituency_parsing, sentence len: {len(text)} time: {dur} ms")
        return tokens, constituency_parse

    def extract_syn_feat(self, text: str) -> str:
        tokens, parse_res = self.constituency_parse(text)
        syntax_tree = SynTree(text, tokens, parse_res)
        feat = syntax_tree.compose()
        return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="syntax_anal.log")
    parser.add_argument("--text", type=str)
    parser.add_argument("--synout", type=str)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log)

    with open(args.text, "r") as rf, open(args.synout, "w") as wf:
        client = StanzaNLPAnal()
        while True:
            line = rf.readline().strip()
            if line == "":
                break
            line = line.split(" ", maxsplit=1)
            lid = line[0]
            text = line[1]
            text = "".join([t for t in text if t.isalpha() or t == " "])
            feat = client.extract_syn_feat(text)
            wf.write(f"{lid} {feat}\n")
