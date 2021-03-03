import argparse
from typing import List
from copy import copy

tags = set()
tag_list = []
tag2idx = {}


# the size of tags is 56
def collect_tags(files: List[str]):
    global tags
    global tag_list
    global tag2idx
    for file in files:
        with open(file, "r") as f:
            while True:
                line = f.readline().strip()
                if line == "":
                    break
                line = line.split(" ", maxsplit=1)[1]
                feats = line.split(",")
                for feat in feats:
                    sub = feat.split("|")
                    signs = sub[1:5]
                    for s in signs:
                        tags.add(s)
    tag_list = list(tags)
    for idx, tag in enumerate(tag_list):
        tag2idx[tag] = idx
    print(f"total tags: {len(tags)}")


def one_hot_str(tag: str) -> str:
    global tag_list
    global tag2idx
    idx = tag2idx[tag]
    code = [0 for _ in tag_list]
    code[idx] = 1
    return "|".join([str(t) for t in code])


def dense_code_str(tag: str) -> str:
    global tag2idx
    idx = tag2idx[tag]
    return str(idx)


def conv2code(raw_file: str, code_file: str, code_type: str = "dense"):
    print(f"use coding type: {code_type}")
    with open(raw_file, "r") as rawf, open(code_file, "w") as codef:
        while True:
            line = rawf.readline().strip()
            if line == "":
                break
            line = line.split(" ", maxsplit=1)
            lid = line[0]
            feat = line[1].split(",")
            code_feat = []
            for w in feat:
                tmp = w.split("|")
                # NOTE we drop the word in code feature
                if code_type == "dense":
                    tmp[0:5] = [dense_code_str(tag) for tag in tmp[1:5]]
                elif code_type == "one_hot":
                    tmp[0:5] = [one_hot_str(tag) for tag in tmp[1:5]]
                else:
                    raise NotImplementedError("only one_hot or dense coding is implemented")
                code_feat.append("|".join(tmp))
            code_feat = " ".join(code_feat)
            codef.write(f"{lid} {code_feat}\n")


def main(args: argparse.Namespace):
    raw_feats = args.raw_feats[0]
    code_feats = args.code_feats[0]
    assert len(raw_feats) == len(code_feats)
    collect_tags(raw_feats)
    for raw_f, code_f in zip(raw_feats, code_feats):
        conv2code(raw_f, code_f, code_type=args.code_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_feats", type=str, action="append", nargs="+")
    parser.add_argument("--code_feats", type=str, action="append", nargs="+")
    parser.add_argument("--code_type", type=str, choices=["dense", "one_hot"], default="dense")
    args = parser.parse_args()
    main(args)
