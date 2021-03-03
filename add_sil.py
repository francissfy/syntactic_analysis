import argparse


def add_dur_sil(dur_line: str, feat_line: str, feats_dim: int) -> str:
    durs = dur_line.split(" ")
    feats = feat_line.split(" ")
    res = []
    idx = 0
    for dur in durs:
        w, d = dur.split("|", maxsplit=1)
        if w == "<SIL>":
            sil_feat = [str(d)] + ["0"]*(feats_dim-1)
            res.append("|".join(sil_feat))
        else:
            res.append(f"{d}|{feats[idx]}")
            idx += 1
    return " ".join(res)


# extarct duration from word_dur file and add to the head of utt2num_phones, as well as add <SIL> to feat
def main(args: argparse.Namespace):
    with open(args.utt2num_phones, "r") as dur_f, open(args.feat, "r") as feat_f, open(args.feat_out, "w") as out_f:
        while True:
            dur_line = dur_f.readline().strip()
            if dur_line == "":
                break
            dur_id, dur_line = dur_line.split(" ", maxsplit=1)
            feat_line = feat_f.readline().strip()
            assert feat_line != ""
            feat_id, feat_line = feat_line.split(" ", maxsplit=1)
            assert feat_id == dur_id, "please make utt2num_phones and feat file in the same order"
            # add 1 for durations
            feat_dim = len(feat_line.split(" ")[0].split("|")) + 1
            new_feat = add_dur_sil(dur_line, feat_line, feat_dim)
            out_f.write(f"{feat_id} {new_feat}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt2num_phones", type=str)
    parser.add_argument("--feat", type=str)
    parser.add_argument("--feat_out", type=str)
    args = parser.parse_args()
    main(args)
