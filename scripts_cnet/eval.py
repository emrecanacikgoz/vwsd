import sys

pred = sys.argv[1]
gold = "/datasets/SemEval/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt"

with open(pred, "r") as fp:
    lines = fp.readlines()
    pred_file_names = [line.split()[0].rstrip() for line in lines]

with open(gold, "r") as fg:
    lines = fg.readlines()
    gold_file_names = [line.rstrip() for line in lines]

acc = 0
for idx in range(len(pred_file_names)):
    if pred_file_names[idx] == gold_file_names[idx]:
        acc +=1

print(f"Eval Acc: {acc}/{len(pred_file_names)}")
