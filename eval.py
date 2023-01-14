import argparse

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="trial", help="Dataset type you work on ('trail'-'train'-'test')")
parser.add_argument("--input_path", required=True, type=str, help="predictions path")
args = parser.parse_args()

# file operations
pred = args.input_path
if args.dataset == "trial":
    gold = "/datasets/SemEval/semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt"
elif args.dataset == "train":
    gold = "/datasets/SemEval/semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt"
elif args.dataset == "test":
    raise NotImplementedError
else:
    raise Exception("Use either: (i) trail, (ii) train, (iii) test as argument!")

# read gts and preds
with open(pred, "r") as fp:
    lines = fp.readlines()
    pred_file_names = [line.split()[0].rstrip() for line in lines]
with open(gold, "r") as fg:
    lines = fg.readlines()
    gold_file_names = [line.rstrip() for line in lines]

# calc score
acc = 0
for idx in range(len(pred_file_names)):
    if pred_file_names[idx] == gold_file_names[idx]:
        acc +=1
print(f"Eval Acc: {acc}/{len(pred_file_names)}")
print(f"Eval Acc: {acc/len(pred_file_names)}")