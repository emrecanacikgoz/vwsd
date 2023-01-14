import torch, argparse
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="trial", help="Dataset type you work on ('trail'-'train'-'test')")
parser.add_argument("--context", default="full", help="giving (andromeda tree): (i) left (andromeda), (ii) right (tree), or (iii) full context (andromeda tree)")
args = parser.parse_args()

# files
if args.dataset == "trial":
    print(f"\nUsing trial data\n")
    train_path   = "/datasets/SemEval/semeval-2023-task-1-V-WSD-train-v1/trial_v1"
    train_file   = "trial.data.v1.txt"
    train_gold   = "trial.gold.v1.txt"
    train_images = "trial_images_v1"
elif args.dataset == "train":
    print("\nUsing train data\n")
    train_path   = "/datasets/SemEval/semeval-2023-task-1-V-WSD-train-v1/train_v1"
    train_file   = "train.data.v1.txt"
    train_gold   = "train.gold.v1.txt"
    train_images = "train_images_v1"
elif args.dataset == "test":
    raise NotImplementedError
else:
    raise Exception("Use either: (i) trail, (ii) train, (iii) test as argument!")

# text
data         = pd.read_csv(train_path + "/" + train_file, sep='\t', header=None)
data.columns = ['word', 'context'] + [f'image{i}' for i in range(10)]
data2        = pd.read_csv(train_path + "/" + train_gold, sep='\t', header=None)
data['image_gold'] = data2

# images
img_cols   = [f'image{i}' for i in range(10)]
all_images = np.unique(data[img_cols].values.flatten())
#images     = {k: Image.open(train_path + "/" + train_images + "/" + k).convert("RGB") for k in tqdm(all_images)} 

# model
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

# zero-shot
results = []
with torch.no_grad():
    for i, (_, row) in tqdm(enumerate(data.iterrows()), total=data.shape[0]):
        if args.context == "right":
            caption = row['context'].split(" ")[0]
        elif args.context == "left":
            caption = row['context'].split(" ")[-1]
        else: 
           caption = row['context'] 
        text = text_processors["eval"](caption)
        predictions = {}
        for idx, img_path in enumerate(row.loc[img_cols]):
            raw_image = Image.open(train_path + "/" + train_images + "/" + img_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            prob = model({"image": image, "text_input": text}, match_head='itc')
            predictions[img_path] = prob.item()
        results.append(predictions)
preds = [max(result, key=result.get) for result in results]

# write predictions
with open("results_" + args.dataset + "_" + args.context + ".txt", "w") as fr:
    for pred in preds:
        fr.write(pred+"\n")

