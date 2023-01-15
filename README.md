# vwsd
SemEval2023-Visual Word Sense Disambiguation Task: https://raganato.github.io/vwsd/

# Get Results
For given task; (i) trial, (ii) train, (iii) test and prediction file path, gives the accuracy.
```
python eval.py --dataset <task> --input_path </path/to/file>
```
Example:
```
python eval.py --dataset trial --input_path /kuacc/users/eacikgoz17/semeval/results_trial_right.txt
```

## Current Results
| Model         | Task            | Context       | Score         | Score         |
| :------------ |:---------------:|:-------------:|:-------------:|:-------------:|
| Clip          | Trial           |         Full  |      9/16     |     56.25%    |
| Clip          | Trial           |     Left-only |      7/16     |     43.75%    |
| Clip          | Trial           |    Right-only |      8/16     |     50.00%    |
| Clip          | Train           |         Full  |  10608/12869  |     82.43%    |
| Clip          | Train           |     Left-only |   9549/12869  |     74.20%    |
| Clip          | Train           |    Right-only |  10539/12869  |     81.89%    |
| Blip          | Trial           |         Full  |      9/16     |     56.25%    |
| Blip          | Trial           |     Left-only |      7/16     |     43.75%    |
| Blip          | Trial           |    Right-only |      6/16     |     37.50%    |
| Blip          | Train           |         Full  |   7793/12869  |     60.55%    |
| Blip          | Train           |     Left-only |   5392/12869  |     41.89%    |
| Blip          | Train           |    Right-only |   6342/12869  |     49.28%    |
| Clip-Google   | Trial           |         Full  |      14/16    |     87.50%    |
| Clip-CNet     | Trial           |         Full  |      10/16    |     62.50%    |
