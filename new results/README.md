We add the results of comparison RTA and seBERT, the latter is also pre-trained on SE data, i.e., GitHub/Jira issues. Additionally, we also fine-tune RTA on bug localization.


### Bug priority prediction (%)
| Model|Accuracy|Precision|Recall|F1-Score|
|:----:|:----:|:----:|:----:|:----:|
|seBERT|72.78|70.61|72.78|70.58|
|RTA|73.49|72.32|73.49|72.69|

### Bug severity prediction (%)
| Model|Accuracy|Precision|Recall|F1-Score|
|:----:|:----:|:----:|:----:|:----:|
|seBERT|60.61|61.81|60.61|60.70|
|RTA|64.72|65.11|64.72|64.73|

PS: For precision, recall, and F1-score, we calculated the weighted average of priority P1-P5.

### Duplicate bug report detection
| Model|Accuracy|Precision|Recall|F1-Score|
|:----:|:----:|:----:|:----:|:----:|
|S-BERT|97.41|98.25|98.19|98.21|
|seBERT|97.25|98.24|97.97|98.10|
|RTA|97.88|98.54|98.54|98.54|

### bug report summarization
| Model|ROUGE-1|ROUGE-2|ROUGE-L|c.BLEU|
|:----:|:----:|:----:|:----:|:----:|
|seBERT|29.47|16.05|27.42|4.19|
|RTA|39.19|20.57|35.97|10.13|

### bug localization
| Model|Acc@1|Acc@5|Acc@10|MAP|MRR|
|:----:|:----:|:----:|:----:|:----:|:----:|
|DreamLoc|0.40|0.72|0.81|0.48|0.54|
|FLIM|0.50|0.73|0.81|0.53|0.60|
|BugLocator|0.20|0.48|0.57|0.22|0.32|
|YBL|0.36|0.53|0.64|0.37|0.44|
|AdaptiveBL|0.41|0.65|0.68|0.43|0.51|
|RTA|0.56|0.77|0.84|0.56|0.63|
