### RTA(RepresentThemAll): Learning to Represent Bug Reports

Hello! Thanks for your attention on our work. Bug report is the specific report that contains the information about what is wrong and where developers should fix the given bug, which can help perform software maintenance. As for RTA, it is a langauge model pretrained on the bug report corpus and can be fine-tuned on different bug report related tasks, e.g., bug priority and severity prediction, bug report summarization, and duoplicate bug report detection, all of which can improve the efficiency of software maintenance.

- 🔭 We are currently working on fine-tuning RTA on bug localization. Some developers think that it is helpful for bug fixing.
- 🌱 We are planing to apply RTA to more downstream tasks, like error estimation and bug fixer recommendation.
- 🌱 We are planing to collect a larger bug report corpus for RTA's pre-training.

#### Dependency
Successfully tested in CentOS 7 with four RTX 3090 GPUs.
- Python == 3.7
- PyTorch == 1.6.0
- transformers == 4.17.0
- datasets == 1.17.0
- numpy == 1.18.5

#### Project Structure
- `brsumm`: It contains source code for bug report summarization.
- `bugpred`: It contains source code and some experimental results for bug priority and severity prediction.
- `dupbrdet`: It contains source code and some experimental results for duplicate bug report detection.
- `new results`: It contains some new experimental results like fine-tuning RTA on bug localization and the comparison between RTA and seBERT.
- `baseline.zip`: It contains no-official implementations for some baselines.
#### Dataset
All data we used in our experiments you can find at [dataset](https://drive.google.com/drive/folders/1gPnZbgOO4XiBBsyF27jS--XwhHaInxlQ?usp=sharing). You can download pre-trained RTA at [RTA](https://huggingface.co/Colorful/RTA).

#### Quick Tour
We use huggingface/transformers framework to train the model. You can use our model like the pre-trained Roberta base. Now, We give an example on how to load the model and obtian embedding from RTA.

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> import torch
>>> tokenizer = AutoTokenizer.from_pretrained("Colorful/RTA")
>>> model = AutoModel.from_pretrained("Colorful/RTA")
>>> br = "\"Select in Projects\" should select classes from jars 1) Open \"Go to Type\" dialog 2) Select java.lang.Object class 3) Open the context menu of the editor where the source file of the Object class is shown after the Go To Type dialog has been closed. Notice that the editor is in read-only mode. 4) In the context menu select \"Select in Projects\" menu item. Notice that you can achieve the same by using Navigate main menu when the editor is active.  Actual result: Project panel is opened, the panel keeps the previous selection. Which is wrong.  Expected result: Project panel is opened, the panel expands the project tree so that   <project_name>/Libraries/<JDK>/rt.jar/java/lang/Object.class  gets selected and revealed."
>>> br_tokens = tokenizer.tokenize(br)
['"', 'Select', 'Ġin', 'ĠProjects', '"', 'Ġshould', 'Ġselect', 'Ġclasses', 'Ġfrom', 'Ġjars', 'Ġ1', ')', 'ĠOpen', 'Ġ"', 'Go', 'Ġto', 'ĠType', '"', 'Ġdialog', 'Ġ2', ')', 'ĠSelect', 'Ġjava', '.', 'lang', '.', 'Object', 'Ġclass', 'Ġ3', ')', 'ĠOpen', 'Ġthe', 'Ġcontext', 'Ġmenu', 'Ġof', 'Ġthe', 'Ġeditor', 'Ġwhere', 'Ġthe', 'Ġsource', 'Ġfile', 'Ġof', 'Ġthe', 'ĠObject', 'Ġclass', 'Ġis', 'Ġshown', 'Ġafter', 'Ġthe', 'ĠGo', 'ĠTo', 'ĠType', 'Ġdialog', 'Ġhas', 'Ġbeen', 'Ġclosed', '.', 'ĠNotice', 'Ġthat', 'Ġthe', 'Ġeditor', 'Ġis', 'Ġin', 'Ġread', '-', 'only', 'Ġmode', '.', 'Ġ4', ')', 'ĠIn', 'Ġthe', 'Ġcontext', 'Ġmenu', 'Ġselect', 'Ġ"', 'Select', 'Ġin', 'ĠProjects', '"', 'Ġmenu', 'Ġitem', '.', 'ĠNotice', 'Ġthat', 'Ġyou', 'Ġcan', 'Ġachieve', 'Ġthe', 'Ġsame', 'Ġby', 'Ġusing', 'ĠNav', 'igate', 'Ġmain', 'Ġmenu', 'Ġwhen', 'Ġthe', 'Ġeditor', 'Ġis', 'Ġactive', '.', 'Ġ', 'ĠActual', 'Ġresult', ':', 'ĠProject', 'Ġpanel', 'Ġis', 'Ġopened', ',', 'Ġthe', 'Ġpanel', 'Ġkeeps', 'Ġthe', 'Ġprevious', 'Ġselection', '.', 'ĠWhich', 'Ġis', 'Ġwrong', '.', 'Ġ', 'ĠEx', 'pected', 'Ġresult', ':', 'ĠProject', 'Ġpanel', 'Ġis', 'Ġopened', ',', 'Ġthe', 'Ġpanel', 'Ġexpands', 'Ġthe', 'Ġproject', 'Ġtree', 'Ġso', 'Ġthat', 'Ġ', 'Ġ', 'Ġ<', 'project', '_', 'name', '>', '/', 'L', 'ibraries', '/', '<', 'JD', 'K', '>', '/', 'rt', '.', 'jar', '/', 'java', '/', 'lang', '/', 'Object', '.', 'class', 'Ġ', 'Ġgets', 'Ġselected', 'Ġand', 'Ġrevealed', '.']
>>> tokens=[tokenizer.cls_token]+br_tokens+[tokenizer.eos_token]
>>> tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
[0, 113, 45356, 11, 23965, 113, 197, 5163, 4050, 31, 35094, 112, 43, 2117, 22, 11478, 7, 7773, 113, 25730, 132, 43, 10908, 46900, 4, 32373, 4, 46674, 1380, 155, 43, 2117, 5, 5377, 5765, 9, 5, 4474, 147, 5, 1300, 2870, 9, 5, 35671, 1380, 16, 2343, 71, 5, 2381, 598, 7773, 25730, 34, 57, 1367, 4, 22873, 14, 5, 4474, 16, 11, 1166, 12, 8338, 5745, 4, 204, 43, 96, 5, 5377, 5765, 5163, 22, 45356, 11, 23965, 113, 5765, 6880, 4, 22873, 14, 47, 64, 3042, 5, 276, 30, 634, 8236, 24343, 1049, 5765, 77, 5, 4474, 16, 2171, 4, 1437, 30144, 898, 35, 3728, 2798, 16, 1357, 6, 5, 2798, 4719, 5, 986, 4230, 4, 6834, 16, 1593, 4, 1437, 3015, 23088, 898, 35, 3728, 2798, 16, 1357, 6, 5, 2798, 20539, 5, 695, 3907, 98, 14, 1437, 1437, 28696, 28258, 1215, 13650, 15698, 73, 574, 47437, 73, 41552, 26697, 530, 15698, 73, 9713, 4, 11978, 73, 43830, 73, 32373, 73, 46674, 4, 4684, 1437, 1516, 3919, 8, 1487, 4, 2]
>>> context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
torch.Size([1, 175, 768])
tensor([[[-0.0376,  0.1328, -0.0124,  ..., -0.1959,  0.0159,  0.0594],
         [ 0.0514,  0.4983, -0.1029,  ..., -0.2608,  0.2634,  0.4894],
         [ 0.1240, -0.0383,  0.2950,  ...,  0.4582,  0.4101, -0.0207],
         ...,
         [ 0.0052, -0.2010,  0.0150,  ..., -0.5293,  0.2408, -0.0815],
         [-0.0500,  0.1313, -0.0308,  ..., -0.2453,  0.0029,  0.0397],
         [-0.0500,  0.1313, -0.0308,  ..., -0.2453,  0.0029,  0.0397]]],
       grad_fn=<NativeLayerNormBackward>)
 ```

#### Probing

We give an example on using RTA for mask prediction task (pre-training).

```python
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

model = RobertaForMaskedLM.from_pretrained("Colorful/RTA")
tokenizer = RobertaTokenizer.from_pretrained("Colorful/RTA")

BR = "[6 Regression] Copying char arrays <mask> constexpr evaluation does not work reliably"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(BR)
print(outputs)
```

Results
```python
'during', 'in', 'for', 'with', 'to'
Gold: 'during'
```

The detailed outputs are as follows:
```python
{'score': 0.26558613777160645, 'token': 148, 'token_str': ' during', 'sequence': '[6 Regression] Copying char arrays during constexpr evaluation does not work reliably'}
{'score': 0.1885920614004135, 'token': 11, 'token_str': ' in', 'sequence': '[6 Regression] Copying char arrays in constexpr evaluation does not work reliably'}
{'score': 0.11839114874601364, 'token': 13, 'token_str': ' for', 'sequence': '[6 Regression] Copying char arrays for constexpr evaluation does not work reliably'}
{'score': 0.09877894818782806, 'token': 19, 'token_str': ' with', 'sequence': '[6 Regression] Copying char arrays with constexpr evaluation does not work reliably'}
{'score': 0.04735679551959038, 'token': 7, 'token_str': ' to', 'sequence': '[6 Regression] Copying char arrays to constexpr evaluation does not work reliably'}
```

#### Downstream Tasks

For downstream tasks, please refer to the [brsumm](https://github.com/ICSE-2023/RepresentThemALL/edit/main/brsumm), [bugpred](https://github.com/ICSE-2023/RepresentThemALL/edit/main/bugpred), and [dupbrdet](https://github.com/ICSE-2023/RepresentThemALL/edit/main/dupbrdet) folders.

<!--**ICSE-2023/RepresentThemALL** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
