### Duplicate bug report detection

We provide the source code for fine-tuning RTA on duplicate bug report detection task. In this task, we follow the prior work and regard the duplicate bug report task as a binary classification task that makes the model predict whether a pair of bug reports is duplicate. To help you quickly run our code, you can just use the following command:
```bash
bash run_dup_br_detection.sh
```

Note that you should replace the input of the following three parameter with the path that you store the dataset:
- --train_file
- --validation_file
- --test_file

### Extra Results
We also run RTA on other projects like Eclipse. The result you can see in the following Table:

| Model| Accuracy|Precision|Recall|F1-Score|
|:----:|:----:|:----:|:----:|:----:|
|Siamese|0.8625|0.8367|0.7535|0.7929|
|DWEN|0.9436|0.9267|0.9079|0.9127|
|DC-CNN|0.9686|0.9680|0.9409|0.9543|
|RTA|0.9739|0.9770|0.9793|0.9782|

If you want to fine-tune RTA with multiple GPUs, you can use the following command:
```python
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch \
  --nproc_per_node 4 run_dup_br_detection.py \
  --model_name_or_path Colorful/RTA \
  --train_file ./open_office/train.csv \
  --validation_file ./open_office/valid.csv \
  --test_file ./open_office/test.csv \
  --cache_dir ./cache_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 10 \
  --save_steps 10000 \
  --output_dir ./openoffice_results
```
