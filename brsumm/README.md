### Bug report summarization

We provide the source code for fine-tuning RTA on bug report summarization task. We combine RTA with Seq2Seq framework and use it to generate summary for the given bug report. To help you quickly run our code, you can just use the following command:
```bash
bash run_summarization.sh
```

Note that you should replace the input of the following three parameters with the path that you store the dataset:
- --train_file
- --validation_file
- --test_file

Different from bug priority & severity prediction or duplicate bug report detection, the architecture used for bug report summarization is based on Seq2Seq framework.

If you want to fine-tune RTA with multiple GPUs, you can use the following comman:
```python
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch \
  --nproc_per_node 4 run_summarization.py \
  --model_name_or_path Colorful/RTA \
  --train_file ./dataset/sum_train_clean.csv \
  --validation_file ./dataset/sum_valid_clean.csv \
  --test_file ./dataset/sum_test_clean.csv \
  --cache_dir ./cache_dir \
  --text_column 'input' \
  --summary_column 'target' \
  --pad_to_max_length \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --preprocessing_num_workers 12 \
  --max_source_length 256 \
  --max_target_length 20 \
  --val_max_target_length 20 \
  --num_beams 5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --save_steps 10000 \
  --output_dir ./results_sum
```
