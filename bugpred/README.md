### Bug priority & severity prediction

We provide the code for fine-tuning RTA on bug priority & severity prediction. In fact, we just add an additional classification layer to RTA. To help you quickly run the code, we also provide a shell script to run the source code. So you can directly run our code by the following command: <br />
```bash
bash run_br_pred.sh
```

Note that you should replace the input of the following parameter with your own path that stores the dataset:
- --train_file
- --validation_file
- --test_file

If you want to fine-tune RTA with multiple GPUs, you can use the following command:
```python
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch \
  --nproc_per_node 4 run_br_pred.py \
  --model_name_or_path Colorful/RTA \
  --train_file ./dataset/priority_train.csv \
  --validation_file ./dataset/priority_valid.csv \
  --test_file ./dataset/priority_test.csv \
  --cache_dir ./cache_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-6 \
  --num_train_epochs 10 \
  --save_steps 10000 \
  --output_dir ./results_priority
```
