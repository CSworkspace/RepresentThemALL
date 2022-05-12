### Bug priority & severity prediction

We provide the code for fine-tuning BureBERT on bug priority & severity prediction. In fact, we just add an additional classification layer to BureBERT. To help you quickly run the code, we also provide a shell script to run the source code. So you can directly run our code by the following command: <br />
```bash
bash run_burebert_ft.sh
```

Note that you should replace the input of the following parameter with your own path that stores the dataset:
- --train_file
- --validation_file
- --test_file

### Extra Results
Except the experimental results we present in the paper, we also compare BureBERT with some effective approaches used in bug priority & severity prediction tasks. The results are shown in the following figure:

![figure](https://github.com/BureBERT/BureBERT/blob/main/picture/pred.png)
