### Bug priority & severity prediction

We provide the code for fine-tuning RTA on bug priority & severity prediction. In fact, we just add an additional classification layer to RTA. To help you quickly run the code, we also provide a shell script to run the source code. So you can directly run our code by the following command: <br />
```bash
bash run_rta_ft.sh
```

Note that you should replace the input of the following parameter with your own path that stores the dataset:
- --train_file
- --validation_file
- --test_file
