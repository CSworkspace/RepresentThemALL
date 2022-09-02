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
