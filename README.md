This is source code for EMNLP 2020  'Towards Interpretable Reasoning over Paragraph Effects in Situation'.

## Install

Code is based on hugging face transformer v2.3.0, to install all dependencies, including external libraries, or a link to such resources, following commands will help you to install easily.

```
pip install . 
pip install allennlp
pip install -r requirements.txt 
```

If you have problems about installing above package, especially the first line. The best way is go to [Huggingface](https://github.com/huggingface/transformers), find the version 2.3.0 then download it. 

After that , replace all the files in the **examples** and **src/transformers/data/** with this repositories.  Then continue installation. 

Be careful with **Pytorch** version, make sure it is compatible with both allennlp and transformers.

## Data & Model

We provide official data, auxiliary labeled data and 5-fold cross-validation data in [here](https://drive.google.com/file/d/1zBQw3rlYuV8ZdHO_wzXN7Rd4jo3peaTF/view?usp=sharing).

Finetuned Model will be uploaded in few days

## Training & Evaluation

The following script is used for evaluation models. If you want to train a model by yourself, just need to modify this script a bit.

```python
# Evaluation example for Interpretable Reasoning
CUDA_VISIBLE_DEVICES=0 python examples/run_ropes_interpretable.py   --model_type roberta   --model_name_or_path  /path/to/model  --do_eval  --do_lower_case --data_dir /path/to/data  --predict_file /path/to/file --max_seq_length 512  --doc_stride 64   --output_dir /path/to/output --overwrite_output_dir --gradient_accumulation_steps 1 --grounding_type s_first --overwrite_cache

# Evaluation example for Answer Prediction
CUDA_VISIBLE_DEVICES=0 python examples/run_ropes.py   --model_type roberta   --model_name_or_path /path/to/model  --do_eval --do_lower_case  --data_dir /path/to/data  --predict_file /path/to/file --max_seq_length 384  --doc_stride 128 --max_answer_length 9   --output_dir /path/to/output --grounding_type synthetic_2nd --multi_answer --overwrite_cache --overwrite_output_dir

```