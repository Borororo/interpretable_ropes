#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python examples/run_ropes_interpretable.py   --model_type roberta   --model_name_or_path  /path/to/model  --do_eval  --do_lower_case --data_dir /path/to/data  --predict_file /path/to/file --max_seq_length 512  --doc_stride 64   --output_dir /path/to/output --overwrite_output_dir --gradient_accumulation_steps 1 --grounding_type s_first --overwrite_cache

CUDA_VISIBLE_DEVICES=0 python examples/run_ropes.py   --model_type roberta   --model_name_or_path /path/to/model  --do_eval --do_lower_case  --data_dir /path/to/data  --predict_file /path/to/file --max_seq_length 384  --doc_stride 128 --max_answer_length 9   --output_dir /path/to/output --grounding_type synthetic_2nd --multi_answer --overwrite_cache --overwrite_output_dir


