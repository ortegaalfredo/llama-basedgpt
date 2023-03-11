#Example config: Run the LLaMa 13B model in 2xRTX390

torchrun --nproc_per_node 2 bot.py --ckpt_dir ../llama/data/13B --tokenizer_path ../llama/data/tokenizer.model
