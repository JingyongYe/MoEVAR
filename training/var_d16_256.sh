torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 /private/yjy/project/VAR/train.py \
  --depth=16 --bs=256 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --data_path="/private/yjy/project/VAR/datasets/imagenet_organized" \
  --local_out_dir_path="/private/yjy/project/VAR/output/var_d16_256" \