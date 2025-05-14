torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 /private/yjy/project/VAR/moetrain.py \
  --depth=16 --bs=1024 --pn=256 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --data_path="/private/yjy/project/VAR/datasets/imagenet_organized" \
  --local_out_dir_path="/private/yjy/project/VAR/output/moevar_d16_15_4top2_256" \
  --var_ckpt_path="/private/yjy/project/VAR/training/var_d16.pth" \
  --moe_layer=15 \
  --moe_num_experts=4 \
  --moe_top_k=2 \
  --moe_router_type="softmax" \