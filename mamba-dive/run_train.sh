python train_mamba_with_context.py --model state-spaces/mamba-2.8b \
   --data_path /data/npl/ViSignboardVQA/data/squad_train.jsonl \
   --output models/mamba-2.8b-context \
   --num_epochs 10 \
   --batch_size 64