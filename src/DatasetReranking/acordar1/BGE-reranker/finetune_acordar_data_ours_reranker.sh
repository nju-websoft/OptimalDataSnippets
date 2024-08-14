
export  CUDA_VISIBLE_DEVICES=0,1,2,3

minehn_acordar_data() {
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path "/path/to/model/bge-large-en-v1.5" \
    --input_file "../../../../data/rerank/acordar1/$3_$2_split_$1$((($1+1)%5))$((($1+2)%5)).jsonl" \
    --output_file "../../../../data/rerank/acordar1/$3_$2_split_$1$((($1+1)%5))$((($1+2)%5))_minedHN.jsonl" \
    --range_for_sampling "2-200" 
}

train_acordar_data() {
    torchrun --nproc_per_node 4 \
    --master_port 25901 \
    -m FlagEmbedding.reranker.run \
    --output_dir "../../../../data/rerank_outputs/acordar1/$6_$2_reranker/fold$1/lr_$3_bs_$4_epoch_$5" \
    --model_name_or_path "/path/to/model/bge-reranker-large" \
    --train_data "../../../../data/rerank/acordar1/$6_$2_split_$1$((($1+1)%5))$((($1+2)%5))_minedHN.jsonl" \
    --learning_rate $3 \
    --fp16 \
    --num_train_epochs $5 \
    --per_device_train_batch_size $4 \
    --gradient_accumulation_steps 4 \
    --dataloader_drop_last True \
    --train_group_size 4 \
    --max_len 512 \
    --weight_decay 0.01 
}

for snippet in ours
do
    for size in 20
    do
        for fold in {0..4}
        do
            minehn_acordar_data $fold $size $snippet
        done
    done
done

for snippet in ours
do
    for size in 20
    do
        for fold in {0..4}
        do
            for bs in 2
            do
                for lr in 1e-5 3e-5 5e-5
                do
                    for epoch in 10
                    do
                        train_acordar_data $fold $size $lr $bs $epoch $snippet
                    done
                done
            done
        done
    done
done