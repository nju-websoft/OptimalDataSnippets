export  CUDA_VISIBLE_DEVICES=0,1,2,3

minehn_ntcir_data() {
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path "/path/to/model/bge-large-en-v1.5" \
    --input_file "../../../../data/rerank/ntcir15/$1_$2_split_train.jsonl" \
    --output_file "../../../../data/rerank/ntcir15/$1_$2_split_train_minedHN.jsonl" \
    --range_for_sampling "2-200" 
}

train_ntcir_data() {
    torchrun --nproc_per_node 4 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir "../../../../data/rerank_outputs/ntcir15/$1_$2/lr_$3_bs_$4_epoch_$5" \
    --model_name_or_path "/path/to/model/bge-large-en-v1.5" \
    --train_data "../../../../data/rerank/ntcir15/$1_$2_split_train_minedHN.jsonl" \
    --learning_rate $3 \
    --fp16 \
    --num_train_epochs $5 \
    --per_device_train_batch_size $4 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 16 \
    --passage_max_len 512 \
    --train_group_size 4 \
    --negatives_cross_device \
    --query_instruction_for_retrieval "Represent this sentence for searching relevant passages: "
}


for snippet_method in ours
do
    for size in 20
    do
        minehn_ntcir_data $snippet_method $size
    done
done

for snippet_method in ours
do
    for size in 20
    do
        for bs in 2
        do
            for lr in 1e-5 3e-5 5e-5
            do
                for epoch in 10
                do
                    train_ntcir_data $snippet_method $size $lr $bs $epoch
                done
            done
        done
    done
done