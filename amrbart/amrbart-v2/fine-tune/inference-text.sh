#DataPath=`pwd`/../examples
#OutputDir=/data/local/amrs3/outputs
#MODEL=/data/local/amrs3/finetune-masked/outputs/AMRBART-large-AMR2Text-bsz8-lr-2e-6-UnifiedInp/checkpoint-80000
DataPath=$1
OutputDir=$2
MODEL=$3

BASEDIR=$(dirname "$0")
ModelCate=AMRBART-large

ModelCache=/data/local/hf_cache
DataCache=/data/local/amrs3/.cache/dump-amr2text

lr=2e-6

#if [ ! -d ${OutputDir} ];then
#  mkdir -p ${OutputDir}
#else
#  read -p "${OutputDir} already exists, delete origin one [y/n]?" yn
#  case $yn in
#    [Yy]* ) rm -rf ${OutputDir}; mkdir -p ${OutputDir};;
#    [Nn]* ) echo "exiting..."; exit;;
#    * ) echo "Please answer yes or no.";;
#  esac
#fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

# torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=1 --rdzv_backend=c10d main.py \
CUDA_VISIBLE_DEVICES="" python -u $BASEDIR/main.py \
    --data_dir $DataPath \
    --task "amr2text" \
    --test_file $DataPath/data4generation.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --overwrite_cache True \
    --model_name_or_path $MODEL \
    --overwrite_output_dir \
    --unified_input True \
    --per_device_eval_batch_size 8 \
    --max_source_length 1024 \
    --max_target_length 400 \
    --val_max_target_length 400 \
    --generation_max_length 400 \
    --generation_num_beams 5 \
    --predict_with_generate \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --seed 42 \
    --dataloader_num_workers 8 \
    --eval_dataloader_num_workers 2 \
    --include_inputs_for_metrics \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log
  #--fp16_backend "auto" \