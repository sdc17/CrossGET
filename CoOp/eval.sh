#!/bin/bash

# custom config
DATA=/mnt/lustre/shidachuan/datasets
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=$2
REDUCE=$3
RV=$4
RL=$5

for SEED in 1
do
    python -W ignore -m torch.distributed.run --nproc_per_node=8 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${REDUCE}_${RV}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --eval-only \
    --reduce ${REDUCE} \
    --rv ${RV} \
    --rl ${RL} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done