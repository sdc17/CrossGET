#!/bin/bash

# custom config
DATA=/mnt/lustre/shidachuan/datasets
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
REDUCE=$7
RV=$8
RL=$9

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_${REDUCE}_${RV}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    python -W ignore -m torch.distributed.run --nproc_per_node=8 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${DATASET}_original/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --reduce ${REDUCE} \
    --rv ${RV} \
    --rl ${RL} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
done

