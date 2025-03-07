clear
GPU_ID=0


SPLIT=0
DATASET='s3dis'
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'

#DATASET='scannet'
#DATA_PATH='./datasets/ScanNet/blocks_bs1_s1'
#SAVE_PATH='./log_scannet/'


NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

PRETRAIN_CHECKPOINT='./log_s3dis/log_pretrain_s3dis_S0'
#PRETRAIN_CHECKPOINT='./log_scannet/log_pretrain_scannet_S0'
N_WAY=2
K_SHOT=1
N_QUESIES=1
N_TEST_EPISODES=100

NUM_ITERS=21000
#EVAL_INTERVAL=500
DECAY_STEP=4000

#NUM_ITERS=4000
EVAL_INTERVAL=20
#DECAY_STEP=1000

LR=0.001
DECAY_RATIO=0.5

N_SUBPROTOTYPES=100 # 每个class 对应 多少个原型
K_CONNECT=200
SIM_FUNCTION='gaussian'
SIGMA=1

MODEL_CHECKPOINT='./log_s3dis/log_mpti_s3dis_S0_N1_K1' #MAKE SURE IT CORRESPONDS TO SPLIT, N_WAY, K_SHOT.
#MODEL_CHECKPOINT='./log_scannet/log_mpti_scannet_S0_N2_K1' #MAKE SURE IT CORRESPONDS TO SPLIT, N_WAY, K_SHOT.

args=(--phase 'mptitrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" --use_attention
      --n_subprototypes $N_SUBPROTOTYPES  --k_connect $K_CONNECT
      --dist_method "$SIM_FUNCTION" --sigma $SIGMA
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS"
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --model_checkpoint_path "$MODEL_CHECKPOINT"
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)


CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"