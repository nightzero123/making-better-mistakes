OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss soft-labels \
    --lr 1e-5 \
    --data tiered-imagenet-224 \
    --beta 4 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output softlabels_tieredimagenet_beta04/ \
    --num_training_steps 200000 \

