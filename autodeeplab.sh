LAYER=12
DATASET=sealer
NAME=autodeeplab
CHECKNAME=${NAME}_layer${LAYER}
MODELPATH=run/${DATASET}/${CHECKNAME}/model_best.pth.tar
SAVEPATH=run/${DATASET}/${CHECKNAME}/models/
NETPATH=run/${DATASET}/${CHECKNAME}/network_path.npy
CELLPATH=run/${DATASET}/${CHECKNAME}/genotype.npy

echo "\n\n********** Search DeepLabv3 **********\n\n"
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --batch-size 16 --dataset $DATASET --num_layers $LAYER --checkname $CHECKNAME

echo "\n\n********** Decode Architecture **********\n\n"
CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset $DATASET --batch_size 16 --resume $MODELPATH

echo "\n\n********** Training Architecture **********\n\n"
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 32 --epochs 300 --warmup-iters 250 --checkname $SAVEPATH --net_arch $NETPATH --cell_arch $CELLPATH --num_layers $LAYER

run/sealer/autodeeplab_layer6/genotype.npy