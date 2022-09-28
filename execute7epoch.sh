### Dependencies and compilation scripts
sh compile.sh
## run training
echo 'startiing the execution of the script now'

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29508./tools/dist_train.sh configs/vid/fgfa/gs_fgfa_faster_rcnn_r50_dc5_1x_imagenetvid.py 4
