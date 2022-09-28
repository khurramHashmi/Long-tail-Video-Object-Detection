srun -K --container-image=/netscratch/hashmi/enroots/mmcv_1.4.8_mmdet_2.23_mmtrack_0.12_torch_1.10.sqsh \
--container-workdir=`pwd` \
--container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/nislam/fgfa/mmtracking:/netscratch/nislam/fgfa/mmtracking/ \
-p A100 --gpus=4 --cpus-per-gpu=2 --time=3-00:00:00 \
--job-name="Training_7_Epoch" sh execute7epoch.sh
