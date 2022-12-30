@REM python data/pre_process.py --data_folder "./data/CityCam" --ques "Q1"
@REM python train.py --project runs/train/Q1 --name binbin --workers 1 --device 0 --batch-size 16
@REM python test.py --verbose --show_div_cams --task test --weights "./runs/train/Q1/binbin/weights/best.pt" --project runs/test/Q1 --name binbin

@REM python data/pre_process.py --data_folder "./data/CityCam" --ques "Q2"
@REM python train.py --project runs/train/Q2 --name pca128_uniform --workers 1 --device 0 --batch-size 16
@REM python test.py --verbose --show_div_cams --task test --weights "./runs/train/Q2/pca128_uniform/weights/best.pt" --project runs/test/Q2 --name pca128_uniform

@REM label data from Q2 (200) + pseudo label from Q3 (1200)
@REM python data/pre_process.py --data_folder "./data/CityCam" --ques "Q3"
@REM python train.py --project runs/train/Q3 --name pseudo_label_thrs0.5 --workers 1 --device 0 --batch-size 16
@REM python test.py --verbose --show_div_cams --task test --weights "./runs/train/Q3/pseudo_label_thrs0.5/weights/best.pt" --project runs/test/Q3 --name pseudo_label_thrs0.5

@REM finetune training on Q2
@REM python data/pre_process.py --data_folder "./data/CityCam" --ques "Q2"
@REM python train.py --project runs/train/Q3 --name pseudo_label_thrs0.5_ft_pw_focal --workers 1 --device 0 --batch-size 16 --freeze 50 --weights "./runs/train/Q3/pseudo_label_thrs0.5/weights/best.pt" --hyp "data/hyp.finetune.yaml"
@REM python test.py --verbose --show_div_cams --task test --weights "./runs/train/Q3/pseudo_label_thrs0.5_ft_pw_focal/weights/best.pt" --project runs/test/Q3 --name pseudo_label_thrs0.5_ft_pw_focal

@REM pseudo label generation
@REM python detect.py --source ./data/CityCam/Q3/170 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 170 --save-txt --conf-thres 0.5 
@REM python detect.py --source ./data/CityCam/Q3/173 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 173 --save-txt --conf-thres 0.5 
@REM python detect.py --source ./data/CityCam/Q3/398 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 398 --save-txt --conf-thres 0.5 
@REM python detect.py --source ./data/CityCam/Q3/410 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 410 --save-txt --conf-thres 0.5 
@REM python detect.py --source ./data/CityCam/Q3/495 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 495 --save-txt --conf-thres 0.5 
@REM python detect.py --source ./data/CityCam/Q3/511 --weights ./runs/train/Q2/pca128_uniform/weights/best.pt --project runs/detect/Q3_0.5 --name 511 --save-txt --conf-thres 0.5 