all: 
	python main.py --invar_path ./datasets/invar_data/GCLN

#dbg:
#	python -m pdb main.py --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/GCLN

orig:
	python -m pdb main_orig.py --coco_path /home/yuzeng/workspace/research/inv_gen/coco_data
 
syn: 
	python main.py --invar_path ./datasets/invar_data/synthetic

dnn:
	python main.py --sel_model dnn --invar_path ./datasets/invar_data/synthetic

dbg:
	#python -m pdb main.py --sel_model double --invar_path ./datasets/invar_data/synthetic_two --device cpu --enable_perm 1 --num_iterations 500 --perm_num 1
	#python -m pdb main.py --early_stop 1 --stop_batch_num 1 --sel_model double --batch_size 8 --invar_path ./datasets/invar_data/synthetic_two --device cpu --enable_perm 1 --num_iterations 500 --perm_num 2
	python -m pdb main.py --early_stop 1 --stop_batch_num 1 --sel_model double --batch_size 40 --invar_path ./datasets/invar_data/synthetic_two --enable_perm 1 --num_iterations 500 --perm_num 2

tran:
	python main.py --sel_model transformer --invar_path ./datasets/invar_data/synthetic

trial:
	python main.py --trial --num_iterations 2 --sel_model transformer --invar_path ./datasets/invar_data/synthetic

trial-dbg:
	python -m pdb main.py --trial --num_iterations 2 --sel_model transformer --invar_path ./datasets/invar_data/synthetic

gpu:
	python main.py --device cuda --sel_model transformer --invar_path ./datasets/invar_data/synthetic

two:
	python main.py --sel_model double --invar_path ./datasets/invar_data/synthetic_two --perm_num 2

pnt:
	python main.py --sel_model pointnet --invar_path ./datasets/invar_data/synthetic
 
double:
	python main.py --early_stop 1 --sel_model double --batch_size 8 --gpu_num 4 --invar_path ./datasets/invar_data/synthetic_two --device cpu --enable_perm 1 --num_iterations 500 --perm_num 2
