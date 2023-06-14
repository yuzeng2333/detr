all: 
	python main.py --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/GCLN

#dbg:
#	python -m pdb main.py --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/GCLN

orig:
	python -m pdb main_orig.py --coco_path /home/yuzeng/workspace/research/inv_gen/coco_data
 
syn: 
	python main.py --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/synthetic

dnn:
	python main.py --sel_model dnn --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/synthetic

dbg-syn:
	python -m pdb main.py --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/synthetic

dbg:
	python -m pdb main.py --sel_model transformer --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/synthetic

tran:
	python main.py --sel_model transformer --invar_path /home/yuzeng/workspace/research/inv_gen/detr/datasets/invar_data/synthetic
