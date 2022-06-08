reproduce: install finetune evaluate
install:
	pip install -r requirements.txt
finetune:
	python exp/clip/finetune.py
evaluate:
	python exp/clip/evaluate.py



