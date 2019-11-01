train:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python train.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__

test:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python test.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__
