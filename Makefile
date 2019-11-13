train:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python train.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__

test:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python test.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__

train_infcomp:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python train_infcomp.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__

test_infcomp:
	[ -n "$(CONFIG)" ] # Please specify path to a config file
	python test_infcomp.py $(CONFIG)
	rm -rf __pycache__ data_loader/__pycache__ neural_net/__pycache__ util/__pycache__