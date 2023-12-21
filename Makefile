.PHONY: train

train:
	cd train && sh start_servers.sh

stop:
	cd train && sh stop_servers.sh