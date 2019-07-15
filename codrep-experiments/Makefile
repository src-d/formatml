CODREP_NETWORK ?= formatml
CODREP_BBLFSHD_NAME ?= formatml_bblfshd
CODREP_COMMAND ?= ~/.virtualenvs/formatml/bin/python -m formatml codrep
CODREP_DIR ?= .
CODREP_STEPS_DIR ?= $(CODREP_DIR)/steps
CODREP_DATA_DIR ?= $(CODREP_DIR)/data
CODREP_ARCHIVE_URL ?= https://github.com/KTH/codrep-2019/archive/master.zip
CODREP_ARCHIVE ?= $(CODREP_DATA_DIR)/master.zip
CODREP_RAW_DIR ?= $(CODREP_DATA_DIR)/raw
CODREP_UASTS_DIR ?= $(CODREP_DATA_DIR)/uasts
CODREP_INSTANCE_ARCHIVE ?= $(CODREP_DATA_DIR)/instance.pickle.bz2
CODREP_TENSORS_DIR ?= $(CODREP_DATA_DIR)/tensors
CODREP_TRAIN_DIR ?= $(CODREP_DATA_DIR)/train

setup:
	mkdir -p $(CODREP_STEPS_DIR) $(CODREP_DATA_DIR)
	docker network create $(CODREP_NETWORK) > /dev/null 2>&1 || true

bblfshd: setup
	docker start $(CODREP_BBLFSHD_NAME) > /dev/null 2>&1 \
		|| docker run \
			--detach \
			--rm \
			--name $(CODREP_BBLFSHD_NAME) \
			--network $(CODREP_NETWORK) \
			--privileged \
			--publish 9432:9432 \
			bblfsh/bblfshd:v2.14.0-drivers \
			--log-level DEBUG

download: setup
	dvc import-url \
		--file $(CODREP_STEPS_DIR)/download.dvc \
		$(CODREP_ARCHIVE_URL) \
		$(CODREP_ARCHIVE)

unzip: setup download
	dvc run \
		--overwrite-dvcfile \
		--file $(CODREP_STEPS_DIR)/unzip.dvc \
		--deps $(CODREP_ARCHIVE) \
		--outs $(CODREP_RAW_DIR) \
		"unzip $(CODREP_ARCHIVE) \
			&& mv codrep-2019-master/Datasets/learning $(CODREP_RAW_DIR) \
			&& rm -rf codrep-2019-master"

parse: setup bblfshd unzip
	dvc run \
		--overwrite-dvcfile \
		--file $(CODREP_STEPS_DIR)/parse.dvc \
		--deps $(CODREP_RAW_DIR) \
		--outs $(CODREP_UASTS_DIR) \
		$(CODREP_COMMAND) parse \
			$(CODREP_RAW_DIR) \
			$(CODREP_UASTS_DIR) 

index: setup parse
	dvc run \
		--overwrite-dvcfile \
		--file $(CODREP_STEPS_DIR)/index.dvc \
		--deps $(CODREP_UASTS_DIR) \
		--outs $(CODREP_INSTANCE_ARCHIVE) \
		$(CODREP_COMMAND) index \
			$(CODREP_UASTS_DIR) \
			$(CODREP_INSTANCE_ARCHIVE) \
			$(CODREP_INDEX_ARGS)

tensorize: setup index
	dvc run \
		--overwrite-dvcfile \
		--file $(CODREP_STEPS_DIR)/tensorize.dvc \
		--deps $(CODREP_UASTS_DIR) \
		--deps $(CODREP_INSTANCE_ARCHIVE) \
		--outs $(CODREP_TENSORS_DIR) \
		$(CODREP_COMMAND) tensorize \
			$(CODREP_UASTS_DIR) \
			$(CODREP_INSTANCE_ARCHIVE) \
			$(CODREP_TENSORS_DIR) \
			$(CODREP_TENSORIZE_ARGS)

train: setup tensorize
	dvc run \
		--overwrite-dvcfile \
		--file $(CODREP_STEPS_DIR)/train.dvc \
		--deps $(CODREP_INSTANCE_ARCHIVE) \
		--deps $(CODREP_TENSORS_DIR) \
		--outs $(CODREP_TRAIN_DIR) \
		$(CODREP_COMMAND) train \
			$(CODREP_INSTANCE_ARCHIVE) \
			$(CODREP_TENSORS_DIR) \
			$(CODREP_TRAIN_DIR) \
			$(CODREP_TRAIN_ARGS)

.PHONY: bblfshd setup download unzip parse index tensorize train
