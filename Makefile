check:
	black --check formatml setup.py
	mypy formatml setup.py
	flake8 --count formatml setup.py
	pylint formatml setup.py

bblfshd:
	docker start formatml_bblfshd > /dev/null 2>&1 \
		|| docker run \
			--detach \
			--rm \
			--name formatml_bblfshd \
			--privileged \
			--publish 9432:9432 \
			bblfsh/bblfshd:v2.14.0-drivers \
			--log-level DEBUG

.PHONY: check bblfshd