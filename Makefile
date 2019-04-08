check:
	black --check formatml setup.py
	mypy formatml
	flake8 --count
	pylint formatml setup.py

bblfsh-start:
	! docker ps | grep formatml_bblfshd # bblfsh server has been run already.
	docker run -d --rm --name formatml_bblfshd --privileged -p 9999\:9432 \
		bblfsh/bblfshd\:v2.12.0 --log-level DEBUG
	docker exec formatml_bblfshd bblfshctl driver install \
javascript docker://bblfsh/javascript-driver\:v2.7.3

.PHONY: check bblfsh-start
