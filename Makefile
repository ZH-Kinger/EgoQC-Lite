.PHONY: setup test check self-test

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip setuptools wheel
	.venv/bin/python -m pip install -e . --no-build-isolation

test:
	PYTHONPATH=tests .venv/bin/python -m unittest discover -s tests -v

check:
	PYTHONPYCACHEPREFIX=/tmp/egoqc-pyc .venv/bin/python -m compileall -q src tests
	$(MAKE) test

self-test:
	.venv/bin/egoqc self-test
