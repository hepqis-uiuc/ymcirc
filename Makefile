.ONESHELL:
VENV_DIR=.venv
ACTIVATE_VENV:=. $(VENV_DIR)/bin/activate
PYTHON_HEADER_PATH:= $(shell python3-config --cflags | cut -d' ' -f1)

venv: requirements.txt
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt

clean:
	rm -rf .venv
	rm -rf __pycache__
