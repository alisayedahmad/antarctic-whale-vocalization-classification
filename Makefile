.PHONY: help install test prepare train evaluate clean app

help:
	@echo "Antarctic Whale Vocalization Classification"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Test installation"
	@echo "  make prepare      - Prepare dataset"
	@echo "  make train        - Train all models"
	@echo "  make train-svm    - Train SVM model"
	@echo "  make train-rf     - Train Random Forest model"
	@echo "  make train-xgb    - Train XGBoost model"
	@echo "  make train-nn     - Train Neural Network model"
	@echo "  make evaluate     - Evaluate models"
	@echo "  make app          - Launch Streamlit app"
	@echo "  make clean        - Clean generated files"
	@echo "  make clean-all    - Clean all including data"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e .
	pip install pytest black flake8 jupyter

test:
	python test_install.py

prepare:
	python scripts/prepare_data.py --config config/config.yaml

train:
	python scripts/train.py --model all --config config/config.yaml

train-svm:
	python scripts/train.py --model svm --grid_search

train-rf:
	python scripts/train.py --model rf --grid_search

train-xgb:
	python scripts/train.py --model xgb

train-lgbm:
	python scripts/train.py --model lgbm

train-nn:
	python scripts/train.py --model nn

train-bayes:
	python scripts/train.py --model bayes

evaluate:
	@echo "Evaluating all models..."
	@for model in svm rf xgb nn bayes; do \
		if [ -f "models/$${model}_model.pkl" ]; then \
			echo "Evaluating $$model..."; \
			python scripts/evaluate.py --model_type $$model --model_path models/$${model}_model.pkl; \
		fi \
	done

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache
	rm -rf logs/* results/* visualizations/* 2>/dev/null || true

clean-all: clean
	rm -rf data/processed/*
	rm -rf models/*

format:
	black src/ scripts/ app/ --line-length 100

lint:
	flake8 src/ scripts/ app/ --max-line-length 100

notebook:
	jupyter notebook notebooks/
