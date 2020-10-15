PIP := pip3

.PHONY: help
help: ## display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: init
init: ## install package
	${PIP} install -r requirements.txt
	## ${PIP} install -r requirements.test.txt
	${PIP} install .

## .PHONY: python_tests
## python_tests: ## run unit tests
## 	pytest --cov=cpcspeech test/
## 	flake8