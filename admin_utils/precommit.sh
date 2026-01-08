set -ex

echo $1

DIRS_TO_CHECK=(
  "admin_utils"
  "config"
  "core_utils"
  # "lab_7_llm"
  # "lab_8_sft"
  # "reference_lab_classification"
  # "reference_lab_classification_sft"
  # "reference_lab_generation"
  # "reference_lab_nli"
  # "reference_lab_nli_sft"
  # "reference_lab_nmt"
  # "reference_lab_nmt_sft"
  # "reference_lab_ner"
  # "reference_lab_open_qa"
  # "reference_lab_summarization"
  # "reference_lab_summarization_sft"
  "seminars"
)

if [ -d "venv" ]; then
    echo "Taking Python from venv"
    source venv/bin/activate
    which python
else
    echo "Taking Python from global environment"
    which python
fi

export PYTHONPATH=$(pwd)

# python config/generate_stubs/generate_labs_stubs.py

# python -m black "${DIRS_TO_CHECK[@]}"

# isort .

# autoflake -vv .

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

pydoctest --config pydoctest.json

if [[ "$1" != "smoke" ]]; then
  python config/static_checks/check_doc8.py

  python config/static_checks/requirements_check.py

  rm -rf dist
  sphinx-build -b html -W --keep-going -n . dist -c admin_utils

  python -m pytest -m "mark10 and lab_7_llm"
  python -m pytest -m "mark10 and lab_8_sft"
fi

