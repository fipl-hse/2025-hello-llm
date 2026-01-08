set -ex

echo $1

if [[ "$1" == "public" ]]; then
  DIRS_TO_CHECK=(
    "admin_utils"
    "config"
    "core_utils"
    "seminars"
  )
else
  DIRS_TO_CHECK=(
    "admin_utils"
    "config"
    "core_utils"
    "seminars"
    "lab_7_llm"
    "lab_8_sft"
    "reference_lab_classification"
    "reference_lab_classification_sft"
    "reference_lab_generation"
    "reference_lab_nli"
    "reference_lab_nli_sft"
    "reference_lab_nmt"
    "reference_lab_nmt_sft"
    "reference_lab_ner"
    "reference_lab_open_qa"
    "reference_lab_summarization"
    "reference_lab_summarization_sft"
  )
fi

if [ -d "venv" ]; then
    echo "Taking Python from venv"
    source venv/bin/activate
    which python
else
    echo "Taking Python from global environment"
    which python
fi

export PYTHONPATH=$(pwd)

autoflake -vv .

if [[ "$1" != "public" ]]; then
  python config/generate_stubs/generate_labs_stubs.py
fi

python -m black "${DIRS_TO_CHECK[@]}"

isort .

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"

if [[ "$1" != "public" ]]; then
  pydoctest --config pydoctest.json
fi

if [[ "$2" != "smoke" ]]; then
  python config/static_checks/check_doc8.py

  python config/static_checks/requirements_check.py

  if [[ "$1" != "public" ]]; then
    rm -rf dist
    sphinx-build -b html -W --keep-going -n . dist -c admin_utils
  fi

  python -m pytest -m "mark10 and lab_7_llm"
  python -m pytest -m "mark10 and lab_8_sft"
fi
