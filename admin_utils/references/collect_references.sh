# Run: nohup bash admin_utils/references/collect_references.sh > references.log 2>&1 &
# Monitor: ps -ef | grep collect_references

set -ex

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

source venv/bin/activate

export PYTHONPATH=$(pwd)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# GPU measurements are for 1x A5000 (24 ГБ)
# ~2 GPU min
python admin_utils/references/get_datasets_analytics.py

# ~1 GPU min
python admin_utils/references/get_model_analytics.py

# ~1 GPU min
python admin_utils/references/get_inference_analytics.py

# ~19 GPU min
python admin_utils/references/get_references.py

# ~9 GPU min
python admin_utils/references/get_sft_references.py

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;
