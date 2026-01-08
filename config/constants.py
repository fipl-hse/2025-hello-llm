"""
Useful constant variables.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONFIG_PACKAGE_PATH = PROJECT_ROOT / "config"
CORE_UTILS_PACKAGE_PATH = PROJECT_ROOT / "config"
TRACKED_JSON_PATH = str(
    (PROJECT_ROOT / "admin_utils" / "external_pr_files" / "tracked_files.json").relative_to(
        PROJECT_ROOT
    )
)
