"""
Python tool for synchronization between source and target repositories.
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from logging518.config import fileConfig
from pydantic import BaseModel, Field, ValidationError
from quality_control.cli_unifier import _run_console_tool, handles_console_error
from quality_control.console_logging import get_child_logger
from quality_control.quality_control_parser import QualityControlArgumentsParser

from admin_utils.constants import TRACKED_JSON_PATH

logger = get_child_logger(__file__)


class SyncArgumentParser(QualityControlArgumentsParser):
    """
    Parser that gets args for sync tool
    """

    repo_name: str
    pr_number: str


class PRData(BaseModel):
    """
    Model for information about PR
    """

    mergedAt: Optional[datetime] = None
    headRefName: str
    baseRefName: str = Field(default="main")


@dataclass(slots=True)
class CommitConfig:
    """
    Storage for commit data
    """

    repo_path: str
    branch_name: str
    repo_name: str
    pr_number: str
    files_to_sync_found: bool


@dataclass(slots=True)
class SyncConfig:
    """
    Storage for final PR data
    """

    target_repo: str
    changed_files: list[str]
    json_content: Optional[dict]
    commit_sha: str


@dataclass(slots=True)
class SyncResult:
    """
    Result of synchronization operation
    """

    has_changes: bool
    files_to_sync_found: bool


# Wrappers for basic commands
@handles_console_error(ok_codes=(0, 1, 128))
def run_git(args: list[str], **kwargs: str) -> tuple[str, str, int]:
    """
    Run git command via imported function

    Args:
        args (list[str]): Arguments for git command.
        kwargs (str): Keyword arguments.

    Returns:
        tuple[str, str, int]: Result of git command.
    """
    return _run_console_tool("git", args, **kwargs)


@handles_console_error(ok_codes=(0, 1))
def run_gh(args: list[str]) -> tuple[str, str, int]:
    """
    Run gh command via imported function

    Args:
        args (list[str]): Arguments for gh command.

    Returns:
        tuple[str, str, int]: Result of gh command.
    """
    return _run_console_tool("gh", args)


def get_pr_data(repo_name: str, pr_number: str) -> Optional["PRData"]:
    """
    Get PR data via gh

    Args:
        repo_name (str): Name of source repo.
        pr_number (str): Number of needed PR in source repo.

    Returns:
        Optional["PRData"]: PR data.
    """
    stdout, stderr, return_code = run_gh(
        [
            "pr",
            "view",
            pr_number,
            "--repo",
            repo_name,
            "--json",
            "mergedAt,headRefName,baseRefName",
        ]
    )

    if return_code != 0 or not stdout:
        logger.warning("Failed to get PR data: %s", stderr)
        return None

    try:
        data = json.loads(stdout)
        return PRData.model_validate(data)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from gh: %s", e)
        return None
    except ValidationError as e:
        logger.error("PR data validation failed: %s", e)
        return None


def check_branch_exists(branch_name: str, repo_path: str = ".") -> bool:
    """
    Check if branch in remote repo exists

    Args:
        branch_name (str): Name of needed branch.
        repo_path (str, optional): Path to repo. Defaults to ".".

    Returns:
        bool: True if needed branch exists in remote repo.
    """
    _, _, return_code = run_git(
        ["show-ref", "--quiet", f"refs/remotes/origin/{branch_name}"], cwd=repo_path
    )
    return bool(return_code == 0)


def clone_repo(target_repo: str, gh_token: str) -> None:
    """
    Clone target repo

    Args:
        target_repo (str): Name of target repo.
        gh_token (str): Token to process operations.
    """
    target_path = Path(target_repo)
    if target_path.exists():
        shutil.rmtree(target_path)

    run_git(["clone", f"https://{gh_token}@github.com/fipl-hse/{target_repo}.git"])


def setup_git_config(repo_path: str) -> None:
    """
    Setup config

    Args:
        repo_path (str): Path to repo.
    """
    run_git(["config", "user.name", "github-actions[bot]"], cwd=repo_path)
    run_git(
        ["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        cwd=repo_path,
    )


def checkout_or_create_branch(branch_name: str, repo_path: str) -> None:
    """
    Checkout on existing branch or create it

    Args:
        branch_name (str): Name of needed branch.
        repo_path (str): Path to repo.
    """
    if check_branch_exists(branch_name, repo_path):
        run_git(["checkout", branch_name], cwd=repo_path)
        run_git(["pull", "origin", branch_name], cwd=repo_path)
    else:
        run_git(["checkout", "-b", branch_name], cwd=repo_path)


def add_remote_and_fetch(remote_name: str, repo_url: str, repo_path: str) -> None:
    """
    Add remote and fetch.

    Args:
        remote_name (str): Name of remote repo.
        repo_url (str): Link to remote repo.
        repo_path (str): Path to remote repo.
    """
    stdout, _, _ = run_git(["remote"], cwd=repo_path)
    remotes = stdout.split()

    if remote_name not in remotes:
        run_git(["remote", "add", remote_name, repo_url], cwd=repo_path)

    run_git(["fetch", remote_name], cwd=repo_path)


def get_json_from_source(source_ref: str, target_repo: str) -> Optional[dict]:
    """
    Get JSON content from source reference and update local file if changed.

    Args:
        source_ref (str): Reference in source repo.
        target_repo (str): Path to target repository.

    Returns:
        Optional[dict]: Parsed JSON content, or None if not found/failed.
    """
    stdout, _, return_code = run_git(["show", f"{source_ref}:{TRACKED_JSON_PATH}"], cwd=target_repo)

    if return_code != 0 or not stdout:
        logger.error("JSON file not found or failed to read from ref '%s'", source_ref)
        return None

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from ref '%s': %s", source_ref, e)
        return None


def get_sync_mapping(json_content: Optional[dict]) -> list[tuple[str, str]]:
    """
    Extract sync mapping from JSON.

    Args:
        json_content (Optional[dict]): Content of JSON file.

    Returns:
        list[tuple[str, str]]: Mapping of source/target files from JSON.
    """
    sync_mapping: list[tuple[str, str]] = []

    if not json_content:
        return []

    for item in json_content:
        source = item.get("source")
        target = item.get("target")
        if source and target:
            sync_mapping.append((source, target))
    return sync_mapping


def sync_files_from_source(
    repo_path: str, source_ref: str, sync_list: list[tuple[str, str]]
) -> bool:
    """
    Sync files from source reference into target repo according to mapping.

    Args:
        repo_path (str): Path to target repo.
        source_ref (str): Reference in source repo.
        sync_list (list[tuple[str, str]]): List of (source_path, target_path).

    Returns:
        bool: True if any file was updated/added/removed.
    """
    has_changes = False
    for source_path, target_path in sync_list:
        stdout, _, return_code = run_git(["show", f"{source_ref}:{source_path}"], cwd=repo_path)
        full_target_path = Path(repo_path) / target_path

        if return_code == 0 and stdout:
            full_target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_target_path, "w", encoding="utf-8") as f:
                f.write(stdout)
            run_git(["add", target_path], cwd=repo_path)
            has_changes = True
        else:
            if full_target_path.exists():
                run_git(["rm", target_path], cwd=repo_path)
                has_changes = True
            else:
                logger.info(
                    "File %s not found in source and not present in target, nothing to do",
                    source_path,
                )
    return has_changes


def run_sync(target_repo: str, source_ref: str, json_content: Optional[dict]) -> SyncResult:
    """
    Run synchronization: compare files from mapping and update if needed.

    Args:
        target_repo (str): Path to target repository.
        source_ref (str): Reference in source repository.
        json_content (Optional[dict]): Parsed JSON content.

    Returns:
        SyncResult: Result of sync operation.
    """
    files_to_sync_found = False

    if json_content is None:
        return SyncResult(has_changes=False, files_to_sync_found=False)

    sync_mapping = get_sync_mapping(json_content)
    files_to_sync = []

    for source_path, target_path in sync_mapping:
        source_sha, _, return_code_source = run_git(
            ["rev-parse", f"{source_ref}:{source_path}"], cwd=target_repo
        )
        if return_code_source != 0:
            source_sha = None

        target_sha, _, return_code_target = run_git(
            ["rev-parse", f"origin/main:{target_path}"], cwd=target_repo
        )
        if return_code_target != 0:
            target_sha = None

        if source_sha != target_sha:
            files_to_sync.append((source_path, target_path))
            files_to_sync_found = True

    synced = False
    if files_to_sync:
        synced = sync_files_from_source(target_repo, source_ref, files_to_sync)

    return SyncResult(
        has_changes=synced,
        files_to_sync_found=files_to_sync_found,
    )


def commit_and_push_changes(commit_config: CommitConfig) -> None:
    """
    Commit and push changes

    Args:
        commit_config (CommitConfig): Schema of Commit.
    """
    commit_msg = f"Sync changes from {commit_config.repo_name} PR {commit_config.pr_number}"

    run_git(["commit", "-m", commit_msg], cwd=commit_config.repo_path)
    run_git(["push", "origin", commit_config.branch_name], cwd=commit_config.repo_path)


def create_or_update_pr(
    target_repo: str, branch_name: str, repo_name: str, pr_number: str, repo_path: str
) -> None:
    """
    Create or update PR in target repo

    Args:
        target_repo (str): Name of source repo.
        branch_name (str): Name of needed branch.
        repo_name (str): Name of target repo
        pr_number (str): Number of source PR.
        repo_path (str): Path to repo.
    """
    stdout, stderr, return_code = run_gh(
        [
            "pr",
            "list",
            "--repo",
            f"fipl-hse/{target_repo}",
            "--head",
            branch_name,
            "--json",
            "number",
        ]
    )

    target_pr_number = None
    if return_code == 0 and stdout:
        pr_list = json.loads(stdout) if stdout else []
        if pr_list and len(pr_list) > 0:
            target_pr_number = pr_list[0].get("number")

    run_git(["fetch", "origin", "main"], cwd=repo_path)

    stdout, stderr, return_code = run_git(
        ["log", "--oneline", f"origin/main..{branch_name}"], cwd=repo_path
    )

    has_commits = return_code == 0 and bool(stdout and stdout.strip())

    if has_commits:
        if target_pr_number is None:
            stdout, stderr, return_code = run_gh(
                [
                    "pr",
                    "create",
                    "--repo",
                    f"fipl-hse/{target_repo}",
                    "--head",
                    branch_name,
                    "--base",
                    "main",
                    "--title",
                    f"[Automated] Sync from {repo_name} PR {pr_number}",
                    "--body",
                    f"Automated synchronization from {repo_name} PR #{pr_number}",
                    "--label",
                    "automated pr",
                    "--assignee",
                    "demid5111",
                ]
            )

            if return_code == 1:
                logger.error("Failed to create PR. Exit code: %s", return_code)
                logger.error("stdout: %s", stdout)
                logger.error("stderr: %s", stderr)
                sys.exit(1)

            logger.info("Created new PR in target repository")

        else:
            stdout, stderr, return_code = run_gh(
                [
                    "pr",
                    "comment",
                    str(target_pr_number),
                    "--repo",
                    f"fipl-hse/{target_repo}",
                    "--body",
                    "Automatically updated",
                ]
            )

            if return_code != 0:
                logger.warning("Failed to update PR %s", target_pr_number)
    else:
        logger.info("No commits in branch %s - skipping PR creation", branch_name)


def validate_and_process_inputs() -> tuple[str, ...]:
    """
    Validating input args and processing basic information for script work

    Returns:
        tuple[str, ...]: Needed data from source repo
    """
    parser = SyncArgumentParser(underscores_to_dashes=True)
    args = parser.parse_args()

    repo_name = args.repo_name
    pr_number = args.pr_number
    target_repo = "fipl-hse.github.io"
    branch_name = f"auto-update-from-{repo_name}-pr-{pr_number}"
    root_dir = args.root_dir.resolve()
    toml_config = (args.toml_config_path or (root_dir / "pyproject.toml")).resolve()
    fileConfig(toml_config)

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        logger.error("GH_TOKEN environment variable is not set")
        sys.exit(1)

    return repo_name, pr_number, target_repo, branch_name, gh_token


def prepare_target_repo(target_repo: str, branch_name: str, gh_token: str) -> None:
    """
    Prepare target repo for PR creation

    Args:
        target_repo (str): Name of target repo.
        branch_name (str): Name of branch in target repo.
        gh_token (str): Token to process operations.
    """
    clone_repo(target_repo, gh_token)
    setup_git_config(target_repo)
    checkout_or_create_branch(branch_name, target_repo)


def main() -> None:
    """
    Main function to create PR in target repo
    """
    repo_name, pr_number, target_repo, branch_name, gh_token = validate_and_process_inputs()

    prepare_target_repo(target_repo, branch_name, gh_token)

    pr_data = get_pr_data(repo_name, pr_number)
    if not pr_data:
        logger.error("PR data in source repo not found")
        sys.exit(0)

    merged_at = pr_data.mergedAt
    head_ref = pr_data.headRefName
    base_ref = pr_data.baseRefName

    if not head_ref:
        logger.error("Could not get head branch name from PR")
        sys.exit(0)

    add_remote_and_fetch(
        "parent-repo", f"https://{gh_token}@github.com/{repo_name}.git", target_repo
    )

    if merged_at:
        source_ref = f"parent-repo/{base_ref}"
        logger.info("PR is merged, comparing %s with target main", source_ref)
    else:
        source_ref = f"parent-repo/{head_ref}"
        logger.info("PR is open, comparing %s with target main", source_ref)

    run_git(["fetch", "origin", "main"], cwd=target_repo)

    json_content = get_json_from_source(source_ref, target_repo)

    if json_content is None:
        logger.error("Could not retrieve JSON, aborting")
        sys.exit(1)

    sync_result = run_sync(target_repo, source_ref, json_content)

    if sync_result.has_changes:
        commit_config = CommitConfig(
            target_repo,
            branch_name,
            repo_name,
            pr_number,
            sync_result.files_to_sync_found,
        )
        commit_and_push_changes(commit_config)
        create_or_update_pr(target_repo, branch_name, repo_name, pr_number, target_repo)
    else:
        logger.info("No changes to commit")
        sys.exit(0)


if __name__ == "__main__":
    main()
