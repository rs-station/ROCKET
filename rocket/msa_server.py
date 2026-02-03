"""MMseqs2 (ColabFold) MSA server utilities."""

from __future__ import annotations

import random
import tarfile
import time
from pathlib import Path

import requests
from loguru import logger


def _read_fasta_sequence(fasta_path: str | Path) -> str:
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    seq = []
    with fasta_path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    if not seq:
        raise ValueError(f"No sequence found in {fasta_path}")
    return "".join(seq)


def _submit(seqs: list[str], mode: str, host_url: str, user_agent: str) -> dict:
    query = ""
    n = 101
    for seq in seqs:
        query += f">{n}\n{seq}\n"
        n += 1

    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified for MSA server. "
            "Set panddamap.mmseqs2_user_agent to identify requests."
        )

    error_count = 0
    while True:
        try:
            res = requests.post(
                f"{host_url}/ticket/msa",
                data={"q": query, "mode": mode},
                timeout=6.02,
                headers=headers,
            )
        except requests.exceptions.Timeout:
            logger.warning("Timeout while submitting to MSA server. Retrying...")
            continue
        except Exception as exc:  # pragma: no cover - network errors
            error_count += 1
            logger.warning(
                "Error while submitting to MSA server (attempt %s/5): %s",
                error_count,
                exc,
            )
            time.sleep(5)
            if error_count > 5:
                raise
            continue
        break

    try:
        return res.json()
    except ValueError:
        logger.error("Server did not return JSON: %s", res.text)
        return {"status": "ERROR"}


def _status(ticket_id: str, host_url: str, user_agent: str) -> dict:
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    error_count = 0
    while True:
        try:
            res = requests.get(
                f"{host_url}/ticket/{ticket_id}",
                timeout=6.02,
                headers=headers,
            )
        except requests.exceptions.Timeout:
            logger.warning("Timeout while fetching MSA status. Retrying...")
            continue
        except Exception as exc:  # pragma: no cover - network errors
            error_count += 1
            logger.warning(
                "Error while fetching MSA status (attempt %s/5): %s",
                error_count,
                exc,
            )
            time.sleep(5)
            if error_count > 5:
                raise
            continue
        break

    try:
        return res.json()
    except ValueError:
        logger.error("Server did not return JSON: %s", res.text)
        return {"status": "ERROR"}


def _download(ticket_id: str, out_path: Path, host_url: str, user_agent: str) -> None:
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    error_count = 0
    while True:
        try:
            res = requests.get(
                f"{host_url}/result/download/{ticket_id}",
                timeout=6.02,
                headers=headers,
            )
        except requests.exceptions.Timeout:
            logger.warning("Timeout while downloading MSA result. Retrying...")
            continue
        except Exception as exc:  # pragma: no cover - network errors
            error_count += 1
            logger.warning(
                "Error while downloading MSA result (attempt %s/5): %s",
                error_count,
                exc,
            )
            time.sleep(5)
            if error_count > 5:
                raise
            continue
        break

    out_path.write_bytes(res.content)


def query_mmseqs2_server(
    fasta_path: str | Path,
    output_dir: str | Path,
    user_agent: str,
    host_url: str = "https://api.colabfold.com",
    use_env: bool = True,
    use_filter: bool = True,
) -> Path:
    """Query ColabFold MMseqs2 server and write .a3m files.

    Returns the alignment directory path containing the extracted .a3m files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq = _read_fasta_sequence(fasta_path)
    seqs = [seq]

    mode = "env" if use_env else "all"
    if not use_filter:
        mode = "env-nofilter" if use_env else "nofilter"

    tar_path = output_dir / "out.tar.gz"
    if not tar_path.exists():
        out = _submit(seqs, mode, host_url, user_agent)
        while out.get("status") in {"UNKNOWN", "RATELIMIT"}:
            sleep_time = 5 + random.randint(0, 5)
            logger.info("MSA server rate-limited; sleeping %ss", sleep_time)
            time.sleep(sleep_time)
            out = _submit(seqs, mode, host_url, user_agent)

        if out.get("status") == "ERROR":
            raise RuntimeError("MMseqs2 API returned ERROR")
        if out.get("status") == "MAINTENANCE":
            raise RuntimeError("MMseqs2 API maintenance; try later")

        ticket_id = out.get("id")
        while out.get("status") in {"UNKNOWN", "RUNNING", "PENDING"}:
            sleep_time = 5 + random.randint(0, 5)
            time.sleep(sleep_time)
            out = _status(ticket_id, host_url, user_agent)

        if out.get("status") != "COMPLETE":
            raise RuntimeError("MMseqs2 API did not complete successfully")

        _download(ticket_id, tar_path, host_url, user_agent)

    with tarfile.open(tar_path) as tar_gz:
        tar_gz.extractall(output_dir)

    # Ensure at least one .a3m exists
    a3m_files = list(output_dir.rglob("*.a3m"))
    if not a3m_files:
        # ColabFold outputs a3m files inside the tar; if none, raise
        raise RuntimeError("MMseqs2 output did not contain .a3m files")

    # Sanitize null bytes in a3m files
    for a3m_file in a3m_files:
        text = a3m_file.read_text()
        if "\x00" in text:
            a3m_file.write_text(text.replace("\x00", ""))

    return output_dir


__all__ = ["query_mmseqs2_server"]
