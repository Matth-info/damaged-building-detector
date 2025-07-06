from __future__ import annotations

import logging
import os
from pathlib import Path
from time import time

import requests
from tqdm import tqdm  # Add tqdm for progress bar

SUCCESSFUL_STATUS = 200


class DownloadFailedError(Exception):
    """Custom exception raised when a file download fails after several attempts."""


def download_file(
    url: str, filename: str | Path, retries: int = 3, *, always_download: bool = False
) -> None:
    """Download a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.
        always_download (bool): If True, downloading is always done

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if Path(filename).is_file() and not always_download:
        logging.info("File '%s' already exists. Skipping download.", filename)
        return

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=10) as response:
                if response.status_code == SUCCESSFUL_STATUS:
                    total = int(response.headers.get("content-length", 0))
                    with (
                        Path.open(filename, "wb") as f,
                        tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=f"Downloading {filename}",
                            ncols=80,
                        ) as pbar,
                    ):
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    logging.info("Download successful on attempt %d", attempt + 1)
                    break
                logging.warning(
                    "Attempt %d failed with status code %d", attempt + 1, response.status_code
                )
        except requests.RequestException as e:
            logging.warning("Attempt %d failed with error: %s", attempt + 1, e)
        time.sleep(2)
    else:
        raise DownloadFailedError("Failed to download the file after several attempts.")
