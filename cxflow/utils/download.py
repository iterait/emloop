import click
import logging
import os
import requests
from typing import Iterable
import shutil


def maybe_download_and_extract(data_root: str, url_root: str, filenames: Iterable[str]) -> None:
    """
    Maybe download the specified files from ``url_root`` to ``download_root`` and
        - unzip any ``.zip`` archives
        - extract any ``.tar.*`` archives based on :py:func:`tarfile.open`
    :param data_root: data root to download the files to
    :param url_root: root url
    :param filenames: list of filenames to download
    """

    os.makedirs(data_root, exist_ok=True)

    for filename in filenames:
        # check whether the archive already exists
        filepath = os.path.join(data_root, filename)
        if os.path.exists(filepath):
            logging.info('\t`%s` already exists; skipping', filepath)
            continue

        # download with progressbar
        logging.info('\tdownloading %s', filepath)
        if not url_root.endswith('/'):
            url_root += '/'

        req = requests.get(url_root + filename, stream=True)
        expected_size = int(req.headers.get('content-length'))
        chunk_size = 1024
        with open(filepath, 'wb') as f_out,\
                click.progressbar(req.iter_content(chunk_size=chunk_size), length=expected_size/chunk_size) as bar:
            for chunk in bar:
                if chunk:
                    f_out.write(chunk)
                    f_out.flush()

        # extract
        try:
            shutil.unpack_archive(filepath, data_root)
        except (shutil.ReadError, ValueError):
            logging.info('File `%s` could not be extracted by `shutil.unpack_archive`. Please process it manually.',
                         filepath)
