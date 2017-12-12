"""
This module contains :py:class:`cxflow.datasets.DownloadableDataset` which can be used as a base class for various
downloadable datasets.
"""

from abc import abstractmethod, ABCMeta
from typing import Iterable

from .base_dataset import BaseDataset
from ..utils import maybe_download_and_extract


class DownloadableDataset(BaseDataset, metaclass=ABCMeta):
    """
    DownloadableDataset is dataset base class implementing routines for downloading and extracting data via
    ``cxflow dataset download`` command.

    The typical use-case is that ``data_root``, ``url_root`` and ``download_filenames`` variables are passed to the
    dataset constructor.
    Alternatively, these properties might be directly implemented in their corresponding methods.
    """

    def _configure_dataset(self, data_root: str=None, download_urls: Iterable[str]=None, **kwargs) -> None:
        """
        Save the passed values and use them as a default property implementation.

        :param data_root: directory to which the files will be downloaded
        :param download_urls: list of URLs to be downloaded
        """
        self._data_root = data_root
        self._download_urls = download_urls

    @property
    def data_root(self) -> str:
        """Path to the data root directory."""
        if self._data_root is None:
            raise ValueError('`data_root` is not specified.')
        return self._data_root

    @property
    def download_urls(self) -> Iterable[str]:
        """A list of URLs to be downloaded."""
        if self._download_urls is None:
            raise ValueError('`download_urls` is not specified.')
        return self._download_urls

    def download(self) -> None:
        """
        Maybe download and extract the extra files required.

        If not already downloaded, download all files specified by :py:meth:`download_urls`. Then, extract
        the downloaded files to :py:meth:`data_root`.

        .. code-block:: bash
            :caption: cxflow CLI example

            cxflow dataset download <path-to-config>

        """
        for url in self.download_urls:
            maybe_download_and_extract(self.data_root, url)
