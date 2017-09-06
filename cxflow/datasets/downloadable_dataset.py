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

    def _configure_dataset(self, data_root: str=None, url_root: str=None, download_filenames: Iterable[str]=None,
                           **kwargs) -> None:
        """
        Save the passed values and use them as a default property implementation.

        :param data_root: directory to which the files will be downloaded
        :param url_root: URL from where the files are downloaded
        :param download_filenames: list of files to be downloaded
        """
        self._data_root = data_root
        self._url_root = url_root
        self._download_filenames = download_filenames

    @property
    @abstractmethod
    def data_root(self) -> str:
        """Path to the data root directory."""
        if self._data_root is None:
            raise ValueError('`data_root` is not specified.')
        return self._data_root

    @property
    @abstractmethod
    def url_root(self) -> str:
        """URL representing the data root from which the files are downloaded."""
        if self._url_root is None:
            raise ValueError('`url_root` is not specified.')
        return self._url_root

    @property
    @abstractmethod
    def download_filenames(self) -> Iterable[str]:
        """A list of filenames to be downloaded from :py:meth:`url_root`."""
        if self._download_filenames is None:
            raise ValueError('`download_filenames` is not specified.')
        return self.download_filenames

    def download(self) -> None:
        """Maybe download and extra files required for training.

        If not already downloaded, download all files specified by :py:meth:`download_filenames` which are supposed to
        be located in URL specified by :py:meth:`url_root`. Then, extract the downloaded files to :py:meth:`data_root`.

        .. code-block:: bash
            :caption: cxflow CLI example

            cxflow dataset download <path-to-config>

        """
        maybe_download_and_extract(self.data_root, self.url_root, self.download_filenames)
