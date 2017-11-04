from typing import Callable, Optional, Iterator
from threading import Thread, Event
from queue import Queue, Empty

from ..types import Batch, Stream, TimeProfile
from ..utils.profile import Timer


class StreamWrapper:
    """
    Dataset stream wrapper which manages buffering, epoch cutting etc.

    The main features are:
        - resets underlying dataset stream after the iteration reaches its end
        - if specified, uses consumer-producer buffer for batches allowing simultaneous batch producing and training
        - if specified, produces epochs of fixed size
        - logs the timings to the given profile

    .. caution::
        Buffered ``StreamWrapper`` must be used in with-resource environment
        so that the enqueueing thread can be properly managed.

    .. code-block:: python
        :caption: non-buffered StreamWrapper

        stream = StreamWrapper(dataset.train_stream, 'train')
        for batch in stream:  # 1st batch
            # do stuff
        for batch in stream:  # 2nd batch
            # do stuff


    .. code-block:: python
        :caption: buffered StreamWrapper with fixed size epochs

        stream = StreamWrapper(dataset.train_stream, 'train', buffer=16, epoch_size=1000)
        with stream:  # we would get error without with-resource directive
            for batch in stream:  # 1st batch
                # do stuff
    """

    def __init__(self,
                 stream_fn: Callable[[], Stream],
                 buffer_size: int=0,
                 epoch_size: int=-1,
                 name: Optional[str]=None,
                 profile: Optional[TimeProfile]=None):
        """
        Create new StreamWrapper.

        :param stream_fn: callable which returns raw dataset stream
        :param buffer_size: buffer size, < 1 means no buffer
        :param epoch_size: if > 0, stop iteration after the specified number of batches
        :param name: optional stream name
        :param profile: profile to record times
        """
        self._get_stream_fn = stream_fn
        self._name = name
        self._buffer_size = buffer_size
        self._epoch_size = epoch_size
        self._profile = profile
        self._batch_count = 0
        self._stream = None
        self._queue = Queue(buffer_size) if buffer_size > 0 else None
        self._stopping_event = None
        self._enqueueing_thread = None

    @property
    def name(self) -> Optional[str]:
        """Stream name."""
        return self._name

    def _get_stream(self) -> Iterator:
        """Possibly create and return raw dataset stream iterator."""
        if self._stream is None:
            self._stream = iter(self._get_stream_fn())
        return self._stream

    def _epoch_limit_reached(self) -> bool:
        """
        Returns True if the number of produced batches reached the specified ``epoch_size``.

        Always return False if no limit was specified.
        """
        return 0 < self._epoch_size <= self._batch_count

    def _enqueue_batches(self, stop_event: Event) -> None:
        """
        Enqueue all the stream batches. If specified, stop after ``epoch_size`` batches.

        .. note::
            Signal the epoch end with ``None``.

        Stop when:
        - ``stop_event`` is risen
        - stream ends and epoch size is not set
        - specified number of batches is enqueued

        .. note::
            This is used only with ``buffer`` > 0.

        :param stop_event: event signaling stop instruction
        """
        while True:
            for batch in self._get_stream():
                self._queue.put(batch)
                self._batch_count += 1
                if stop_event.is_set():
                    return
                if self._epoch_limit_reached():
                    self._queue.put(None)
                    self._batch_count = 0
                    return
            self._stream = None  # yield a new iterator next time
            if self._epoch_size <= 0:  # for non-fixed size epochs
                self._queue.put(None)
                self._batch_count = 0
                return

    def _dequeue_batch(self) -> Optional[Batch]:
        """
        Return a single batch from queue or ``None`` signaling epoch end.

        :raise ChildProcessError: if the enqueueing thread ended unexpectedly
        """
        if self._enqueueing_thread is None:
            raise ValueError('StreamWrapper `{}` with buffer of size `{}` was used outside with-resource environment.'
                             .format(self._name, self._buffer_size))
        if not self._enqueueing_thread.is_alive() and self._queue.empty():
            self._start_thread()
        while True:
            try:
                batch = self._queue.get(timeout=2)
                self._queue.task_done()
                break
            except Empty:
                if not self._enqueueing_thread.is_alive():
                    try:
                        # the enqueueing thread may just finished properly so lets check the queue eagerly
                        batch = self._queue.get_nowait()
                        self._queue.task_done()
                        break
                    except Empty:
                        # so we failed to retrieve a batch and the enqueueing thread is dead
                        # there is no hope, something must went wrong
                        raise ChildProcessError('Enqueueing thread ended unexpectedly.')
        return batch

    def _next_batch(self) -> Optional[Batch]:
        """
        Return a single batch or ``None`` signaling epoch end.

        .. note::
            Signal the epoch end with ``None``.

        Stop when:
        - stream ends and epoch size is not set
        - specified number of batches is returned

        :return: a single batch or ``None`` signaling epoch end
        """
        if self._epoch_limit_reached():
            self._batch_count = 0
            return None
        try:
            batch = next(self._get_stream())
            self._batch_count += 1
            return batch
        except StopIteration:
            self._stream = None  # yield a new iterator next time
            if self._epoch_size > 0:  # underlying stream ended but our fixed size epoch did not
                batch = next(self._get_stream())  # get another stream and return its 1st batch
                self._batch_count += 1
                return batch
            else:
                self._batch_count = 0
                return None

    def _start_thread(self):
        """Start an enqueueing thread."""
        self._stopping_event = Event()
        self._enqueueing_thread = Thread(target=self._enqueue_batches, args=(self._stopping_event,))
        self._enqueueing_thread.start()

    def _stop_thread(self):
        """Stop the enqueueing thread. Keep the queue content and stream state."""
        self._stopping_event.set()
        queue_content = []
        try:  # give the enqueueing thread chance to put a batch to the queue and check the stopping event
            while True:
                queue_content.append(self._queue.get_nowait())
        except Empty:
            pass
        self._enqueueing_thread.join()
        try:
            queue_content.append(self._queue.get_nowait())  # collect the very last item
        except Empty:
            pass
        self._queue = Queue(max(len(queue_content), self._buffer_size))  # queue content may be bigger than queue size
        for batch in queue_content:
            self._queue.put(batch)

    def __enter__(self) -> Iterator[Batch]:
        """If buffered, start the enqueueing thread."""
        if self._buffer_size > 0:
            self._start_thread()
        return self

    def __exit__(self, *args) -> None:
        """If buffered, terminate the enqueueing thrad."""
        if self._buffer_size > 0:
            self._stop_thread()

    def __iter__(self) -> Iterator[Batch]:
        """Get stream iterator."""
        return self

    def __next__(self) -> Batch:
        """
        Return next batch or end epoch with ``StopIteration``.

        :return: next batch
        :raises StopIteration: at the end of the epoch
        """
        get_batch_fn = self._dequeue_batch if self._buffer_size > 0 else self._next_batch
        event_name = 'read_batch_{}'.format(self._name)

        if self._profile is not None:
            with Timer(event_name, self._profile):
                batch = get_batch_fn()
        else:
            batch = get_batch_fn()

        if batch is None:
            if self._profile:
                self._profile[event_name].pop()
            raise StopIteration
        return batch
