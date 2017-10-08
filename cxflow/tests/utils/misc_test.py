"""
Test module for :py:class:`cxflow.utils.misc.CatchSigint`.
"""
import os
import signal
import threading
import platform

from cxflow.hooks import TrainingTerminated
from cxflow.utils.misc import CatchSigint

from ..test_core import CXTestCase

kill = os.kill

if 'Windows' in platform.system():
    def kill_windows(pid, signum):
        """
        Adapt os.kill for Windows platform.

        Reference: <https://stackoverflow.com/questions/35772001/how-to-handle-the-signal-in-python-on-windows-machine>
        """
        sigmap = {signal.SIGINT: signal.CTRL_C_EVENT, signal.SIGBREAK: signal.CTRL_BREAK_EVENT}
        if signum in sigmap and pid == os.getpid():
            # we don't know if the current process is a
            # process group leader, so just broadcast
            # to all processes attached to this console.
            pid = 0
        thread = threading.current_thread()
        handler = signal.getsignal(signum)
        # work around the synchronization problem when calling
        # kill from the main thread.
        if (signum in sigmap and
            thread.name == 'MainThread' and
            callable(handler) and
            pid == 0):
            event = threading.Event()

            def handler_set_event(signum, frame):
                event.set()
                return handler(signum, frame)
            signal.signal(signum, handler_set_event)
            try:
                os.kill(pid, sigmap[signum])
                # busy wait because we can't block in the main
                # thread, else the signal handler can't execute.
                while not event.is_set():
                    pass
            finally:
                signal.signal(signum, handler)
        else:
            os.kill(pid, sigmap.get(signum, signum))
    kill = kill_windows


class CatchSigintTest(CXTestCase):
    """Test case for ``CatchSigint`` with-resource class."""

    def _sigint_handler(self, *_):
        self._sigint_unhandled = True

    def setUp(self):
        """Register the _sigint_handler."""
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._sigint_unhandled = False

    def test_catching_outside(self):
        """Test ``CatchSigint`` does not handle sigints outside with-resource environment."""
        CatchSigint()
        with CatchSigint():
            pass
        kill(os.getpid(), signal.SIGINT)
        self.assertTrue(self._sigint_unhandled, 'CatchSigint handles SIGINT outside with-resource environment')

    def test_catching_inside(self):
        """Test ``CatchSigint`` does handle sigints inside with-resource environment."""
        with CatchSigint():
            kill(os.getpid(), signal.SIGINT)
        self.assertFalse(self._sigint_unhandled, 'SigintHook does not handle SIGINT while training')
        # double SIGINT cannot be easily tested (it quits(1))

    def test_raising(self):
        """Test ``CatchSigint`` does rise ``TrainingTerminated`` exception."""
        with CatchSigint() as catch:
            kill(os.getpid(), signal.SIGINT)
            with self.assertRaises(TrainingTerminated):
                catch.raise_check_sigint()
