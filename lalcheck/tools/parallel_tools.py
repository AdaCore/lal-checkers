from multiprocessing import Process, Manager, queues
import os
import time


# The environment variables that are tried by the tempfile package to decide
# where to create its temporary files/directories.
_tmpdir_env_vars = ['TMPDIR', 'TEMP', 'TMP']


def _remove_tmpdir_env_vars():
    """
    Removes from the environment the variables designating directories which
    the tempfile package uses to decide where to store its temporary files and
    directories
    """
    for env_var in _tmpdir_env_vars:
        if env_var in os.environ:
            del os.environ[env_var]


_keep_alive_fun = None


def keepalive(seconds, timeout_cause=None):
    """
    Function to call to communicate to the process driver that this thread has
    not yet timed out and must stay alive for at least the given number of
    seconds. Moreover, a cause can be given such that if this process times
    out, the driver can react appropriately (e.g. retrying without the element
    that caused the timeout) using the provided 'readjust' function.

    :param float seconds: The minimal number of seconds that this process
        should stay alive before timing out. A negative amount indicates that
        this process should never timeout.

    :param object timeout_cause: An object representing the cause of the
        timeout.
    """
    if _keep_alive_fun is not None:
        _keep_alive_fun(seconds, timeout_cause)


def _target_proxy(target, arg, result_queue, timeout_queue, timeout_factor):
    """
    A wrapper of the actual target. Takes care of placing its return value
    in the result_queue and setting the keepalive function for this process.

    :param (object)->object target: The actual target function.
    :param object arg: The argument to the function.
    :param queues.Queue result_queue: The queue containing the results of each
        worker.
    :param queues.Queue timeout_queue: The queue that this worker uses to
        communicate its timeout barriers.
    :param float timeout_factor:
    """
    def keepalive_fun(x, timeout_cause=None):
        timeout_queue.put((timeout_factor * x, timeout_cause))

    global _keep_alive_fun
    _keep_alive_fun = keepalive_fun

    result_queue.put(target(arg))


class TimedProcess(object):
    """
    Wraps a process that can timeout.

    The process is supposed to use the timeout_queue to communicate timeout
    information (delta and cause). For example, a process that places the
    element (2, 'some_cause') in the queue indicates that this process should
    timeout in 2 seconds and that if it does, 'some_cause' is the cause of the
    timeout.

    Previous timeout information is ignored when a new element is placed in the
    queue (unless it's too late).

    Typically, the processes will not use the timeout_queue object directly,
    but will call the keepalive function. (see keepalive).
    """
    def __init__(self, timeout_queue, element, process):
        """
        :param queues.Queue timeout_queue: The queue used to communicate
            timeout information.
        :param object element: The element that this process takes care of
            transforming.
        :param Process process: The process that is wrapped.
        """
        self.timeout_queue = timeout_queue
        self.element = element
        self.process = process
        self.last_update = time.time()
        self.timeout_info = (2, None)

    def start(self):
        """
        Starts this process.
        """
        self.process.start()

    def terminate(self):
        """
        Terminates this process. (sends it the SIGTERM signal).
        """
        self.process.terminate()

    def is_alive(self):
        """
        Returns True iff this process is still alive (i.e. is in state
        'started')

        :rtype: bool
        """
        return self.process.is_alive()

    def check_timeout_and_get_cause(self):
        """
        Returns a pair which first element is True iff this process has timed
        out according to the timeout information it provided. If it is the
        case, the second element represents the cause of the timeout.

        :rtype: (bool, object)
        """
        self._read_timeout_info()

        timeout_delta, timeout_cause = self.timeout_info

        if timeout_delta >= 0:
            if time.time() - self.last_update > timeout_delta:
                self.timeout_info = (-1, None)
                return True, timeout_cause

        return False, None

    def _read_timeout_info(self):
        """
        Reads timeout information from the timeout_queue.
        """
        try:
            while True:
                self.timeout_info = self.timeout_queue.get_nowait()
                self.last_update = time.time()
        except queues.Empty:
            pass


def _default_readjust(elem, cause):
    return None


def parallel_map(process_count, target, elements, timeout_factor=1.0,
                 timeout_callback=None, readjust=_default_readjust):
    """
    Performs a map across several processes.

    :param int process_count: The maximal number of processes to use
        simultaneously.
    :param (object)->object target: The mapping function.
    :param iterable[object] elements: The elements to map.
    :param float timeout_factor: The factor to multiply processes' timeout
        deltas with.
    :param (object)->None | None timeout_callback: The function to call
        if a process times out. It is called with the cause of the timeout.
    :param (object, object)->object|None readjust: The readjusting function,
        used to refine an element of the collection to map when its
        transformation failed. Once refined, the transformation is retried.
        This function can return None to indicate that no further tries should
        be attempted.
    :rtype: list[object]
    """

    # The multiprocessing package creates multiple temporary files/directories
    # using the tempfile python package. However, this package uses environment
    # variables such as TMPDIR, TEMP and TMP to decide where to create its
    # temporary files. A consequence of that is that if such a variable is in
    # the environment and designates a directory which path is too long, some
    # procedures in the multiprocessing package will fail, such as binding an
    # AF_UNIX socket to a that path, since its max length is 108 characters.
    _remove_tmpdir_env_vars()

    m = Manager()

    processes = []

    elements = list(elements)
    result_queue = m.Queue()

    def refill_workers():
        for _ in range(process_count - len(processes)):
            if len(elements) > 0:
                elem = elements.pop(0)
                timeout_queue = m.Queue()
                new_process = TimedProcess(timeout_queue, elem, Process(
                    target=_target_proxy,
                    args=(target, elem, result_queue, timeout_queue,
                          timeout_factor)
                ))
                processes.append(new_process)
                new_process.start()

    def handle_timedout_processes():
        for proc in processes:
            has_timed_out, timeout_cause = proc.check_timeout_and_get_cause()
            if False:
                proc.terminate()
                if timeout_callback:
                    timeout_callback(timeout_cause)

                readjusted_element = readjust(proc.element, timeout_cause)
                if readjusted_element is not None:
                    elements.insert(0, readjusted_element)

    refill_workers()

    while len(processes) > 0:
        time.sleep(1)
        handle_timedout_processes()
        processes = [p for p in processes if p.is_alive()]
        refill_workers()

    results = []
    try:
        while True:
            results.append(result_queue.get_nowait())
    except queues.Empty:
        pass

    try:
        m.shutdown()
    except OSError:
        pass  # Deliberately ignore this error, assume OS will kill it anyway.

    return results
