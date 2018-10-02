from multiprocessing import Process, Manager, queues
import time


def _target_proxy(target, arg, result_queue):
    """
    A wrapper of the actual target. Takes care of placing its return value
    in the result_queue.

    :param (object)->object target: The actual target function.
    :param object arg: The argument to the function.
    :param queues.Queue result_queue: The queue containing the results of each
        worker.
    """
    result_queue.put(target(arg))


def parallel_map(process_count, target, elements):
    """
    Performs a map across several processes.

    :param int process_count: The maximal number of processes to use
        simultaneously.
    :param (object)->object target: The mapping function.
    :param iterable[object] elements: The elements to map.
    :rtype: list[object]
    """
    m = Manager()

    processes = []

    elements = list(elements)
    result_queue = m.Queue()

    def refill_workers():
        for _ in range(process_count - len(processes)):
            if len(elements) > 0:
                elem = elements.pop(0)
                new_process = Process(
                    target=_target_proxy,
                    args=(target, elem, result_queue)
                )
                processes.append(new_process)
                new_process.start()

    refill_workers()

    while len(processes) > 0:
        time.sleep(1)
        processes = [p for p in processes if p.is_alive()]
        refill_workers()

    results = []
    try:
        while True:
            results.append(result_queue.get_nowait())
    except queues.Empty:
        pass

    return results
