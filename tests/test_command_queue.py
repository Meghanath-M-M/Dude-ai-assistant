from nova_agent.core.command_queue import CommandQueue


def test_command_queue_runs_in_order():
    queue = CommandQueue()
    seen = []

    def worker(value):
        seen.append(value)

    queue.enqueue(worker, "first")
    queue.enqueue(worker, "second")
    queue.flush()

    assert seen == ["first", "second"]


def test_command_queue_ignores_empty_queue_flush():
    queue = CommandQueue()
    assert queue.flush() == []
