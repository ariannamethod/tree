import pathlib
import threading
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import branches  # noqa: E402


def _count_workers() -> int:
    return len(
        [
            t
            for t in threading.enumerate()
            if t.name.startswith("branches-worker")
        ]
    )


def test_mass_learn_does_not_spawn_extra_threads() -> None:
    # Initial number of worker threads should match the configured limit.
    assert _count_workers() == branches.WORKER_LIMIT

    # Flood the system with learn requests.
    for _ in range(200):
        branches.learn(["alpha", "beta"], "ctx")

    # Wait for all tasks to be processed before checking the count again.
    branches.wait()

    assert _count_workers() == branches.WORKER_LIMIT
