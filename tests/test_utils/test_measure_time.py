from time import sleep

from generic_iterative_stemmer.helpers import MeasureTime

SLEEP_TIME_SEC = 0.05


def test_measure_time_is_a_context_manager():
    def work():
        sleep(SLEEP_TIME_SEC)

    with MeasureTime() as measure:
        work()

    assert abs(SLEEP_TIME_SEC - measure.duration.total_seconds()) < 0.01
