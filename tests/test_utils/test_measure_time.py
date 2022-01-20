from time import sleep

from generic_iterative_stemmer.utils import MeasureTime, configure_logging, measure_time

SLEEP_TIME_SEC = 0.05


def test_measure_time_is_a_context_manager():
    def work():
        sleep(SLEEP_TIME_SEC)

    with MeasureTime() as measure:
        work()

    assert abs(SLEEP_TIME_SEC - measure.duration.total_seconds()) < 0.01


def test_measure_time_is_a_decorator():
    configure_logging()

    @measure_time
    def work():
        sleep(0.05)

    work()
