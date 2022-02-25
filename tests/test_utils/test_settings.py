import os

from generic_iterative_stemmer.utils import Settings


def test_settings_reload():
    assert not Settings.is_debug

    os.environ["DEBUG"] = "true"
    assert not Settings.is_debug

    Settings.reload()
    assert Settings.is_debug
