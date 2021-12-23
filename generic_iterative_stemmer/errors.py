class StemmingTrainerError(Exception):
    pass


class StemDictFileNotFoundError(Exception):
    def __init__(self):
        super().__init__("Stem dict file not found, can't load model.")
