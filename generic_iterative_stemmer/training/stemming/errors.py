class StemmingTrainerError(Exception):
    pass


class MissingIterationFolderError(Exception):
    def __init__(self):
        super().__init__("Iteration folder was missing.")
