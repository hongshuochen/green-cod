class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()  # Ensure the output is written immediately
            except ValueError:
                pass  # Ignore if the file is already closed

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass  # Ignore if the file is already closed