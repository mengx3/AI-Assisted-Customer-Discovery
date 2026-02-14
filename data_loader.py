import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_csv(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("File loaded successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")
        return self.data
