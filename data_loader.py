import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            print(f"File loaded successfully: {self.file_path}")
            self.data = df
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading file '{self.file_path}': {e}")
