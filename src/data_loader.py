import json


class DataLoader:
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load_file()

    def _load_file(self):
        with open(self.filepath, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def get(self, idx):
        """
        Returns a single scenario by index
        """
        return self.data[idx]

    def __iter__(self):
        """
        Allows iteration over all scenarios
        """
        for example in self.data:
            yield example


# Convenience function
def load_dataset(filepath):
    return DataLoader(filepath)