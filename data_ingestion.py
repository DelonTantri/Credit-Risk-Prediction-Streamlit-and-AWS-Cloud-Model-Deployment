from pathlib import Path
import pandas as pd


class DataIngestion:

    def __init__(self, input_path, output_dir="ingested"):
        self.input_path = Path(input_path)
        self.output_dir = Path(__file__).parent / output_dir

    def run(self):

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input file tidak ditemukan:\n{self.input_path}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.output_dir / self.input_path.name

        df = pd.read_csv(self.input_path)
        df.to_csv(output_file, index=False)

        print("DATA INGESTION")
        print(f"Input File : {self.input_path.resolve()}")
        print(f"Output Dir : {self.output_dir.resolve()}")
        print(f"Saved File : {output_file.resolve()}")

        return output_file


if __name__ == "__main__":

    ingestion = DataIngestion("data_A.csv")

    output = ingestion.run()

    print("\nFinished")
    print(output)