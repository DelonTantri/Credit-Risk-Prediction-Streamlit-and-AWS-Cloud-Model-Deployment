from data_ingestion import DataIngestion
from preprocessing import CreditScorePreprocessor
from train import CreditScoreTrainer
from evaluation import CreditScoreEvaluator

class CreditScorePipeline:

    def execute(self, csv_path):
        print("DATA INGESTION\n")

        ingested_file = DataIngestion(csv_path).run()
        print(f"Dataset saved to : {ingested_file}\n")

        print("PREPROCESSING\n")
        CreditScorePreprocessor().process(ingested_file)

        print("MODEL TRAINING\n")
        CreditScoreTrainer().run()

        print("MODEL EVALUATION\n")
        metrics = CreditScoreEvaluator().run()

        print("PIPELINE COMPLETED\n")

        print(f"Accuracy  : {metrics['accuracy']:.4f}\nPrecision : {metrics['precision']:.4f}\nRecall    : {metrics['recall']:.4f}\nF1 Score  : {metrics['f1_score']:.4f}")
        print("\nAll artifacts generated successfully.")

        return metrics

if __name__ == "__main__":
    pipeline = CreditScorePipeline()
    pipeline.execute("data_A.csv")