import os
import sagemaker

from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (ProcessingStep, TrainingStep)
from sagemaker.workflow.properties import PropertyFile

from sagemaker.processing import (ProcessingInput, ProcessingOutput)
from sagemaker.sklearn.processing import (SKLearnProcessor)
from sagemaker.sklearn.estimator import (SKLearn)


class CreditScoreSageMakerPipeline:

    def __init__(self):
        self.session = LocalPipelineSession()
        try:
            self.role = sagemaker.get_execution_role()
        except:
            self.role = "LabRole"

        self.instance_type = "local"
        self.base_path = "/home/ec2-user/SageMaker/UAS"
        self.local_output_path = "file:///home/ec2-user/SageMaker/UAS"

        for folder in ["ingested", "train", "test", "eval"]:
            os.makedirs(f"{self.base_path}/{folder}", exist_ok=True)

    def create_processor(self):
        return SKLearnProcessor(framework_version="1.2-1", role=self.role, instance_type=self.instance_type, instance_count=1, sagemaker_session=self.session)

    def create_estimator(self):
        return SKLearn(entry_point="train.py", role=self.role, instance_type=self.instance_type, framework_version="1.2-1", sagemaker_session=self.session)

    def build_pipeline(self):
        processor = self.create_processor()
        estimator = self.create_estimator()

        step_ingest = ProcessingStep(
            name="CreditScoreIngest",
            processor=processor,
            inputs=[ProcessingInput(source=self.base_path, destination="/opt/ml/processing/input")],
            outputs=[ProcessingOutput(output_name="ingested_data", source="/opt/ml/processing/ingested", destination=f"{self.local_output_path}/ingested")],
            code="data_ingestion.py"
        )

        step_preprocess = ProcessingStep(
            name="CreditScorePreprocess",
            processor=processor,
            inputs=[ProcessingInput(source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested_data"].S3Output.S3Uri, destination="/opt/ml/processing/ingested")],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"{self.local_output_path}/train"),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"{self.local_output_path}/test")
            ],
            code="preprocessing.py"
        )

        step_train = TrainingStep(
            name="CreditScoreTrain",
            estimator=estimator,
            inputs={"train": f"{self.local_output_path}/train"}
        )

        evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

        step_eval = ProcessingStep(
            name="CreditScoreEvaluate",
            processor=processor,
            inputs=[
                ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
                ProcessingInput(source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
            ],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=f"{self.local_output_path}/eval")],
            code="evaluation.py",
            property_files=[evaluation_report]
        )

        pipeline = Pipeline(name="CreditScoreDeploymentWorkflow", steps=[step_ingest, step_preprocess, step_train, step_eval], sagemaker_session=self.session)
        return pipeline

    def execute(self):
        print("Building Pipeline...")
        pipeline = self.build_pipeline()
        print("Registering Pipeline...")
        pipeline.upsert(role_arn=self.role)
        print("Starting Pipeline...")
        execution = pipeline.start()
        print("Pipeline started successfully.")
        return execution


if __name__ == "__main__":
    CreditScoreSageMakerPipeline().execute()