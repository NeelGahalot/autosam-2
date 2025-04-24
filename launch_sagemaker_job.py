import boto3
import time
from datetime import datetime

# SageMaker client
sm_client = boto3.client('sagemaker')

# Configuration
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"autosam2-training-crf-sam-dino-annotations-{timestamp}"

# Replace with your values
account_id = ""
region = "eu-central-1"
role_arn = ""
image_uri = ""
read_bucket_name = "treetracker-training-images"
write_bucket_name = 'sagemaker-segmentation-neel'
data_prefix = "crf_sam_annotations_large"
output_prefix = "autosam2_experiments/crf_sam_annotations_large"

# Create training job
response = sm_client.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        'TrainingImage': image_uri,
        'TrainingInputMode': 'File',
    },
    RoleArn=role_arn,
    InputDataConfig=[
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f"s3://{read_bucket_name}/{data_prefix}/",
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'application/x-directory',
            'CompressionType': 'None'
        }
    ],
    OutputDataConfig={
        'S3OutputPath': f"s3://{write_bucket_name}/{output_prefix}/"
    },
    ResourceConfig={
        'InstanceType': 'ml.g5.2xlarge',  # GPU instance for training
        'InstanceCount': 1,
        'VolumeSizeInGB': 30
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 86400  # 24 hours max runtime
    },
    CheckpointConfig={
    'S3Uri': f's3://{write_bucket_name}/autoSAM2_checkpoints/',
    'LocalPath': '/opt/ml/checkpoints'
}

)

print(f"Training job '{job_name}' created.")
print("Waiting for training job to complete...")

# Wait for the training job to complete
while True:
    status = sm_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(f"Job status: {status}")
    
    if status in ['Completed', 'Failed', 'Stopped']:
        break
    
    time.sleep(60)  # Check every minute

print(f"Job {job_name} finished with status: {status}")