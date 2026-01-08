"""
LendingFlowStack - AWS CDK Stack for Document Processing Workflow

This stack sets up an end-to-end document processing pipeline for lending workflows using AWS services:

Architecture:
  1. S3 Bucket: Stores input documents (documents/) and sample files (samples/)
  2. EventBridge: Monitors S3 for new file uploads on specific prefixes
  3. Lambda Function: Triggers AWS Bedrock Data Automation for document processing
  4. Output: Processed results stored in dedicated output folders

Data Flow:
  1. User uploads document to s3://bucket/documents/
  2. S3 emits event → EventBridge receives it
  3. EventBridge routes to Lambda based on prefix pattern matching
  4. Lambda extracts file location from S3 event details (bucket, key)
  5. Lambda constructs input/output S3 URIs (replaces "documents" with "documents-output")
  6. Lambda retrieves project ARN using DATA_PROJECT_NAME from context
  7. Lambda calls BDA's invoke_data_automation_async() API with:
     - Input S3 location (document to process)
     - Output S3 location (where results will be written)
     - Project ARN (specifies extraction workflow)
     - EventBridge notification enabled (BDA notifies on completion)
  8. BDA processes document asynchronously (extracts data, applies AI models)
  9. Results written to s3://bucket/documents-output/
  10. EventBridge notification sent on completion

Key Design Decisions:
  - EventBridge for flexible routing and filtering (vs direct S3 notifications)
  - Pre-created placeholder files ensure folder structure is visible in S3 console
  - IAM policies grant specific Bedrock permissions (principle of least privilege)
  - Lambda layer centralizes dependencies for reusability
  - Environment variables enable runtime configuration without code changes
"""

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_lambda as _lambda,
    Duration,
    aws_events as events,
    aws_s3_deployment as s3_deployment,
    aws_events_targets as targets,
    RemovalPolicy
)
from constructs import Construct
from typing import Optional


class LendingFlowStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        data_project_name = self.node.try_get_context("data_project_name")
        bda_runtime_endpoint = self.node.try_get_context("bda_runtime_endpoint")

        # Create S3 bucket with security and operational best practices
        # Security configs: encryption, SSL enforcement, block public access
        # Operational configs: auto-delete, destroy on stack deletion (demo-safe)
        bucket = s3.Bucket(
            self,
            "bucket",
            auto_delete_objects=True,  # Automatically delete contents on stack deletion (demo purposes - NOT for production)
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # Prevent accidental public exposure
            encryption=s3.BucketEncryption.S3_MANAGED,  # Enable S3 default encryption (AES-256)
            enforce_ssl=True,  # Only allow HTTPS access (in-transit encryption)
            removal_policy=RemovalPolicy.DESTROY,  # Delete bucket when stack is destroyed (demo purposes - NOT for production)
        )

        # Enable EventBridge notifications for the S3 bucket
        # Why EventBridge instead of direct S3→Lambda invocation?
        # 1. Advanced filtering: EventBridge allows filtering by object key prefix in the event pattern,
        #    enabling precise routing of events based on object location (e.g., documents/, samples/)
        # 2. Decoupling: S3 doesn't directly invoke Lambda; EventBridge acts as an intermediary,
        #    allowing future extensibility (add SNS, SQS, HTTP endpoints without changing S3 config)
        # 3. Multiple targets: One S3 event can route to multiple targets through EventBridge rules
        # 4. Better error handling: EventBridge provides Dead Letter Queues, retries, and exponential backoff
        # 5. Event transformation: EventBridge can enrich or transform events before delivery
        bucket.enable_event_bridge_notification()

        # Pre-create folders by uploading empty ".placeholder" files
        # S3 has no true folders, only key prefixes. Placeholder files ensure prefixes are visible
        # in S3 console and provide clear directory structure for users before data is processed.
        # Prefixes:
        #   - documents/: Input folder for documents to process
        #   - documents-output/: Output folder for processed documents
        #   - samples/: Optional sample documents for testing
        #   - samples-output/: Output folder for processed samples
        prefixes = ['samples/', 'samples-output/', 'documents/', 'documents-output/']
        for prefix in prefixes:
            s3_deployment.BucketDeployment(
                self,
                f"Deploy{prefix.replace('/', '')}",
                sources=[s3_deployment.Source.data(f"{prefix.replace('/', '')}.placeholder", "")],
                destination_bucket=bucket,
                destination_key_prefix=prefix
            )

        # Lambda function to trigger bedrock data insight
        # This function will be invoked by EventBridge when documents are uploaded
        invoke_data_automation_lambda_function = self.create_invoke_data_automation_function(
            bucket.bucket_name,
            **({'bda_runtime_endpoint': bda_runtime_endpoint} if bda_runtime_endpoint is not None else {}),
            **({'data_project_name': data_project_name} if data_project_name is not None else {})
        )

        # Grant permissions
        # Allows Lambda to read input documents and write processed results
        bucket.grant_read_write(invoke_data_automation_lambda_function)

        # EventBridge rules for specific prefixes
        # This pattern-based routing enables prefix-level filtering that would be cumbersome with direct S3 notifications
        def create_event_rule(id: str, prefix: str, target: _lambda.IFunction):
            rule = events.Rule(
                self,
                id,
                event_pattern=events.EventPattern(
                    source=["aws.s3"],
                    detail_type=["Object Created"],
                    detail={
                        "bucket": {"name": [bucket.bucket_name]},
                        "object": {"key": [{"prefix": prefix}]}
                    },
                )
            )
            rule.add_target(targets.LambdaFunction(target))

        # Create rule for documents
        create_event_rule("DocumentsRule", "documents/", invoke_data_automation_lambda_function)

        # Define an output for the bucket name
        CfnOutput(self, "lending-flow-bucket", value=bucket.bucket_name)

    def create_invoke_data_automation_function(self,
            target_bucket_name: s3.Bucket,
            data_project_name: Optional[str] = None,
            bda_runtime_endpoint: Optional[str] = None
    ):
        """
        Creates a Lambda function that processes documents using AWS Bedrock Data Automation.
        
        Args:
            target_bucket_name: S3 bucket for reading input and writing output
            data_project_name: Optional Bedrock Data Automation project name from context
            bda_runtime_endpoint: Optional custom Bedrock Data Automation endpoint
        
        Returns:
            The configured Lambda function with appropriate IAM permissions
        """
        # Create layer
        # Lambda layer centralizes shared dependencies (boto3, custom utilities)
        # allowing reuse across multiple Lambda functions without code duplication
        layer = _lambda.LayerVersion(
            self,
            'invoke_data_automation_lambda_layer',
            description='Dependencies for the document automation lambda function',
            code=_lambda.Code.from_asset('lambda/lending_flow/layer/'),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_10],
        )

        # Create Lambda function
        # Runtime: Python 3.10 | Timeout: 300s (5 minutes for large document processing)
        # Triggered by: EventBridge rule on documents/ prefix
        # Output: Processed results written to documents-output/ prefix
        lending_document_automation_lambda_function = _lambda.Function(
            self,
            'invoke_data_automation',
            runtime=_lambda.Runtime.PYTHON_3_10,
            code=_lambda.Code.from_asset('lambda/lending_flow/documents_processor'),
            handler='index.lambda_handler',
            timeout=Duration.seconds(300),
            layers=[layer],
            environment={
                k: v
                for k, v in {
                    'TARGET_BUCKET_NAME': target_bucket_name,  # S3 bucket for I/O operations
                    'BDA_RUNTIME_ENDPOINT': bda_runtime_endpoint,  # Bedrock Data Automation endpoint (optional)
                    'DATA_PROJECT_NAME': data_project_name,  # Bedrock Data Automation project name (optional)
                }.items()
                if v is not None  # Only set env vars if provided (avoid None values)
            }
        )

        # Grant IAM permissions to Lambda
        # bedrock:InvokeDataAutomationAsync: Invoke Bedrock Data Automation asynchronously
        # bedrock:List*: List available Bedrock Data Automation projects and resources
        # These permissions follow principle of least privilege (only what's needed)
        lending_document_automation_lambda_function.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeDataAutomationAsync"],
            resources=["*"]
        ))
        lending_document_automation_lambda_function.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:List*"],
            resources=["*"]
        ))

        return lending_document_automation_lambda_function