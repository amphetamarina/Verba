import json
import os
import boto3

class BedrockClient:
    def __init__(self):
        self.bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=os.environ.get('BEDROCK_AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.environ.get('BEDROCK_AWS_ACCESS_KEY', ''),
            aws_secret_access_key=os.environ.get('BEDROCK_AWS_SECRET_KEY', ''),
        )

    def invoke_model(self, body, modelId, accept, contentType):
        """"
        Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
        """
        response = self.bedrock_runtime_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion')

    async def invoke_model_with_response_stream(self, modelId, body):
        response = self.bedrock_runtime_client.invoke_model_with_response_stream(
            modelId=modelId,
            body=body
        )

        stream = response.get('body')
        if stream:
            async for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    yield json.loads(chunk.get('bytes').decode())
