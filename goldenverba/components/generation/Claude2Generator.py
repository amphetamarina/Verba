import asyncio
import json
from collections.abc import Iterator
from goldenverba.components.generation.interface import Generator
from goldenverba.utils.bedrock_client import BedrockClient

class Claude2Generator(Generator):
    """
    Bedrock with Claude 2.1 Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "Claude2Generator"
        self.description = "Generator using Bedrock model"
        self.streamable = False
        self.model_name = "anthropic.claude-v2"  # replace with your model id
        self.context_window = 10000
        self.bedrock_client = BedrockClient()

    async def generate(
        self,
        queries: list[str],
        context: list[str],
        conversation: dict = None,
    ) -> str:
        if conversation is None:
            conversation = {}
        messages = self.prepare_messages(queries, context, conversation)

        try:
            response = self.bedrock_client.invoke_model(
                body=messages,
                modelId=self.model_name,
                accept="application/json",
                contentType="application/json"
            )
            system_msg = str(response["completion"])

        except Exception:
            raise

        return system_msg

    async def generate_stream(
        self,
        queries: list[str],
        context: list[str],
        conversation: dict = None,
    ) -> Iterator[dict]:
        if conversation is None:
            conversation = {}
        messages = self.prepare_messages(queries, context, conversation)

        try:
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.model_name,
                body=messages
            )

            try:
                while True:
                    chunk = await response.__anext__()
                    if "content" in chunk["choices"][0]["delta"]:
                        yield {
                            "message": chunk["choices"][0]["delta"]["content"],
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
                    else:
                        yield {
                            "message": "",
                            "finish_reason": chunk["choices"][0]["finish_reason"],
                        }
            except StopAsyncIteration:
                pass

        except Exception:
            raise

    def prepare_messages(
        self, queries: list[str], context: list[str], conversation: dict[str, str]
    ) -> str:
        messages = "You are a Retrieval Augmented Generation chatbot. Please answer user queries only with the provided context. If the provided documentation does not provide enough information, say so."

        for message in conversation:
            messages += "\n\n" + message.type + ": " + message.content

        query = " ".join(queries)
        user_context = " ".join(context)

        messages += "\n\nUser: " + f"Answer this query: '{query}' given the following context: {user_context}" + "\n\nAssistant:"

        body = json.dumps({
            "prompt": f'"{messages}"',
            "max_tokens_to_sample": 300,
            "temperature": 0.1,
            "top_p": 0.9,
        })

        return body
