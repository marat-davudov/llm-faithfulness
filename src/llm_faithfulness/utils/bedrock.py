from functools import lru_cache

import boto3
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

from llm_faithfulness.utils.logging_utils import get_logger

logger = get_logger("bedrock_utils")


@lru_cache(maxsize=1)
def _get_bedrock_runtime_client() -> BedrockRuntimeClient:
    """
    Get a cached Bedrock client instance.

    Returns:
        BedrockRuntimeClient: A cached instance of the Bedrock runtime client.
    """
    return boto3.client("bedrock-runtime")


def invoke_model_converse(
    model_id: str, system: str, user: str, client: BedrockRuntimeClient | None = None
) -> str:
    """
    Invoke a Bedrock model using the converse API with the given system and user prompts.

    Args:
        model_id (str): AWS Bedrock model identifier
        system (str): System prompt
        user (str): User prompt
        client (BedrockRuntimeClient, optional): Bedrock runtime client instance. Defaults to None.

    Returns:
        str: Model's response text.
    """
    logger.debug(
        "Invoking model %s (system=%d chars, user=%d chars)",
        model_id,
        len(system) if system is not None else 0,
        len(user) if user is not None else 0,
    )

    if client is None:
        client = _get_bedrock_runtime_client()

    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": user}]}],
        system=[{"text": system}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.0},
    )

    text = response["output"]["message"]["content"][0]["text"]
    logger.debug("Received response from %s (%d chars)", model_id, len(text))
    return text


if __name__ == "__main__":
    logger.info("Bedrock utils module executed directly.")
