"""
Utils module for LLM and Embedding model configuration using litellm.
"""

import json
import os
from typing import Optional, Dict, Any, Callable
from litellm import completion, embedding
from dotenv import load_dotenv

load_dotenv()

# 从环境变量加载 LLM 模型配置
LLM_MODELS = json.loads(os.getenv("LLM_MODELS", "[]"))

# 从环境变量加载 Embedding 模型配置
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")


def get_llm(model_id: str = "glm5", thinking: bool = False) -> Callable:
    """
    返回指定模型的 LLM 调用函数。

    Args:
        model_id: 模型标识符，对应 .env 中 LLM_MODELS 的 model_name。
                  默认为 "glm5"。
                  可选值: "qwen3.5-27b", "qwen3.5-flash-thinking", "glm5", "glm5-thinking"
        thinking: 是否开启思考模式，默认为 False。
                  True 时自动选择带 "-thinking" 后缀的模型。

    Returns:
        一个可调用的函数，接受 messages 参数进行 LLM 调用。

    Raises:
        ValueError: 当指定的 model_id 不存在时抛出。

    Example:
        >>> llm = get_llm(model_id="glm5", thinking=True)
        >>> response = llm(messages=[{"role": "user", "content": "Hello"}])
    """
    # 根据 thinking 参数确定实际要查找的 model_name
    if thinking:
        target_model_name = f"{model_id}-thinking"
    else:
        target_model_name = model_id

    # 查找指定的模型配置
    model_config = None
    for m in LLM_MODELS:
        if m.get("model_name") == target_model_name:
            model_config = m
            break

    if model_config is None:
        available_models = [m.get("model_name") for m in LLM_MODELS]
        raise ValueError(
            f"Model '{target_model_name}' not found. Available models: {available_models}"
        )

    # 直接使用 litellm_params
    litellm_params = model_config.get("litellm_params", {})

    def llm_call(
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        调用 LLM 的函数。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 可选的温度参数
            max_tokens: 可选的最大 token 数
            **kwargs: 其他传递给 litellm.completion 的参数

        Returns:
            litellm 的 completion 响应对象
        """
        call_params = {
            "model": litellm_params.get("model", ""),
            "api_key": litellm_params.get("api_key", ""),
            "messages": messages,
        }

        extra_body = litellm_params.get("extra_body")
        if extra_body:
            call_params["extra_body"] = extra_body

        if temperature is not None:
            call_params["temperature"] = temperature
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens

        call_params.update(kwargs)

        return completion(**call_params)

    return llm_call


def get_embedding_model() -> Dict[str, Any]:
    """
    返回 embedding 模型配置和调用函数。

    Returns:
        包含模型配置和调用函数的字典：
        - model: 模型名称
        - api_key: API 密钥
        - embed: 嵌入函数，接受文本输入返回 embedding 向量

    Raises:
        ValueError: 当 EMBEDDING_MODEL_NAME 或 EMBEDDING_API_KEY 未配置时抛出。

    Example:
        >>> embedding_config = get_embedding_model()
        >>> embeddings = embedding_config["embed"](["Hello", "World"])
    """
    if not EMBEDDING_MODEL_NAME or not EMBEDDING_API_KEY:
        raise ValueError(
            "EMBEDDING_MODEL_NAME and EMBEDDING_API_KEY must be set in environment variables"
        )

    def embed_function(texts: list, **kwargs) -> Any:
        """
        调用 embedding 模型的函数。

        Args:
            texts: 文本列表
            **kwargs: 其他传递给 litellm.embedding 的参数

        Returns:
            litellm 的 embedding 响应对象
        """
        return embedding(
            model=EMBEDDING_MODEL_NAME, api_key=EMBEDDING_API_KEY, input=texts, **kwargs
        )

    return {
        "model": EMBEDDING_MODEL_NAME,
        "api_key": EMBEDDING_API_KEY,
        "embed": embed_function,
    }


# 便捷函数：直接获取 embedding
def embed_texts(texts: list, **kwargs) -> Any:
    """
    便捷函数：直接对文本列表进行 embedding。

    Args:
        texts: 文本列表
        **kwargs: 其他传递给 litellm.embedding 的参数

    Returns:
        litellm 的 embedding 响应对象

    Example:
        >>> embeddings = embed_texts(["Hello", "World"])
    """
    embedding_config = get_embedding_model()
    return embedding_config["embed"](texts, **kwargs)


def get_llm_simple(model_id: str = "glm5", thinking: bool = False) -> Callable:
    """
    返回一个接受单个 prompt 字符串的 LLM 调用函数。

    与 get_llm 不同，此函数返回的调用函数接受单个 prompt 字符串，
    而不是 messages 列表，适合一次性问答场景。

    Args:
        model_id: 模型标识符，对应 .env 中 LLM_MODELS 的 model_name。
                  默认为 "glm5"。
        thinking: 是否开启思考模式，默认为 False。

    Returns:
        一个可调用的函数，接受单个 prompt 字符串进行 LLM 调用。

    Raises:
        ValueError: 当指定的 model_id 不存在时抛出。

    Example:
        >>> llm = get_llm_simple(model_id="glm5")
        >>> response = llm("请解释什么是机器学习？")
        >>> # 或者使用 invoke 方法（类似 LangChain 风格）
        >>> response = llm.invoke("请解释什么是机器学习？")
    """
    llm_call = get_llm(model_id=model_id, thinking=thinking)

    def simple_call(prompt: str, **kwargs) -> Any:
        """
        调用 LLM 的函数。

        Args:
            prompt: 单个 prompt 字符串
            **kwargs: 其他传递给底层 LLM 调用的参数（如 temperature, max_tokens）

        Returns:
            litellm 的 completion 响应对象
        """
        return llm_call(messages=[{"role": "user", "content": prompt}], **kwargs)

    # 添加 invoke 方法，使其兼容 LangChain 风格的调用
    simple_call.invoke = simple_call

    return simple_call


def list_available_models() -> list:
    """
    列出所有可用的 LLM 模型。

    Returns:
        可用模型名称列表
    """
    return [m.get("model_name") for m in LLM_MODELS]


if __name__ == "__main__":
    # 测试代码
    print("Available models:", list_available_models())

    print("\nTesting get_llm with glm5 (thinking=False)...")
    llm = get_llm(model_id="glm5", thinking=False)
    print(f"LLM function created: {llm}")

    print("\nTesting get_llm with glm5 (thinking=True)...")
    llm_thinking = get_llm(model_id="glm5", thinking=True)
    print(f"LLM thinking function created: {llm_thinking}")

    print("\nTesting get_llm_simple with glm5...")
    llm_simple = get_llm_simple(model_id="glm5")
    print(f"LLM simple function created: {llm_simple}")
    print(f"LLM simple invoke method: {llm_simple.invoke}")

    print("\nTesting get_embedding_model...")
    emb = get_embedding_model()
    print(f"Embedding model: {emb['model']}")
    print(f"Embed function: {emb['embed']}")
