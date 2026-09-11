import os
import requests
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling. ``with_structured_output`` defaults
    to function-calling so the Responses-API parse path is avoided
    (langchain-openai's parse path emits noisy
    PydanticSerializationUnexpectedValue warnings per call without
    affecting correctness).

    Provider-specific quirks (e.g. DeepSeek's thinking mode) live in
    purpose-built subclasses below so this base class stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Two quirks that don't apply to other OpenAI-compatible providers:

    1. **Thinking-mode round-trip.** When DeepSeek's thinking models return
       a response with ``reasoning_content``, that field must be echoed
       back as part of the assistant message on the next turn or the API
       fails with HTTP 400. ``_create_chat_result`` captures the field on
       receive and ``_get_request_payload`` re-attaches it on send.

    2. **deepseek-reasoner has no tool_choice.** Structured output via
       function-calling is unavailable, so we raise NotImplementedError
       and let the agent factories fall back to free-text generation
       (see ``tradingagents/agents/utils/structured.py``).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError(
                "deepseek-reasoner does not support tool_choice; structured "
                "output is unavailable. Agent factories fall back to "
                "free-text generation automatically."
            )
        return super().with_structured_output(schema, method=method, **kwargs)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "monorouter": ("https://monogpt.kr/api/monorouter/v1/openai/v1", "MONOROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

class MonoRouterChatOpenAI(NormalizedChatOpenAI):
    """MonoRouter client using raw requests to avoid OpenAI SDK incompatibility."""

    def _convert_message(self, message):
        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": str(message.content),
            }

        elif isinstance(message, SystemMessage):
            return {
                "role": "system",
                "content": str(message.content),
            }

        elif isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": str(message.content),
            }

        elif isinstance(message, AIMessage):

            payload = {
                "role": "assistant",
                "content": (
                    str(message.content)
                    if message.content is not None
                    else ""
                ),
            }

            tool_calls = message.additional_kwargs.get("tool_calls")

            if tool_calls:
                payload["tool_calls"] = tool_calls

            return payload

        else:
            return {
                "role": "user",
                "content": str(message.content),
            }   





    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        print("MONOROUTER _generate CALLED")
        api_key = os.environ.get("MONOROUTER_API_KEY")
        if not api_key:
            raise ValueError("MONOROUTER_API_KEY is not set.")

        base_url = str(self.openai_api_base or "https://monogpt.kr/api/monorouter/v1/openai/v1")
        url = base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [self._convert_message(m) for m in messages],
            "max_completion_tokens": 2500,
        }

        # Optional fields
        if stop:
            payload["stop"] = stop

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload["temperature"] = kwargs["temperature"]

        # Keep tools if later agents use function/tool calling
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]

        res = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        if res.status_code >= 400:
            raise RuntimeError(f"MonoRouter error {res.status_code}: {res.text}")

        data = res.json()

        choice = data["choices"][0]
        message = choice.get("message", {})

        choice = data["choices"][0]
        message = choice.get("message", {})

        content = message.get("content") or ""

        additional_kwargs = {}

        tool_calls = message.get("tool_calls")
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls

        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

        refusal = message.get("refusal")
        if refusal is not None:
            additional_kwargs["refusal"] = refusal

        if not content and not tool_calls:
            raise RuntimeError(f"MonoRouter returned empty content without tool calls: {data}")

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        response_metadata={
                            "finish_reason": choice.get("finish_reason"),
                            "model_name": self.model_name,
                        },
                    )
                )
            ]
        )

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_CONFIG:
            default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = self.base_url or default_base
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        # DeepSeek's thinking-mode quirks live in their own subclass so the
        # base NormalizedChatOpenAI stays free of provider-specific branches.
        if self.provider == "monorouter":
            chat_cls = MonoRouterChatOpenAI
        elif self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI

        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
