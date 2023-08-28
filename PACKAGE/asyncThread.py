"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random


async def _throttled_openai_chat_completion_acreate(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(10):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    f"OpenAI API rate limit exceeded. Sleeping for 60 seconds."
                )
                await asyncio.sleep(60)
            except asyncio.exceptions.TimeoutError or openai.error.Timeout:
                logging.warning(f"OpenAI API timeout. Sleeping for 60 seconds.")
                await asyncio.sleep(60)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}.Sleeping for 60 seconds.")
                await asyncio.sleep(60)
            except openai.error.ServiceUnavailableError as e:
                logging.warning(f"OpenAI error:{e}")
                await asyncio.sleep(10)
            except Exception as e:
                logging.warning(f"Exception OR Error:{e}")
                return {"choices": [{"message": {"content": ""}}]}
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        api_key: str,
        messages,
        engine_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        requests_per_minute: int,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = api_key
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]


def run(messages, engine_name, temperature, max_tokens, top_p, api_key, requests_per_minute):
    # messages = [[{"role": "user", "content": f"print the number {i}"}] for i in range(100)]
    responses = asyncio.run(generate_from_openai_chat_completion(
        messages=messages,
        engine_name=engine_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_key=api_key,
        requests_per_minute=requests_per_minute
    ))
    return responses
