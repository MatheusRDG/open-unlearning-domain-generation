"""Utility functions for logging and LLM initialization."""

import logging
import pprint
import sys
from datetime import datetime
from typing import Dict

from langchain.chat_models import init_chat_model
from loguru import logger as loguru_logger
from retrying import retry

from src.domain_generation.config import config

# -----------------------------------------------------------------------------
# Logging setup (Loguru + stdlib interoperability)
# -----------------------------------------------------------------------------

LOG_FORMAT = (
    "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> "
    "| <level>{level: <7}</level> "
    "| <cyan>{extra[component]: <18}</cyan> "
    "| {message}"
)

loguru_logger.remove()
loguru_logger.add(sys.stderr, format=LOG_FORMAT, colorize=True, enqueue=False)


class InterceptHandler(logging.Handler):
    """Redirect standard logging records through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - passthrough
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Add component field if missing (for third-party libraries)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind a default component for third-party logs
        loguru_logger.bind(component=record.name or "external").opt(
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = loguru_logger.bind(component="domain_generation")

pp = pprint.PrettyPrinter(indent=2)

STAGE_COLOR_HINTS: Dict[str, str] = {
    "start": "green",
    "waiting": "yellow",
    "wait": "yellow",
    "complete": "blue",
    "chapters_complete": "blue",
    "sections_complete": "blue",
    "end": "cyan",
    "truncate": "magenta",
    "error": "red",
    "warning": "yellow",
    "fail": "red",
}

STEP_COLOR_PALETTE = [
    "magenta",
    "cyan",
    "yellow",
    "blue",
    "green",
    "white",
]


def _color_for_stage(stage: str) -> str:
    stage_lower = stage.lower()
    for key, color in STAGE_COLOR_HINTS.items():
        if key in stage_lower:
            return color
    return "white"


def _color_for_step(step: str) -> str:
    index = abs(hash(step)) % len(STEP_COLOR_PALETTE)
    return STEP_COLOR_PALETTE[index]


def pretty_log(step: str, stage: str, payload, full_dump: bool = False):
    """Pretty-print a log entry for a graph step with color-coded context."""

    try:
        serialized = payload.model_dump() if hasattr(payload, "model_dump") else payload
    except Exception:
        serialized = payload

    if full_dump:
        formatted = pp.pformat(serialized)
    elif isinstance(serialized, (dict, list, tuple, set)):
        formatted = pp.pformat(serialized)
    elif serialized is None:
        formatted = ""
    else:
        formatted = str(serialized)
    stage_color = _color_for_stage(stage)
    step_color = _color_for_step(step)

    colored_stage = f"<{stage_color}>{stage}</{stage_color}>"
    colored_step = f"<{step_color}>{step}</{step_color}>"

    if formatted and formatted != "None":
        message = f"{colored_step} | {colored_stage}\n<dim>{formatted}</dim>"
    else:
        message = f"{colored_step} | {colored_stage}"

    logger.opt(colors=True).info(message)


# -----------------------------------------------------------------------------
# LLM Retry Wrapper
# -----------------------------------------------------------------------------


def _should_retry_on_exception(exception: Exception) -> bool:
    """Determine if we should retry based on the exception type.

    Args:
        exception: The exception that was raised

    Returns:
        True if we should retry, False otherwise
    """
    import openai

    # Retry on rate limit errors
    if isinstance(exception, openai.RateLimitError):
        logger.opt(colors=True).warning(
            "<yellow>[RETRY]</yellow> Rate limit error, will retry with exponential backoff"
        )
        return True

    # Retry on API connection errors
    if isinstance(exception, (openai.APIConnectionError, openai.APITimeoutError)):
        logger.opt(colors=True).warning(
            "<yellow>[RETRY]</yellow> API connection/timeout error, will retry"
        )
        return True

    # Retry on length limit errors (token limit reached)
    if isinstance(exception, openai.LengthFinishReasonError):
        logger.opt(colors=True).warning(
            "<yellow>[RETRY]</yellow> Length limit reached, will retry"
        )
        return True

    # Don't retry on other errors
    return False


class RetryableLLM:
    """Wrapper that adds retry logic to LLM invoke calls with fallback support."""

    def __init__(self, llm, fallback_model_name=None):
        """Initialize with an LLM instance.

        Args:
            llm: The underlying LLM instance to wrap
            fallback_model_name: Optional fallback model to use after retries exhausted
        """
        self._llm = llm
        self._fallback_model_name = fallback_model_name
        self._schema = None  # Track schema for structured output

    def with_structured_output(self, schema):
        """Return a new RetryableLLM with structured output.

        Args:
            schema: Pydantic model or schema for structured output

        Returns:
            New RetryableLLM instance with structured output configured
        """
        structured_llm = self._llm.with_structured_output(schema)
        new_instance = RetryableLLM(structured_llm, self._fallback_model_name)
        new_instance._schema = schema  # Store schema for fallback
        return new_instance

    @retry(
        retry_on_exception=_should_retry_on_exception,
        wait_exponential_multiplier=2000,  # Start at 2 seconds
        wait_exponential_max=120000,  # Max 120 seconds between retries
        stop_max_attempt_number=10,  # Max 10 attempts
    )
    def invoke(self, *args, **kwargs):
        """Invoke the LLM with retry logic and fallback support.

        Retries with exponential backoff on rate limit, connection, and length errors.
        Wait times: 2s, 4s, 8s, 16s, 32s, 64s, 120s, 120s, 120s (capped at 120s)

        After all retries exhausted, falls back to gpt-4o-mini if configured.

        Args:
            *args: Positional arguments to pass to LLM invoke
            **kwargs: Keyword arguments to pass to LLM invoke

        Returns:
            LLM response

        Raises:
            Exception if all retries and fallback fail
        """
        try:
            return self._llm.invoke(*args, **kwargs)
        except Exception as e:
            # If all retries exhausted and we have a fallback model, try it
            if self._fallback_model_name:
                logger.opt(colors=True).warning(
                    "<red>[FALLBACK]</red> All retries exhausted. "
                    "Attempting fallback to '<yellow>{}</yellow>'",
                    self._fallback_model_name
                )
                try:
                    # Initialize fallback model
                    fallback_llm = init_chat_model(
                        self._fallback_model_name,
                        temperature=config.temperature,
                        max_retries=5,
                        max_tokens=config.max_tokens,
                    )

                    # Apply structured output if needed
                    if self._schema:
                        fallback_llm = fallback_llm.with_structured_output(self._schema)

                    # Try fallback with its own retry logic (5 attempts)
                    @retry(
                        retry_on_exception=_should_retry_on_exception,
                        wait_exponential_multiplier=2000,
                        wait_exponential_max=60000,
                        stop_max_attempt_number=5,
                    )
                    def _invoke_fallback():
                        return fallback_llm.invoke(*args, **kwargs)

                    result = _invoke_fallback()
                    logger.opt(colors=True).success(
                        "<green>[FALLBACK]</green> Successfully used fallback model"
                    )
                    return result

                except Exception as fallback_error:
                    logger.opt(colors=True).error(
                        "<red>[FALLBACK]</red> Fallback model also failed: {}",
                        fallback_error
                    )
                    raise fallback_error
            else:
                # No fallback configured, re-raise original error
                raise e

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying LLM.

        Args:
            name: Attribute name

        Returns:
            Attribute value from underlying LLM
        """
        return getattr(self._llm, name)


# -----------------------------------------------------------------------------
# LLM Initialization
# -----------------------------------------------------------------------------


def get_llm():
    """Get configured LLM instance with fallback support and retry logic.

    Returns:
        RetryableLLM instance wrapping the configured ChatOpenAI instance.
    """
    # Set fallback to gpt-4o-mini if not already configured
    fallback_model = config.fallback_model_name or "gpt-4o-mini"

    try:
        logger.opt(colors=True).info(
            "<green>[LLM]</green> Initializing model '<yellow>{}</yellow>' "
            "(retries=10, max_tokens={}, fallback={})",
            config.model_name,
            config.max_tokens,
            fallback_model,
        )
        base_llm = init_chat_model(
            config.model_name,
            temperature=config.temperature,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
        )
        # Wrap with retry logic and fallback support
        return RetryableLLM(base_llm, fallback_model_name=fallback_model)
    except Exception as exc:
        logger.opt(colors=True).error(
            "<red>[LLM]</red> Failed to initialize primary model '<red>{}</red>': {}",
            config.model_name,
            exc,
        )
        raise RuntimeError(
            f"Unable to initialize configured chat model '{config.model_name}': {exc}"
        )


def get_current_date() -> str:
    """Get current date formatted for prompts.

    Returns:
        Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")
