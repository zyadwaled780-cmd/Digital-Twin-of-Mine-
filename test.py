import argparse
from dotenv import load_dotenv

from providers import (
    GeminiProvider,
    GroqProvider,
    ProviderConfig,
    ProviderInvocationError,
)


def run_smoke_test(live: bool = False, provider_name: str = "auto") -> int:
    """Verify LLM provider configuration offline by default; pass --live to invoke a provider."""
    load_dotenv(override=False)
    config = ProviderConfig.from_env()
    selected = config.primary_provider if provider_name == "auto" else provider_name

    has_groq = bool(config.groq_api_key)
    has_gemini = bool(config.gemini_api_key)
    print(
        f"Provider config: primary={config.primary_provider} "
        f"groq_key_set={has_groq} groq_model={config.groq_model} "
        f"fallback_enabled={config.fallback_enabled} "
        f"gemini_key_set={has_gemini} gemini_model={config.gemini_model}"
    )

    if not live:
        print(
            f"[DRY-RUN] Ready to test provider={selected}. "
            "Pass --live (and optionally --provider groq|gemini) to execute a real API call."
        )
        return 0

    provider = GroqProvider(config) if selected == "groq" else GeminiProvider(config)
    if not provider.is_configured():
        env_var = "GROQ_API_KEY" if selected == "groq" else "GEMINI_API_KEY"
        print(f"[SKIP] {env_var} is not set. Configure {env_var} before running --live --provider {selected}.")
        return 1

    try:
        res = provider.generate_step(
            system_prompt="Reply with one short confirmation sentence.",
            history=[],
            events=[{"type": "user", "text": "Say one short sentence to confirm you are working."}],
            tools=[],
        )
        print(f"[{res.provider}:{res.model}] {res.text}")
        return 0
    except ProviderInvocationError as err:
        print(f"[ERROR] Live {err.provider} check failed ({err.category}).")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Groq / Gemini smoke check (offline dry-run by default).")
    parser.add_argument("--live", action="store_true", help="Send a real test prompt to the selected provider.")
    parser.add_argument(
        "--provider",
        choices=("auto", "groq", "gemini"),
        default="auto",
        help="Target provider (defaults to configured primary provider).",
    )
    args = parser.parse_args()
    return run_smoke_test(live=args.live, provider_name=args.provider)


if __name__ == "__main__":
    raise SystemExit(main())
