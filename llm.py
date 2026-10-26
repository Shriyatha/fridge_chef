"""One place to choose the LLM. Switch provider with env vars, no code changes.

  LLM_PROVIDER = groq (default) | google | ollama | anthropic
  MODEL        = optional model name override (free-tier model names change over time)
"""
import os


def get_llm(temperature: float = 0, max_tokens: int | None = None):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":  # needs GROQ_API_KEY
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("MODEL", "openai/gpt-oss-120b"),
                        temperature=temperature, max_tokens=max_tokens)

    if provider == "google":  # needs GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("MODEL", "gemini-2.0-flash"),
                                      temperature=temperature, max_output_tokens=max_tokens)

    if provider == "ollama":  # local, no key. Install Ollama, then: ollama pull llama3.1
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("MODEL", "llama3.1"), temperature=temperature)

    if provider == "anthropic":  # needs ANTHROPIC_API_KEY
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("MODEL", "claude-sonnet-5-5"),
                             temperature=temperature, max_tokens=max_tokens or 4096)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


if __name__ == "__main__":
    # List the models your Groq key can use right now:  python llm.py
    import requests
    r = requests.get("https://api.groq.com/openai/v1/models",
                     headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}, timeout=20)
    r.raise_for_status()
    for m in sorted(x["id"] for x in r.json()["data"]):
        print(m)
    print("\nPick one, then: export MODEL=<model id>")