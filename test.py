"""
Smoke test — verifies the active LLM backend with a single inference call.
Run: .venv/bin/python3 test.py
"""
import ghost_config as cfg

print(f"Backend : {'Ollama' if cfg.USE_OLLAMA else f'HuggingFace'}")

if cfg.USE_OLLAMA:
    # ── Ollama path ───────────────────────────────────────────────────────────
    from openai import OpenAI

    model = cfg.OLLAMA_MODEL
    print(f"Model   : {model}")
    print(f"URL     : {cfg.OLLAMA_BASE_URL}\n")

    print(f"Calling {model}…")
    try:
        client = OpenAI(base_url=cfg.OLLAMA_BASE_URL, api_key="ollama")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        print(f"  ✓  Response: {response.choices[0].message.content.strip()}\n")
    except Exception as e:
        print(f"  ✗  Failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  And the model is pulled: ollama pull {model}\n")
        raise SystemExit(1)

else:
    # ── HuggingFace path ──────────────────────────────────────────────────────
    from huggingface_hub import HfApi, InferenceClient

    print(f"Model   : {cfg.HF_TRIAGE_MODEL}")

    print("Checking token…")
    try:
        info = HfApi(token=cfg.HF_TOKEN).whoami()
        role = info.get("auth", {}).get("accessToken", {}).get("role", "unknown")
        print(f"  User : {info.get('name')}")
        print(f"  Role : {role}")
        if role == "read":
            print("\n  ✗  READ-ONLY token — inference providers will reject it.")
            print("     Create a Fine-grained token with 'Make calls to Inference Providers'")
            print("     at https://huggingface.co/settings/tokens\n")
            raise SystemExit(1)
        print("  ✓  Token ok\n")
    except SystemExit:
        raise
    except Exception as e:
        print(f"  ✗  Token check failed: {e}\n")
        raise SystemExit(1)

    print(f"Calling {cfg.HF_TRIAGE_MODEL} via HuggingFace…")
    try:
        client = InferenceClient(api_key=cfg.HF_TOKEN)
        response = client.chat_completion(
            model=cfg.HF_TRIAGE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        print(f"  ✓  Response: {response.choices[0].message.content.strip()}\n")
    except Exception as e:
        print(f"  ✗  Inference failed: {e}\n")
        raise SystemExit(1)

print("All checks passed.")
