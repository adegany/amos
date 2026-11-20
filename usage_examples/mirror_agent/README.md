# Mirror Agent

Run the full reference agent from the repository root after trying a [single memory behavior](../README.md).

```sh
PYTHONPATH=src python3 usage_examples/mirror_agent/mirror_agent_demo.py --format text
```

For the browser inspector and chat, start the UI and open the printed loopback address. The offline renderer runs locally. Choose `--lm codex` to use a configured Codex CLI for chat.

```sh
PYTHONPATH=src python3 usage_examples/mirror_agent/mirror_agent_ui.py --host 127.0.0.1 --port 8787 --lm offline
```

The walkthrough exercises memory, governance, reasoning, and maintenance in one agent. [Demo specification](../../docs/mirror-agent-demo-spec.md)
