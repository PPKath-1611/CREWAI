# CREWAI

This crewAI tool is available to easily integrate LLM models and perform agentic operations.

## AI powered SmartDebate with CrewAI

### Project Description:
SmartDebate is a CrewAI project that stages an automatic, AI‑driven debate on a given topic: one agent argues for the motion, the same (or another) agent argues against it, and a separate judge agent reads both arguments and decides which side is more convincing.
It showcases how to orchestrate multiple LLM agents and tasks (propose, oppose, decide) using CrewAI, with support for local models via Ollama and outputs saved as markdown files.

### What is CrewAI and a Crew?

CrewAI is a framework for building multi-agent LLM systems where multiple specialized agents collaborate on tasks (e.g., a debater and a judge).

#### Core concepts:

- **Agent**: Has a `role`, `goal`, `backstory`, and an `llm`. It executes tasks and can use tools.
- **Task**: Has a `description`, `expected_output`, `agent`, and optional `context` (outputs of other tasks) and `output_file`.
- **Crew**: A collection of agents and tasks, orchestrated with a `process` (e.g., `Process.sequential`) and kicked off with `crew.kickoff(inputs=...)`.

In our Smartdebate project, the crew is:

- **debater** agent: generates arguments for and against a motion.
- **judge** agent: reads both sides and decides which is more convincing.
- **Tasks**: `propose_task` (for), `oppose_task` (against), `decide_task` (judge decides).

## Supported Models and Providers

### Native LLM Providers

CrewAI has **native integrations** for these providers:

- **OpenAI** (`openai/gpt-4o`, etc.)
- **Anthropic / Claude** (`anthropic/claude-sonnet-4-20250514`, etc.)
- **Azure / Azure OpenAI**
- **Google / Gemini**
- **AWS Bedrock**
- And other named native providers listed in the CrewAI docs.

For native providers, you typically set:

- **Model string**: `llm: "openai/gpt-4o"` (or equivalent)
- **Environment variables**: e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### Non‑Native Models via LiteLLM

For everything *not* in the native list (e.g., **Ollama models**):

- CrewAI uses **LiteLLM as a fallback router**.

#### What is LiteLLM?

**LiteLLM** is a unified proxy/router that standardizes LLM calls across providers. It:

- Translates requests into provider-specific formats (OpenAI-compatible, Anthropic, etc.).
- Routes to local models (Ollama, vLLM) and cloud APIs (OpenAI, Anthropic, etc.).
- Provides a consistent interface so frameworks like CrewAI can use any model without custom code per provider.

#### How LiteLLM Supports CrewAI:

1. **Unified interface**: CrewAI sends requests in a standard format; LiteLLM handles provider differences.
2. **Fallback mechanism**: When CrewAI sees a non-native model string (e.g., `ollama/gemma3:27b`), it delegates to LiteLLM instead of failing.
3. **Local model support**: LiteLLM connects to local Ollama instances, enabling offline use without API keys.
4. **Seamless integration**: Once `crewai[litellm]` is installed, CrewAI automatically uses LiteLLM for unrecognized model strings.

**Example non‑native model string**: `llm: "ollama/gemma3:27b"` or `llm: "ollama/llama3.2:3b"`.

**To enable this:**

- Install CrewAI with LiteLLM support:
  - `uv add 'crewai[litellm]'`
  - or `pip install 'crewai[litellm]'`

## Using Ollama with CrewAI

### Basic Setup

1. **Install and run Ollama** locally.
2. **Pull a model**, e.g.:

```bash
ollama pull gemma3:27b
ollama serve
```

3. **Configure agents** via YAML (recommended CrewAI pattern):

```yaml
debater:
  role: >
    A compelling debater
  goal: >
    Present the argument either in favor or against the motion. The motion is {motion}.
  backstory: >
    You're a debater who is very knowledgeable about the topic and can present a strong argument either in favor or against the motion.
    The motion is {motion}.
  llm: ollama/gemma3:27b

judge:
  role: >
    Decide the winner based on the arguments presented by the debaters.
  goal: >
    Given arguments for and against the motion, decide which side is more convincing, based purely on the arguments presented.
  backstory: >
    You're a fair judge with a reputation for weighing arguments without factoring in your own views.
    The motion is {motion}.
  llm: ollama/gemma3:27b
```

**Important**: Use `llm` (not `model`) in YAML; otherwise CrewAI falls back to environment defaults (e.g., OpenAI) and demands `OPENAI_API_KEY`.

### Offline Usage

- Once a model is downloaded via `ollama pull`, **all inference is local**.
- CrewAI → LiteLLM → Ollama will work **without internet** as long as:
  - Ollama is running.
  - You're not using tools that themselves call external APIs.

## When CrewAI Is a Good Fit (and When It Isn't)

### Good Use Cases

- **Multi‑step, multi‑role workflows**:
  - Debates, code review pipelines, research + writing, moderation + summarization, etc.
- **Orchestration of multiple agents** with clearly defined roles and sequential or hierarchical processes.
- **Projects where you want structured outputs**, logging, test/training loops, or flows with state.

### Less Suitable Cases

- **Single, simple chatbots or one‑shot prompts**:
  - A direct OpenAI/Ollama call (or a simpler framework) may be easier.
- **Ultra‑low‑latency, tiny deployments**:
  - CrewAI overhead plus multiple LLM calls may be too heavy if you need millisecond responses.
- **Very constrained environments** where installing Python packages, LiteLLM, or running a local server (Ollama) isn't allowed.

## Model Choice and Behavior (Gemma / Llama Notes)

- **Very small models (e.g., `gemma3:270m`)**:
  - Often **ignore instructions**, conflate roles, and **parrot prompt text**.
  - In our case: `oppose_task` produced nearly the same text as `propose_task`, and `decide_task` just echoed `expected_output`.
- **Medium / large models (e.g., `gemma3:27b`, `llama3.2:3b`, `gemma2:2b/9b`)**:
  - Much better instruction-following and role separation.
  - Better at:
    - Arguing for vs against the same motion.
    - Weighing arguments and making a coherent decision as the judge.
- **Trade‑off**:
  - Larger = better quality but slower and more resource‑intensive.
  - For a laptop, **3B–9B** is often a sweet spot; **27B** is great if your GPU/VRAM can handle it.

## Wiring Tasks Correctly (SmartDebate Example)

### Agents (`crew.py`):

```python
@agent
def debater(self) -> Agent:
    return Agent(
        config=self.agents_config['debater'],  # type: ignore[index]
        verbose=True
    )

@agent
def judge(self) -> Agent:
    return Agent(
        config=self.agents_config['judge'],  # type: ignore[index]
        verbose=True
    )
```

### Tasks (`tasks.yaml`):

```yaml
propose_task:
  description: >
    Be very convincing in presenting the argument for the motion.
  expected_output: >
    Your clear argument in favor of the motion, in a concise manner
  agent: debater
  output_file: output/propose.md

oppose_task:
  description: >
    You are in opposition to the motion: {motion}.
    Be very convincing in presenting the argument against the motion.
  expected_output: >
    A persuasive argument against the motion.
  agent: debater
  output_file: output/oppose.md

decide_task:
  description: >
    Review the arguments presented by the debaters and decide which side is more convincing,
    based purely on the arguments presented.
  expected_output: >
    A decision on which side is more convincing and why,
    based purely on the arguments presented.
  agent: judge
  output_file: output/decide.md
```

### Optional: Give the Judge Actual Context

Give the judge actual context (so it can see both arguments):

```python
@task
def decide_task(self) -> Task:
    return Task(
        config=self.tasks_config['decide_task'],  # type: ignore[index]
        context=[self.propose_task(), self.oppose_task()],
    )
```

## Common Troubleshooting Tips for CrewAI + Ollama

### `OPENAI_API_KEY is required` even though you're using Ollama

- **Likely cause**: using `model:` in YAML or not setting `llm` on the agent.
- **Fix**: use `llm: ollama/<model>` in `agents.yaml` so CrewAI doesn't fall back to the OpenAI env-based default.

### Error: model did not match any supported native provider, LiteLLM not installed

- **Cause**: using non‑native model string (e.g., `ollama/gemma3:27b`) without LiteLLM.
- **Fix**: `uv add 'crewai[litellm]'` or `pip install 'crewai[litellm]'`.

### `KeyError: 'judge'` or similar

- **Cause**: task refers to an agent key that doesn't have a matching `@agent` method name.
- **Fix**: ensure agent method names match YAML keys exactly (e.g., `def judge(self)` for `agent: judge`).

### Tasks writing unexpected content (e.g., decide.md just shows expected_output)

- **With very small models**, the LLM may just echo prompt text or ignore roles.
- **Fix**: use a stronger model (3B–27B), and provide proper task context.

### Crew runs, but outputs from different tasks look the same

- **Likely a combination of**:
  - Same agent/LLM used for both tasks.
  - Very small or weak model.
  - Not enough instruction contrast in descriptions.
- **Fix**: upgrade model and refine task descriptions; consider separate agents if needed.

### Offline usage concerns

- After model is pulled, Ollama runs fully local.
- Ensure no tools depend on external APIs if you want a fully offline Crew.

---
