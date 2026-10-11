# Prompts for building ToolShortlist

Open a coding-agent session in this folder and paste the kickoff prompt. Use the stage prompt for each later stage.

## Kickoff prompt

```
Read PROJECT.md in this folder from start to finish before doing anything else. It is the specification for this project and it overrides your defaults wherever the two differ.

What I am building: a small trained model that retrieves the right tools for an AI agent from a large catalog, evaluated against public benchmarks and shipped as an installable package with a live demo. This is my portfolio project for Machine Learning Engineer applications, and I will be questioned on every design decision in interviews. So the goal is twofold: a correct, reproducible project, and my being able to explain all of it.

Start with Stage 0 only. Do not write project code yet.

For Stage 0:
1. Read the ToolRet and LiveMCPBench papers and repositories, and skim the other prior work listed in section 9.
2. For each dataset in section 6, find where it is hosted, its licence, its format, its official train/eval split and its official metrics. Write this up in DATA.md.
3. Confirm from each provider's own page that the free resources in section 5 are still free and need no credit card. Report the limits you find.
4. Go through every item in PROJECT.md marked "to confirm" or "not confirmed" and resolve it. If you cannot verify something, say so plainly; do not fill the gap from memory.
5. Tell me how this project differs from the prior work you read, and whether the angle suggested in section 9 still holds. If someone has already built exactly this, tell me and propose two alternative angles.

Rules for the whole project:
- Zero cost. If any step seems to need a paid service or a credit card, stop and tell me. The one exception is the $3 OpenAI budget described in section 5.1: it is for Stage 3 reference comparisons only, and every batch needs a printed cost estimate and my approval before it runs. Never ask me to paste the key; read it from the OPENAI_API_KEY environment variable.
- Never state a number that a script did not produce. Results go to results/*.json and tables are rendered from those files.
- Never let test data touch training, hard-negative mining, checkpoint selection or hyperparameter choice.
- Prefer the simplest thing that works. Do not add frameworks, abstractions or features that PROJECT.md does not ask for.
- If the evidence goes against the claim in section 2, report it as it is. A clear negative result is acceptable; a flattering one that does not hold is not.
- When you make a choice I did not specify, tell me what you chose, what you rejected and why, in two or three sentences.
- You cannot run Kaggle. When GPU training is needed, prepare a self-contained notebook and tell me exactly what to run and what to bring back.
- Leave the items in section 11 to me. For those, give me questions or an outline, not finished text.

When Stage 0 is finished, stop and give me:
- what you verified, with links
- what you could not verify
- any change you recommend to PROJECT.md, and why
- the open decisions that need my answer before Stage 1
```

## Stage prompt (Stages 1 to 4)

Replace `N` with the stage number.

```
Re-read PROJECT.md, then carry out Stage N. The same rules as before apply.

Before writing code, give me a short plan: the files you will create or change, the commands that will prove the exit criteria are met, and anything in the stage you think is wrong or underspecified. Wait for my go-ahead if you are proposing to deviate from PROJECT.md; otherwise proceed.

While working:
- Write tests for anything that computes a metric, and check them against cases small enough to work out by hand.
- Run what you build. Do not report something as working unless you ran it and saw the output.
- If a result looks too good, assume a bug or a leak first and check for it before telling me.

When the stage is finished, stop and give me:
- each exit criterion from PROJECT.md and the evidence that it is met, or a plain statement that it is not
- the results table, rendered from results/*.json
- every decision you made that I did not specify, with the alternative you rejected
- three questions an interviewer would be likely to ask me about this stage, so I can check that I understand it
```

## Review prompt (run in a fresh session before publishing)

```
Read PROJECT.md, then review this repository as a sceptical ML engineer screening a candidate. You did not write this code. Look for:
- test data leaking into training, mining or model selection
- numbers in README.md that do not match results/*.json or cannot be regenerated
- metric implementations that disagree with the benchmark's official definition
- claims that are stronger than the evidence
- steps in the reproduction instructions that fail from a clean clone

Run the tests and the evaluation yourself. Report findings most serious first, each with the file and line and how you confirmed it. Say clearly what you checked and found sound.
```
