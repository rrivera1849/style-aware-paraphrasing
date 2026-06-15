"""
Mistral-7B-Instruct-v0.3 paraphraser wrapper.

The style-aware paraphraser is trained to map *Mistral-7B paraphrases* of a
machine text back into a target author's style. So before invoking our model
on a novel input, you must first paraphrase that input P=5 times with
Mistral-7B using the prompt from Appendix H.1 of the paper.

This module is a thin wrapper around vLLM that does exactly that. The author
bank shipped with the release contains pre-computed paraphrases — only run
this module for inputs that aren't already in the bank.
"""

from typing import List

from vllm import LLM, SamplingParams

from utils import MISTRAL_MODEL_ID

PARAPHRASE_PROMPT = (
    "[INST]Paraphrase the following text, do NOT output explanations, "
    "comments, or anything else, only the paraphrase: {passage}[/INST] Output:"
)


def paraphrase_p5(
    text: str,
    llm: LLM = None,
    num_paraphrases: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    model_id: str = MISTRAL_MODEL_ID,
) -> List[str]:
    """Return `num_paraphrases` Mistral-7B paraphrases of `text`.

    Pass an already-loaded `vllm.LLM` instance via `llm` to avoid the multi-minute
    cold start when calling repeatedly.
    """
    if llm is None:
        llm = LLM(model_id, gpu_memory_utilization=0.90)

    sp = SamplingParams(
        n=num_paraphrases,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    [out] = llm.generate([PARAPHRASE_PROMPT.format(passage=text)], sp)
    return [o.text.strip() for o in out.outputs]


if __name__ == "__main__":
    import fire

    def cli(text: str, num_paraphrases: int = 5):
        for i, p in enumerate(paraphrase_p5(text, num_paraphrases=num_paraphrases)):
            print(f"Paraphrase {i}: {p}")

    fire.Fire(cli)
