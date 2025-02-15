import dspy
from prompt import Prompt
import json
import os
import random
from typing import Callable
import os
from data import dev

SOLVE_TEMPERATURE = 0.0

class Population:
    def __init__(self, prompts: list[Prompt], solve) -> None:
        self.prompts: list[Prompt] = prompts
        self.avg_score, self.max_score = -1.0, -1.0
        self.ranked = False
        self.solve = solve
        self.current_gen = 0

        # "tool_name": (uses, average_score)
        self.tool_effectivity = {
            "lamarckian": (0,0.0),
            "reflective": (0,0.0),
            "iterative": (0,0.0),
            "crossover": (0,0.0),
        }

    def update_tool_effectivity(self, tool: str, score: float) -> None:
        count = self.tool_effectivity[tool][0] + 1
        prev_avg = self.tool_effectivity[tool][1] 
        new_avg = prev_avg + (score-prev_avg)/count
        self.tool_effectivity[tool] = (count, new_avg)

    def extend_and_score(self, prompt: Prompt) -> None:
        score = prompt.score(dev, self.solve)
        prompt.gen = self.current_gen
        self.prompts.append(prompt)
        self.prompts.sort(key=lambda p: p.get_dev(), reverse=True)
        return score
    
    def __iter__(self):
        return iter(self.prompts)
    
    def top_n(self, n: int) -> list[Prompt]:
        return self.prompts[:n]

    def select(self, n: int) -> list[Prompt]:
        counts = [p.score_to_count() for p in self.prompts]
        return random.sample(self.prompts, n, counts=counts)

    def stats(self) -> tuple[float, float]:
        scores = [p.get_dev() for p in self.prompts]
        self.avg_score = sum(scores) / len(scores)
        self.max_score = max(scores)
        return self.avg_score, self.max_score

    def dump(self):
        with open(f"{os.getenv('RUN_FOLDER')}/prompts{self.current_gen}.jsonl", "w", encoding="utf-8") as f:
            for prompt in self.prompts:
                json.dump(prompt.jsoned(), f)
                f.write("\n")
                
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, index):
        return self.prompts[index]

    def filter_by_iteration(self) -> list[list[Prompt]]:
        max_gen = max([p.gen for p in self.prompts])
        return [list(filter(lambda p: p.gen == i, self.prompts)) for i in range(max_gen+1)]
    
    def evaluate_iterations(self, data: list[dspy.Example], top_n: int = -1) -> list[list[float]]:
        generations = self.filter_by_iteration()
        scores_by_gen = []
        for gen in generations:
            gen_scores = []
            gen = gen if top_n == -1 else gen[:top_n]
            for prompt in gen:
                score = prompt.score(data, self.solve, final=True)
                gen_scores.append(score)
            scores_by_gen.append(gen_scores)
        return scores_by_gen
    
    def quartile(self, i: int) -> list[Prompt]:
        quarter = len(self.prompts) //4
        return self.prompts[(i-1)*quarter:i*quarter]

lm = os.environ["SOLVE_LM"]
vllm_port = os.getenv("VLLM_MY_PORT")
api_key = os.getenv("API_KEY")
if vllm_port:
    solve_lm = dspy.LM(lm, api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
else:
    from dotenv import load_dotenv
    load_dotenv()
    solve_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)
solve_module = dspy.ChainOfThought("question: str -> answer: float", temperature=SOLVE_TEMPERATURE)
def solve(question):
    try:
        with dspy.context(lm=solve_lm):
            ret = solve_module(question=question)
        return ret
    except ValueError as e:
        return None

population = Population([], solve)