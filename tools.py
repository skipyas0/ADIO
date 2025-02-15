import dspy
import signatures
import logging
from population import population
from data import train
from prompt import Prompt
import random
import Levenshtein
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{os.getenv('RUN_FOLDER')}/tools.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



TEMPERATURE = 0.7


### TOOLS FOR PROMPT GENERATION
def lamarckian(hint: str = "") -> tuple[str, float]:
    """
    Task another component with access to the data to generate a new prompt.
    You may provide a short hint.
    This tool returns the resulting prompt and its score.

    Args:
        hint (str): Specific instruction on how a new prompt should be constructed.

    Returns:
        tuple[str, float]: The resulting prompt and its score on the training batch.
    """
    module = dspy.ChainOfThought(signature=signatures.Lamarckian, temperature=TEMPERATURE)
    prompt = module(examples=str(train), supervisor_hint=hint).prompt_proposal
    prompt_obj = Prompt(prompt, origin="lamarckian")
    score = population.extend_and_score(prompt_obj)
    population.update_tool_effectivity("lamarckian", score)
    logger.info(f"LAMARCKIAN generated prompt:\n {str(prompt_obj)}\nSCORE: {score}\nTools:{population.tool_effectivity}")
    return (str(prompt_obj), score)

def reflective(hint: str = "") -> tuple[str, float]:
    """
    Task another component to improve an underperfoming prompt.
    The prompt is automatically chosen from the worst quartile.
    Your subordinate will write a critique of the prompt and try to write a better one.
    You may provide a short hint.
    This tool returns the resulting prompt and its score.

    Args:
        hint (str): Specific instruction on how a new prompt should be constructed.

    Returns:
        tuple[str, float]: The resulting prompt and its score on the training batch.
    """
    if len(population) == 0:
        return lamarckian()
    module = dspy.ChainOfThought(signature=signatures.Reflective, temperature=TEMPERATURE)
    worst_quartile = population.quartile(4)
    original = random.choice(worst_quartile) if len(worst_quartile) > 1 else population.prompts[0]
    completion = original.get_completion(0)
    task = completion[0]
    reasoning = completion[1]
    completion = module(prompt=original.text, task_question=task.question, task_gold_answer=task.answer, reasoning=reasoning, supervisor_hint=hint)
    prompt = completion.prompt_proposal
    critique = completion.prompt_critique
    prompt_obj = Prompt(prompt, origin="reflective")
    score = population.extend_and_score(prompt_obj)
    logger.info(f"REFLECTIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nOriginal prompt:\n{original.text}\nCritique:\n{critique}\nTools:{population.tool_effectivity}")
    population.update_tool_effectivity("reflective", score)
    return (str(prompt_obj), score)

def iterative(hint: str = "") -> tuple[str, float]:
    """
    Task another component to create a new prompt based on several prompt+score examples.
    The examples are chosen automatically using roulette selection with the best prompts having the biggest chance of being featured.
    You may provide a short hint.
    This tool returns the resulting prompt and its score.

    Args:
        hint (str): Specific instruction on how a new prompt should be constructed.

    Returns:
        tuple[str, float]: The resulting prompt and its score on the training batch.
    """
    if len(population) < 2:
        return lamarckian()
    module = dspy.ChainOfThought(signature=signatures.Iterative, temperature=TEMPERATURE)
    examples = [p.prompt_and_perf() for p in population.select(5)]
    prompt = module(old_prompts=examples, supervisor_hint=hint).prompt_proposal
    prompt_obj = Prompt(prompt, origin="iterative")
    score = population.extend_and_score(prompt_obj)
    logger.info(f"ITERATIVE generated prompt:\n {str(prompt_obj)}\nSCORE: {score}.\nExamples:\n{examples}\nTools:{population.tool_effectivity}")
    population.update_tool_effectivity("iterative", score)
    return (str(prompt_obj), score)

def crossover(hint: str = "") -> tuple[str, float]:
    """
    Task another component to create a new prompt using two diverse parent prompts.
    The prompts are chosen automatically.
    You may provide a short hint.
    This tool returns the resulting prompt and its score.

    Args:
        hint (str): Specific instruction on how a new prompt should be constructed.

    Returns:
        tuple[str, float]: The resulting prompt and its score on the training batch.
    """
    if len(population) < 2:
        return lamarckian()
    
    module = dspy.ChainOfThought(signature=signatures.Crossover, temperature=TEMPERATURE)
    best_quartile = population.quartile(1)
    prompt1 = random.choice(best_quartile) if len(best_quartile) > 1 else population.prompts[0]
    # prompt2 most distinct to prompt1
    prompt2 = sorted(population, key= lambda p: Levenshtein.distance(prompt1.text, p.text))[-1]
    prompts = [prompt1, prompt2]
    random.shuffle(prompts)
    prompt = module(prompts=tuple(prompts), supervisor_hint=hint).prompt_proposal
    prompt_obj = Prompt(prompt, origin="crossover")
    score = population.extend_and_score(prompt_obj)
    logger.info(f"CROSSOVER generated prompt:\n {str(prompt_obj)}\n SCORE: {score}\nfrom prompt1:\n{prompts[0]}\n and from prompt2:\n{prompts[1]}\nTools:{population.tool_effectivity}.")
    population.update_tool_effectivity("crossover", score)
    return (str(prompt_obj), score)

## INSIGHT TOOLS 

def ask_analyst(question: str = "Can we optimize further or should I finish?") -> str:
    """
    Task your analyst to look at relevant data and use it to answer a question.
    Here are some examples of what you could ask:
        "Can we optimize further or should I finish?"
        "Are the prompts diverse enough?"
        "What's the most common problem in the prompts?"
        "What reasoning errors do some prompts promote?"
        "Is my population size optimal?"
        "Which operations create the best prompts?"

        Args:
            question (str): Your question relating to the optimization process. 

        Returns:
            str: Your analyst's answer.
    """
    module = dspy.ChainOfThought(signature=signatures.Analyst, temperature=TEMPERATURE)
    bad = population[-1].get_completion(0)
    good = population[0].get_completion(1)
    context = {
        "population size": len(population),
        "score": "Avg: {}, Max: {}".format(*population.stats()),
        "best prompts": "\n".join([f"{i+1}:\n{str(p)}\nwith score {p.get_dev()} was created by operation '{p.origin}'" for i,p in enumerate(population.quartile(1))]),
        "worst prompts": "\n".join([f"{i+1}:\n{str(p)}\nwith score {p.get_dev()} was created by operation '{p.origin}'" for i,p in enumerate(population.quartile(4))]),
        "solution by a good prompt": f"Good prompt:\n{population[0]}\non problem:\n{good[0]}\ngenerated reasoning\n{good[1]}",
        "solution by a bad prompt": f"Bad prompt:\n{population[-1]}\non problem:\n{bad[0]}\ngenerated reasoning\n{bad[1]}",
        "prompt creation tools statistics": "\n".join([f"Tool name {k} was used {v[0]} times with an average prompt score of {v[1]}" for k,v in population.tool_effectivity.items()])
    }
    answer = module(question=question, context=context).answer
    log_context = '\n'.join([f'{k}: {v}' for k,v in context.items()])
    logger.info(f"ANALYST question: {question}, answer: {answer}\ncontext: {log_context}")
    return answer
    

def peek_data() -> str:
    """
    Show some samples from the dataset in a formatted Q/A template.

    Args:
        None

    Returns:
        str: Examples of the data
    """
    logger.info("Peeking data")
    return str(train)

def peek_pop() -> str:
    """
    Show the total number of prompts in population and 3 random samples with their scores.

    Args:
        None

    Returns:
        str: Examples of the population
    """
    n = len(population)
    formatter = lambda p: f"Prompt:\n```{p[0]}\n```" + f"\nhas score {p[1]}" if p[1] >= 0 else "\nhasn't been scored yet"
    samples_text = '\n'.join(map(formatter, [p.prompt_and_perf() for p in random.sample(population.prompts, 3)]))
    logger.info("Peeking pop")
    return f"Population has {n} prompts\nSamples:\n{samples_text}"

## POPULATION CONTROL TOOLS

def purge_worst() -> None:
    """
    Remove the worst quarter (1/4) of prompts from the population.
    Begins new generation.

    Args:
        None

    Returns:
        None
    """
    population.dump()
    for i in range(len(population)//4):
        logger.info(f"PURGE WORST ({i}): {population.prompts[-1].text}")
        population.prompts.pop()
    population.current_gen += 1

def purge_duplicates() -> None:
    """
    After sorting by score, go prompt by prompt and remove the most similar prompt until a quarter (1/4) of the population is deleted.
    Begins new generation.

    Args:
        None

    Returns:
        None
    """
    population.dump()
    for i in range(len(population)//4):
        curr = population[i]
        most_similar = sorted(population.prompts[i+1:], key=lambda p: Levenshtein.distance(curr.text, p.text))[0]
        most_similar_ix = population.prompts.index(most_similar)
        population.prompts.pop(most_similar_ix)
        logger.info(f"PURGE DUPLICATES ({i}): {most_similar.text}")
    population.current_gen += 1

tools = [
    lamarckian,
    reflective,
    iterative,
    crossover,
    ask_analyst,
    peek_data,
    peek_pop,
    purge_worst,
    purge_duplicates
]