import dspy

class Lamarckian(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Given several input/output examples, design a suitable zero-shot prompt for a Large Language Model for that task.
    """
    examples: str = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prefix_prompt: str = dspy.OutputField()
    suffix_prompt: str = dspy.OutputField()

class Reflective(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Analyze a prompt and its suboptimal performance on a task sample along with its generated reasoning chain.
    Identify weak points and flaws and the prompt and generate a critique.
    Propose a new improved prompt based on the critique. 
    """
    prompt: str = dspy.InputField()
    task: dspy.Example = dspy.InputField()
    reasoning: str = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prompt_critique: str = dspy.OutputField()
    prompt_proposal: str = dspy.OutputField()

class Iterative(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Given several prompts and their respective scores, propose new prompt.
    """
    old_prompts: list[tuple[str,float]] = dspy.InputField()
    supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Crossover(dspy.Signature):
    """
    Your supervisor tasked you with generating a prompt for a Large Language Model.
    Given two prompts, try to combine them in a fitting way to create a better prompt as a offspring.
    """
    prompts: tuple[str, str] = dspy.InpuField()
    supervisor_hint: str = dspy.InputField()
    prompt_proposal: str = dspy.OutputField()

class Analyst(dspy.Signature):
    """
    You are an analyst in a prompt optimization process.
    Your supervisor asked you a question and you will answer it concisely based on the provided context.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField()

class OptimizationSuccess(dspy.Signature):
    """Guide a prompt optimization procedure to achieve the best possible result."""
    introduction: str = dspy.InputField(desc="Optimization Director metaprompt.")
    result: bool = dspy.OutputField(desc="Was the optimization succesful?")