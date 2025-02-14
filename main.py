import os
import time
import json
import dspy
from dotenv import load_dotenv
if __name__ == "__main__":
    # env setup

    stamp = round(time.time() % 31536000)
    folder = f"runs/{stamp}"
    os.environ['RUN_FOLDER'] = folder
    
    if os.path.exists(folder):
        with open(folder+'/prompts.jsonl', 'r') as f:
            prompts = [json.loads(l) for l in f.readlines()]
            initial_population = [Prompt.from_json(p) for p in prompts]
    else:
        initial_population = []
        os.mkdir(folder)

    os.environ['SOLVE_LM'] = "gpt-4o-mini"
    import signatures
    import tools
    from prompt import Prompt   
    from population import population

    
    vllm_port = os.getenv("VLLM_MY_PORT")
    if vllm_port:
        # self hosted model on cluster
        optim_lm = dspy.LM("hosted_vllm/ibnzterrell/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4", api_base=f"http://localhost:{vllm_port}/v1", api_key="EMPTY", cache=False)
    else:
        load_dotenv()
        api_key = os.getenv("API_KEY")
        optim_lm = dspy.LM("gpt-4o-mini", api_key=api_key, cache=False)
    dspy.configure(lm=optim_lm)

    
    INITIAL_POPULATION_SIZE = 10
    for _ in range(INITIAL_POPULATION_SIZE):
        tools.lamarckian()
        
    metaprompt = """You are the directing component in a prompt optimization procedure. 
    Use the provided tools to get information about the underlying task, inspect and modify the prompt population and check remaining budget.
    You are the manager of this project so delegate as much of the tasks to your subordinates using the provided tools."""
    director = dspy.ReAct(signature=signatures.OptimizationSuccess, tools=tools.tools, max_iters=100)
    res = director(introduction=metaprompt)
    t = res.trajectory
    print(t)
    with open(f"{folder}/trajectory.json", "w+") as f:
        json.dump(t, f, indent=4)
    population.dump()
    #for _ in range(3):
    #    for t in tools:
    #        t()