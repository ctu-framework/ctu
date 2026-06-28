from src.data_loader import load_dataset
from src.agent import Agent
from src.engines.no_constraint_engine import NoConstraintEngine
from src.engines.json_constraint_engine import JSONConstraintEngine
from src.engines.kg_constraint_engine import KGConstraintEngine
from src.controller import Controller
from src.evaluator import Evaluator

import os
import json
from tqdm import tqdm

# Parallelism
from concurrent.futures import ThreadPoolExecutor


os.makedirs("results", exist_ok=True)


import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


mode = config["mode"]

dataset_path = config["execution"][mode]["dataset_path"]
max_steps = config["execution"][mode]["max_steps"]
engine_types = config["execution"]["constraint_engines"]
max_workers = config["execution"]["max_workers"]


# engine capabilities (desired)
enforce = config["execution"]["capabilities"]["enforce"]
feedback = config["execution"]["capabilities"]["feedback"]
guidance = config["execution"]["capabilities"]["guidance"]



# helper function
def resolve_capabilities(engine, enforce, feedback, guidance):
    engine.enforce = engine.enforces_constraints and enforce
    engine.feedback = engine.provides_feedback and feedback
    engine.guidance = engine.provides_guidance and guidance





def run_example(i, example, agent, engine_type):


    query = example["query"]
    tools = example["tools"]
    constraints = example["constraints"]


    # Select engine
    if engine_type == "no":
        engine = NoConstraintEngine(constraints)
    elif engine_type == "json":
        engine = JSONConstraintEngine(constraints)
    elif engine_type == "kg":
        engine = KGConstraintEngine(tools, constraints)

    # Resolve capabilities for ablation study
    resolve_capabilities(engine, enforce, feedback, guidance)

    # Initialize controller
    controller = Controller(agent, engine, max_steps)

    # Run pipeline
    result = controller.run(query, tools)

    # -------------------------------
    # Output result
    # -------------------------------
    status = result["status"]

    history = result.get("history", [])
    trace = result.get("trace", [])


    return result












def run():

    # Load dataset
    dataset = load_dataset(dataset_path)

    for engine_type in engine_types:

        # Initialize components
        agent = Agent(
            api_url=config["execution"]["api_url"],
            model=config["execution"]["model"],
            temperature=config["execution"]["temperature"],
            num_ctx=config["execution"]["num_ctx"]
        )

        evaluator = Evaluator()

        all_results = []




        # Parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            all_results = list(tqdm(
                pool.map(lambda x: run_example(*x, agent, engine_type),enumerate(dataset)),
                total=len(dataset),
                desc=f"Running Evaluation (engine: {engine_type})"
            ))

        # Update evaluator
        for r in all_results:
            evaluator.update(r)

        # Save results
        with open(f"results/run_results_{engine_type}_{mode}.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Save evaluation summary
        summary = evaluator.summarize()

        with open(f"results/evaluation_{engine_type}_{mode}.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run()