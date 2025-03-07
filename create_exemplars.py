import json
import os
import random
import argparse
from tqdm import tqdm

def create_exemplars_from_problems(data_path, output_path, num_examples=100, seed=42):
    """
    Extract exemplars from the problems.json file in ScienceQA format.
    
    Args:
        data_path: Path to the problems.json file
        output_path: Path to save the exemplars JSON file
        num_examples: Number of exemplars to extract
        seed: Random seed for reproducibility
    """
    print(f"Loading data from {data_path}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the dataset
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded data with {len(data)} examples")
    
    # Process and filter examples
    candidates = []
    
    for key, item in tqdm(data.items(), desc="Processing examples"):
        # Extract fields
        question = item.get('question', '')
        choices = item.get('choices', [])
        answer_idx = item.get('answer', 0)  # Get answer index
        solution = item.get('solution', '')
        lecture = item.get('lecture', '')
        
        # Skip examples without solutions or with very short solutions
        if not solution or len(solution.split()) < 10:
            continue
        
        # Format choices as text
        choices_text = ""
        for i, choice in enumerate(choices):
            choices_text += f"({chr(65+i)}) {choice} "
        
        # Get the answer text
        answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
        
        # Combine question and choices
        full_question = f"Question: {question}\nChoices: {choices_text}"
        
        # Add lecture content if available
        if lecture and len(lecture.strip()) > 0:
            full_question = f"Context: {lecture}\n{full_question}"
        
        # Extract reasoning steps
        steps = extract_reasoning_steps(solution)
        
        candidates.append({
            "id": key,
            "question": full_question,
            "steps": steps,
            "answer": answer,
            "raw_solution": solution
        })
    
    print(f"Found {len(candidates)} candidates with solutions")
    
    # Select examples
    if len(candidates) <= num_examples:
        selected = candidates
        print(f"Using all {len(selected)} available candidates")
    else:
        selected = random.sample(candidates, num_examples)
        print(f"Randomly selected {len(selected)} exemplars")
    
    # Create the final exemplars list
    exemplars = []
    for example in selected:
        exemplars.append({
            "question": example["question"],
            "steps": example["steps"],
            "answer": example["answer"]
        })
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(exemplars, f, indent=2)
    
    print(f"Created exemplar file at {output_path}")
    return exemplars

def extract_reasoning_steps(solution):
    """Extract reasoning steps from solution text"""
    import re
    
    # Method 1: Split by sentences assuming each is a step
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', solution)
    steps = [s.strip() for s in sentences if s.strip()]
    
    # If only one step or no steps, use the whole solution
    if len(steps) <= 1:
        return [solution]
    
    return steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exemplars from ScienceQA problems.json")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the problems.json file")
    parser.add_argument("--output_path", type=str, default="exemplars.json", help="Path to save the exemplars JSON file")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of exemplars to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_exemplars_from_problems(
        args.data_path,
        args.output_path,
        args.num_examples,
        args.seed
    )