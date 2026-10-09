"""
Training Data Generator using OpenAI API (Alternative)
=======================================================

Alternative to generate_training_data.py using OpenAI API instead of Gemini.
Falls back to template-based generation if no API key is provided.
"""

import argparse
import csv
import random
import time
import os
import sys
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Import shared resources from main generator
from generate_training_data import (
    LANGUAGES, CEFR_LEVELS, CEFR_DESCRIPTIONS,
    SCORE_DISTRIBUTIONS, SCENARIOS, NAMES, COMPANIES, TOPICS,
    TEMPLATES, generate_scores, add_errors_for_level,
    generate_fallback_data
)


def generate_with_openai(api_key: str, samples_per_level: int) -> List[Dict]:
    """Generate training data using OpenAI API."""
    if not HAS_OPENAI:
        print("openai package is not installed. Run: pip install openai")
        print("Falling back to template generation.")
        return generate_fallback_data(max(100, samples_per_level))
    
    client = OpenAI(api_key=api_key)
    
    data = []
    q_id_counter = 1
    total = len(CEFR_LEVELS) * samples_per_level
    
    print(f"Generating {total} samples using OpenAI API...")
    
    for cefr in CEFR_LEVELS:
        pbar = tqdm(range(samples_per_level), desc=f"CEFR {cefr}")
        for _ in pbar:
            lang = random.choice(LANGUAGES)
            grammar, content = generate_scores(cefr)
            scenario_idx = random.randint(0, len(SCENARIOS[lang]) - 1)
            scenario = SCENARIOS[lang][scenario_idx]
            
            prompt = f"""Write a realistic work/business email in {lang} language.

Target CEFR proficiency level: {cefr}
Level description: {CEFR_DESCRIPTIONS[cefr]}
Target grammar quality (0=terrible, 5=perfect): {grammar}
Target content completeness (0=empty/off-topic, 4=complete): {content}

Context/scenario: {scenario}

IMPORTANT INSTRUCTIONS:
- Write the ENTIRE email in {lang} language only
- Match the proficiency level exactly
- If A1/A2: use SIMPLE vocabulary, make grammatical mistakes, use short sentences
- If C1/C2: use sophisticated vocabulary, complex sentences, professional tone
- If grammar score is low, deliberately include grammatical errors
- If content score is low, make the email brief/vague
- Include proper email structure (greeting, body, closing)
- Length: 100-500 words
- Return ONLY the email text"""
            
            retries = 3
            success = False
            while retries > 0 and not success:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert multilingual email writer who can produce text at any CEFR proficiency level."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                        temperature=0.8,
                    )
                    text = response.choices[0].message.content.strip()
                    if len(text) > 50:
                        success = True
                        data.append({
                            "questionID": q_id_counter,
                            "questionStatement": scenario,
                            "text": text,
                            "grammar": grammar,
                            "content": content,
                            "cefr": cefr,
                            "language": lang
                        })
                        q_id_counter += 1
                    else:
                        retries -= 1
                    time.sleep(0.3)
                except Exception as e:
                    pbar.set_postfix({"error": str(e)[:30]})
                    retries -= 1
                    time.sleep(2)
            
            if not success:
                # Fallback
                templates = TEMPLATES[lang][cefr]
                template = random.choice(templates)
                name = random.choice(NAMES[lang])
                text = template.format(
                    name=name, name_short=name.split()[0],
                    company=random.choice(COMPANIES),
                    topic=random.choice(TOPICS[lang])
                )
                data.append({
                    "questionID": q_id_counter,
                    "questionStatement": scenario,
                    "text": text,
                    "grammar": grammar,
                    "content": content,
                    "cefr": cefr,
                    "language": lang
                })
                q_id_counter += 1
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using OpenAI API"
    )
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key")
    parser.add_argument("--output", type=str, default="training_data.csv")
    parser.add_argument("--samples-per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    if args.api_key:
        data = generate_with_openai(args.api_key, args.samples_per_level)
    else:
        print("No API key provided. Using template-based generation.")
        data = generate_fallback_data(args.samples_per_level)
    
    random.shuffle(data)
    
    print(f"\nWriting {len(data)} samples to {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["questionID", "questionStatement", "text", "grammar", "content", "cefr", "language"]
        )
        writer.writeheader()
        writer.writerows(data)
    
    print("Done!")


if __name__ == "__main__":
    main()
