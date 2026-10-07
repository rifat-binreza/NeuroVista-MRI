"""Run from the repo root: python -m examples.inspect_plan. No data needed."""
from neurovista.experiments import experiment_plan
from neurovista.lora import generation_deficit
for row in experiment_plan():
    print(f"{row.name:35} real={sum(row.real_counts.values()):4} synthetic={sum(row.synthetic_counts.values()):3}")
print("Example minority-class deficit:", generation_deficit(250))
