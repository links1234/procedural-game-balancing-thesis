# Project Summary

## Problem

Procedural content generators can produce large numbers of game artifacts, but it is difficult to know whether those artifacts are balanced, stable, or useful without testing them in a game-like environment.

## Approach

This thesis used a deterministic Super Auto Pets simulation backend to evaluate generated artifacts through repeatable experiments. The pipeline was designed around reproducible seeds, validated actions, generated-content injection, and config-driven experiment runs.

## Evaluation

Generated artifacts were compared using simulator-backed metrics:

- Fairness: whether generated content avoids giving one side an unreasonable advantage.
- Diversity: whether generated content creates meaningful variation instead of repeating similar artifacts.
- Stability: whether results remain consistent across seeds and repeated evaluations.
- Safety: whether generated artifacts avoid invalid, broken, or degenerate game states.

## Outcome

The system produced repeatable experiment outputs and automated analysis artifacts that could be converted into thesis-ready tables and figures.
