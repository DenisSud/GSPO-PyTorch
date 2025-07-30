# LLM-Teacher Competition: Mathematical Problem Solving

## Overview

This is a machine learning competition focused on developing systems that can automatically solve mathematical problems. The goal is to create a model that takes text-based mathematical problems as input and returns accurate numerical or symbolic answers.

## Competition Structure

The competition consists of multiple stages with cumulative scoring:
- **Stage 1**: Problem solving and answer derivation (described below)
- **Stage 2**: Semi-final (top 10 participants)
- **Stage 3**: Final (top 3 participants)

Rankings are determined by cumulative points across all stages.

## Task Description

### Objective
Develop a model that can automatically solve mathematical problems presented as text and return the final answer in numerical or symbolic form with maximum accuracy.

### Data Format

**Training Data** (`train.csv`):
- 1,000 mathematical problems with answers from open benchmarks
- Multiple languages supported
- Format: `task` (problem text), `answer` (correct answer in brackets)

**Test Data**:
- `test_public.csv`: 200 problems for public leaderboard
- `test_private.csv`: 500 problems for private leaderboard (released 48 hours before deadline)
- **Important**: All private test problems are in Russian

**Sample Data**:
```csv
task,answer
"The value of $y$ varies inversely as $\sqrt x$ and when $x=24$, $y=15$. What is $x$ when $y=3$?",[600]
"Calculate (4/(-8)*(-32)/4)/(115/(-5)), please return the calculation result in its simplest fraction form.",[-4/23]
"What is the product of 56397 and -21.6?",[-1218175.2]
```

### Submission Format
- `submission.csv`: Predicted answers in CSV format
- `model.ipynb`: Complete solution notebook (must be reproducible)

## Evaluation Criteria

### 1. Accuracy (80 points)
- **Metric**: Accuracy (proportion of correctly solved problems)
- **Scoring**: Relative scale between baseline and best solution
- **Formula**:

$$\text{Points} = \max\left(0, \frac{\text{score}_{\text{baseline}} - \text{score}}{\text{score}_{\text{baseline}} - \text{score}_{\text{winner}}} \cdot 100\right)$$

Where:
- $\text{score}_{\text{baseline}}$ = baseline metric value
- $\text{score}$ = your model metric value
- $\text{score}_{\text{winner}}$ = best leaderboard value

### 2. Solution Quality (10 points)
**Mandatory reproducibility requirement**:
- Must provide `model.ipynb` that runs end-to-end without errors
- Final result must match submission within ±1% accuracy
- **Critical**: Non-reproducible solutions are disqualified regardless of accuracy

**Scoring**:
- 10 points: Notebook is reproducible and generates correct results
- 0 points: Notebook fails to reproduce or results don't match

### 3. Solution Logic (10 points)
**Requirements**:
- Clear description of solution logic in first cell of `model.ipynb`
- Should explain how the solution works step-by-step
- Can use schemes, lists, coherent text, graphics, or flowcharts
- LLM-generated descriptions allowed but must be edited for clarity

**Scoring**:
- 10 points: Logic described clearly and step-by-step with explanations
- 5 points: Description present but incomplete/unstructured/superficial
- 0 points: No description provided

**Total Score**: Sum of all three criteria (maximum 100 points)

## Technical Requirements

### Model Constraints
- **Maximum model size**: 8 billion parameters
- **Independence**: Each test example must be processed independently
- **No test data leakage**: Prohibited to use test data for training or hyperparameter tuning

### Allowed Approaches
- ✅ Any open-source models (including retrained on open datasets)
- ✅ LLM-as-a-judge, RAG, MCP approaches
- ✅ Multiple models in pipeline (each must comply with rules)
- ✅ API calls (must provide keys/reproduction instructions)
- ❌ Internet fact checking prohibited

### Reproducibility Requirements
- All retraining procedures must be reproducible
- Retraining should produce comparable results
- Complete solution pipeline must be documented

## Key Considerations

1. **Language Handling**: Private test is entirely in Russian - ensure model handles Russian mathematical notation and terminology
2. **Answer Format**: Answers must be in exact format `[value]` where value can be integers, decimals, or fractions
3. **Mathematical Accuracy**: Focus on mathematical reasoning accuracy rather than just text processing
4. **Reproducibility**: This is critical - ensure your notebook runs completely from scratch

## Success Strategy

Given your ML engineering background, consider:
- **Model Selection**: Choose models with strong mathematical reasoning capabilities
- **Fine-tuning**: Consider fine-tuning on mathematical problem datasets
- **Pipeline Design**: Design robust preprocessing and post-processing for answer extraction
- **Validation**: Implement thorough validation to ensure reproducibility
- **Documentation**: Clearly document your approach for the logic evaluation criterion

The competition emphasizes both technical performance and engineering best practices, making reproducibility and clear documentation as important as raw accuracy.
