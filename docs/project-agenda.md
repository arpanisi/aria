# Project Agenda

## Goal

Build an open-weight system that can recommend the next step in a statistical analysis workflow from the previous analysis steps, the available data context, and a constrained set of analysis tools.

## Step 1: Build a Small Statistical Analysis Knowledge Graph

Create a compact knowledge graph from 10-50 statistics papers.

### Document Steps

1. Select 10-50 statistics papers that contain clear end-to-end analysis workflows.
2. For each paper, extract the core analysis structure:
   - hypotheses
   - methods
   - inputs and datasets
   - outputs
   - statistical tests
   - evaluations
   - assumptions
   - limitations
3. Normalize the extracted information into graph entities and relationships.
4. Define the graph schema.
5. Encode paper-level workflows as ordered analysis steps.
6. Add a tool layer for data analysis operations borrowed from scikit-learn and related statistical tooling.
7. Link each workflow step to the tools that could plausibly execute or support it.
8. Validate the graph by checking whether paper workflows can be reconstructed from the encoded nodes and edges.

### Initial Graph Concepts

- `Paper`
- `Dataset`
- `Hypothesis`
- `Method`
- `AnalysisStep`
- `StatisticalTest`
- `Evaluation`
- `Output`
- `Assumption`
- `Limitation`
- `Tool`

### Initial Relationship Types

- `paper_proposes_hypothesis`
- `paper_uses_dataset`
- `hypothesis_tested_by_method`
- `method_contains_step`
- `step_uses_tool`
- `step_produces_output`
- `output_evaluated_by`
- `test_supports_hypothesis`
- `method_requires_assumption`
- `result_has_limitation`
- `step_follows_step`

### Tool Coverage

Start with common data analysis tools and operations inspired by scikit-learn:

- preprocessing
- feature extraction
- dimensionality reduction
- model fitting
- clustering
- classification
- regression
- cross-validation
- model selection
- metrics and scoring
- pipelines

## Step 2: Train a Model to Predict the Next Analysis Step

Fine-tune and then apply GRPO to an open-weight model so it can predict the next statistical analysis step from prior workflow context.

### Document Steps

1. Convert knowledge graph workflows into training examples.
2. Represent each example as:
   - previous analysis steps
   - available hypotheses
   - dataset context
   - available tools
   - expected next analysis step
3. Fine-tune an open-weight base model on next-step prediction.
4. Define reward functions for GRPO.
5. Reward predictions that are:
   - statistically coherent
   - compatible with available tools
   - consistent with the previous workflow
   - aligned with the paper-derived target step
   - explicit about assumptions and evaluation criteria
6. Penalize predictions that:
   - introduce unsupported methods
   - ignore available tools
   - skip necessary validation
   - mismatch the stated hypothesis or data type
7. Evaluate the model on held-out paper workflows.
8. Compare model predictions against expert-authored or paper-derived next steps.

## Near-Term Milestones

1. Define the first version of the knowledge graph schema.
2. Select the initial paper set.
3. Create an extraction template for paper annotation.
4. Build the first graph from 10 papers.
5. Create the first next-step prediction dataset.
6. Run a baseline model before fine-tuning.
7. Fine-tune the first open-weight model.
8. Design and test the first GRPO reward functions.
