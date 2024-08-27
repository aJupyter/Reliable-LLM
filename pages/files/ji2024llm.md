# LLM Internal States Reveal Hallucination Risk Faced With a Query

[[Link](https://arxiv.org/abs/2407.03282)]

## Motivation

Whether LLMs can estimate their own hallucination risk before response generation.

Previous works examine texts not exclusively produced by the same LLMs whose internal states are analyzed, highlighting the necessity for further investigation into the LLM self-awareness and how their internal states correlate with their uncertainty and own hallucination occurrence.

**Insights**:

1. Whether they have seen the query in training data;
2. Whether models are likely to hallucinate regarding the query.

## Method

- Cnstruct datasets focusing on two dimensions: (1) the distinction between queries seen and unseen in the training data; (2) the likelihood of hallucination risk faced with the queries.
- Visualize the neurons for perception extracted from a specified LLM layer and then leverage the probing classifier technique.

### Problem Formulation

- The selfawareness of LLM, specifically how their internal states I relate to their level of hallucination risk h when faced with a query q.
- Hallucination risk label, which is determined based on (1) the query’s presence in the training data or (2) the degree of hallucination in the response r to q.

### Data Construction

#### Seen/Unseen Query in Training Data

- Hallucinations triggered by unseen queries are data-related.
- Seen group: BBC 2020; Unseen group: BBC 2024.

#### Hallucination Risk faced with the Query

- Construct data using LLM to directly generate responses to queries in diverse NLG task
- Evaluate by integrating three metrics comprehensively: Rouge-L, NLI, Questeval (QA-based metric for evaluating the faithfulness of the output in generation tasks).

### Preliminary Analysis: Neurons for Hallucination Perception from Internal States

- To analyze the self-assess sense of internal states and the role of specific neurons in the uncertainty and hallucination estimation, employ a feature selection method based on Mutual Information.
- There exist individual neurons within LLM that can fairly perceive uncertainty and predict future hallucinations.

### Internal State-based Estimator

- Use internal states corresponding to the last token of queries, taken from a specified layer within the LLM, serve as the input to estimation model.

## Experiment

### Tow Proposed Metrics

- F1, Accuracy.

### Datasets

- Seen/Unseen Query in Training Data
- Hallucination Risk faced with the Query

### Models

- Llama2-7B, Mistral-7B.

### Baselines

- Zero-shot Prompt
- In-Context-Learning (ICL) Prompt
- Perplexity (PPL)

## Results and Analysis

- The model can distinguish answerable and unanswerable questions that include future information.
- It performs less effectively in the translation task (F1 and ACC 76.90%) while excelling in the Number Conversion task (F1 94.04%, ACC 96.00%). Zero-shot prompt and ICL yield similar results.
- Layer Depth Positively Correlates with its Prediction Performance.
- Consistency of Internal States across Different LLMs.
- Internal States Share Features inner-task but do not Cross-task.
- Internal State as an Efficient Hallucination Estimator.
