# Evaluating Turkish BERT for Semantic Similarity in Question Answering

## Overview

This project evaluates the performance of the `dbmdz/bert-base-turkish-cased` model on a Turkish question-answering task. The primary goal is to assess the model's ability to understand semantic similarity by generating embeddings for questions and their corresponding answers. The evaluation is based on retrieving the correct answer from a pool of candidates using cosine similarity between the embeddings.

The performance is measured by top-1 and top-5 success rates. Additionally, the project includes t-SNE visualizations to explore the separability of question and answer embeddings in a lower-dimensional space.

## Dataset

The project utilizes the `merve/turkish_instructions` dataset from the Hugging Face Datasets library. This dataset consists of instructional prompts, inputs, and corresponding outputs in Turkish.

For this task, a random sample of 1000 instruction-input pairs (as questions) and their corresponding outputs (as answers) are selected from the original dataset.

## Methodology

The project follows these steps:

1.  **Data Loading and Preprocessing**: The `merve/turkish_instructions` dataset is loaded. A random sample of 1000 question-answer pairs is created by combining the 'talimat' (instruction) and 'giriş' (input) fields to form the questions, and using the 'çıktı' (output) field for the answers. This subset is then saved to a CSV file (`selected_questions_answers.csv`) for consistent use.
2.  **Model Selection**: The pre-trained `dbmdz/bert-base-turkish-cased` model is used for generating text embeddings. This model is a cased BERT model specifically trained on a large corpus of Turkish text.
3.  **Embedding Generation**: Vector representations (embeddings) of the questions and answers are generated using the selected BERT model. The embeddings are derived from the last hidden state of the model's output, specifically the representation of the `[CLS]` token.
4.  **Similarity Calculation**: The cosine similarity between the embeddings of a question and all candidate answers is calculated. Cosine similarity measures the cosine of the angle between two non-zero vectors, providing a score of how similar the texts are in meaning.
5.  **Evaluation Metrics**: The model's performance is evaluated based on the following metrics:
    * **Top-1 Success Rate**: The percentage of questions for which the correct answer is the most similar (highest cosine similarity).
    * **Top-5 Success Rate**: The percentage of questions for which the correct answer is among the top 5 most similar answers.
6.  **Visualization**: t-SNE (t-Distributed Stochastic Neighbor Embedding) is used to visualize the high-dimensional question and answer embeddings in a 2D space. A scatter plot is generated to visually inspect the clustering and separability of questions and answers.

## Code Description

The Jupyter Notebook `ana_kod_tugcantopaloglu.ipynb` contains the complete code for this project. The key functions are:

-   **`get_model_and_tokenizer(model_name)`**: Loads the pre-trained model and its corresponding tokenizer from the Hugging Face Hub and moves the model to the GPU ("cuda") for faster computation.
-   **`get_embeddings(texts, tokenizer, model)`**: Takes a list of texts and returns their embeddings.
-   **`get_embeddings_in_batches(texts, tokenizer, model, batch_size=64)`**: A batch-processing version of `get_embeddings` for more efficient embedding generation of a large number of texts.
-   **`calculate_similarity(embedding1, embedding2)`**: Computes the cosine similarity between two embeddings.
-   **`calculate_top_k_success(questions, answers, tokenizer, model, k=5)`**: Calculates the top-1 and top-5 success rates for the given model on the provided questions and answers.
-   **`plot_tsne(embeddings, labels, title)`**: Applies t-SNE to the embeddings and generates a 2D scatter plot.

## How to Run

1.  **Environment Setup**: Ensure you have Python 3 and a GPU-enabled environment for optimal performance.
2.  **Install Dependencies**:
    ```bash
    pip install datasets transformers torch scikit-learn matplotlib pandas
    ```
3.  **Execute the Notebook**: Run the cells in the `ana_kod_tugcantopaloglu.ipynb` Jupyter Notebook sequentially.

## Results

The `dbmdz/bert-base-turkish-cased` model was evaluated on the 1000 sampled question-answer pairs. The achieved success rates are:

-   **Top-1 Success Rate**: 3.50%
-   **Top-5 Success Rate**: 4.90%

The t-SNE visualization shows the distribution of question and answer embeddings in a 2D space, providing a qualitative assessment of their semantic separability.

## Future Work

As mentioned in the notebook, the initial plan was to evaluate several other models for this task. The code is structured to be easily adaptable for different models. The next steps would involve running the evaluation in parallel for the following models to compare their performance:

-   `KaLM-embedding-multilingual-mini-v1`
-   `pingkeest/learning2_model`
-   `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
-   `Alibaba-NLP/gte-multilingual-base`
-   `Alibaba-NLP/gte-large-en-v1.5`

Additionally, the notebook has a variation for predicting the question from the answer, which can be another interesting direction for future experiments.
