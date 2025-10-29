"""
Advanced Deep Learning Sentiment Analysis using RoBERTa and BERT
For One NZ App Review Analysis

This notebook provides three approaches:
1. Pre-trained RoBERTa (Twitter-optimized)
2. Fine-tuned BERT on your dataset
3. Ensemble approach combining all methods
"""

# ============================================================================
# SETUP AND INSTALLATIONS
# ============================================================================

# Run this cell first in Kaggle
# !pip install transformers torch accelerate -q

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# METHOD 1: PRE-TRAINED ROBERTA SENTIMENT ANALYSIS
# ============================================================================

def roberta_sentiment_analysis(df, text_column='content_clean',
                               batch_size=16, max_samples=None):
    """
    Apply pre-trained RoBERTa model for sentiment analysis
    """
    print("\n" + "=" * 80)
    print("METHOD 1: PRE-TRAINED ROBERTA SENTIMENT ANALYSIS")
    print("=" * 80)

    # Load model
    print("\nLoading RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
    device = 0 if torch.cuda.is_available() else -1

    # sentiment_model = pipeline(
    #     "sentiment-analysis",
    #     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    #     device=device,
    #     truncation=True,
    #     max_length=512
    # )

    # IMPORTANT: Work with the full dataframe, mark which rows to process
    df_result = df.copy()

    # Identify rows that can be processed (have text)
    processable_mask = df_result[text_column].notna()
    df_to_process = df_result[processable_mask].copy()

    if max_samples and len(df_to_process) > max_samples:
        print(f"\nProcessing {max_samples} samples (set max_samples=None for full dataset)...")
        # Sample but keep track of original indices
        df_to_process = df_to_process.sample(n=max_samples, random_state=42)
    else:
        print(f"\nProcessing {len(df_to_process)} reviews...")

    reviews = df_to_process[text_column].tolist()

    # Process in batches
    results = []
    for i in tqdm(range(0, len(reviews), batch_size), desc="Processing"):
        batch = reviews[i:i + batch_size]
        try:
            batch_results = sentiment_model(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"\nError in batch {i // batch_size}: {str(e)}")
            results.extend([{'label': 'neutral', 'score': 0.5}] * len(batch))

    # Map labels
    label_map = {
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        'POSITIVE': 'Positive',
        'NEGATIVE': 'Negative',
        'NEUTRAL': 'Neutral'
    }

    # Assign results back to the processed rows
    df_to_process['roberta_sentiment'] = [
        label_map.get(r['label'], 'Neutral') for r in results
    ]
    df_to_process['roberta_confidence'] = [r['score'] for r in results]

    # Initialize columns in full dataframe with defaults
    df_result['roberta_sentiment'] = 'Not Processed'
    df_result['roberta_confidence'] = 0.0

    # Update only the processed rows using index alignment
    df_result.loc[df_to_process.index, 'roberta_sentiment'] = df_to_process['roberta_sentiment']
    df_result.loc[df_to_process.index, 'roberta_confidence'] = df_to_process['roberta_confidence']

    # Validation
    if 'sentiment_label' in df_result.columns:
        # Only validate processed rows
        processed_df = df_result[df_result['roberta_sentiment'] != 'Not Processed']
        if len(processed_df) > 0:
            accuracy = (processed_df['roberta_sentiment'] == processed_df['sentiment_label']).mean()
            print(f"\nRoBERTa Accuracy vs Ground Truth: {accuracy * 100:.2f}%")

            print("\nClassification Report:")
            print(classification_report(
                processed_df['sentiment_label'],
                processed_df['roberta_sentiment']
            ))

    print("\nRoBERTa Sentiment Distribution:")
    print(df_result['roberta_sentiment'].value_counts())

    print("\nAverage Confidence Scores:")
    confidence_by_sentiment = df_result[df_result['roberta_sentiment'] != 'Not Processed'].groupby('roberta_sentiment')[
        'roberta_confidence'].mean()
    print(confidence_by_sentiment)

    return df_result


# ============================================================================
# METHOD 2: FINE-TUNED BERT MODEL
# ============================================================================

class SentimentDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for BERT fine-tuning
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_bert_data(df, text_column='content_clean', label_column='sentiment_label'):
    """
    Prepare data for BERT fine-tuning
    """
    print("\nPreparing data for BERT fine-tuning...")

    # Filter and clean
    df_bert = df[[text_column, label_column]].dropna()

    # Encode labels
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df_bert['label_encoded'] = df_bert[label_column].map(label_map)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_bert[text_column].values,
        df_bert['label_encoded'].values,
        test_size=0.2,
        random_state=42,
        stratify=df_bert['label_encoded']
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    return train_texts, val_texts, train_labels, val_labels, label_map


def fine_tune_bert(train_texts, train_labels, val_texts, val_labels,
                   epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Fine-tune BERT model on the dataset
    """
    print("\n" + "=" * 80)
    print("METHOD 2: FINE-TUNING BERT MODEL")
    print("=" * 80)

    # Load tokenizer and model
    print("\nLoading BERT base model...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    # Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    trainer.train()

    # Evaluate
    print("\nEvaluating on validation set...")
    results = trainer.evaluate()
    print(f"\nValidation Accuracy: {results['eval_accuracy'] * 100:.2f}%")

    # Save model
    model.save_pretrained('./fine_tuned_bert')
    tokenizer.save_pretrained('./fine_tuned_bert')
    print("\nModel saved to './fine_tuned_bert'")

    return model, tokenizer, results


def predict_with_finetuned_bert(df, model, tokenizer, text_column='content_clean'):
    """
    Make predictions using fine-tuned BERT model
    """
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS WITH FINE-TUNED BERT")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    df_pred = df[df[text_column].notna()].copy()
    texts = df_pred[text_column].tolist()

    predictions = []
    confidences = []

    print(f"\nPredicting {len(texts)} reviews...")

    for text in tqdm(texts):
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()

        predictions.append(pred)
        confidences.append(conf)

    # Decode predictions
    label_decode = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df_pred['bert_sentiment'] = [label_decode[p] for p in predictions]
    df_pred['bert_confidence'] = confidences

    print("\nBERT Sentiment Distribution:")
    print(df_pred['bert_sentiment'].value_counts())

    if 'sentiment_label' in df_pred.columns:
        accuracy = (df_pred['bert_sentiment'] == df_pred['sentiment_label']).mean()
        print(f"\nBERT Accuracy vs Ground Truth: {accuracy * 100:.2f}%")

    return df_pred


# ============================================================================
# METHOD 3: ENSEMBLE APPROACH
# ============================================================================

def ensemble_sentiment_analysis(df):
    """
    Combine VADER, TextBlob, RoBERTa, and BERT predictions
    """
    print("\n" + "=" * 80)
    print("METHOD 3: ENSEMBLE SENTIMENT ANALYSIS")
    print("=" * 80)

    # Check what sentiment columns are available
    available_methods = []

    if 'vader_sentiment' in df.columns:
        available_methods.append('vader_sentiment')
    if 'textblob_sentiment' in df.columns:
        available_methods.append('textblob_sentiment')
    if 'roberta_sentiment' in df.columns:
        available_methods.append('roberta_sentiment')
    if 'bert_sentiment' in df.columns:
        available_methods.append('bert_sentiment')

    print(f"\nAvailable sentiment methods: {available_methods}")

    if len(available_methods) < 2:
        print(f"\nWARNING: Only {len(available_methods)} sentiment method(s) available.")
        print("Ensemble requires at least 2 methods. Skipping ensemble analysis.")

        # If only one method, use it as ensemble
        if len(available_methods) == 1:
            df['ensemble_sentiment'] = df[available_methods[0]]
            df['ensemble_score'] = 0
        else:
            df['ensemble_sentiment'] = 'Not Available'
            df['ensemble_score'] = 0

        return df

    # Encode sentiments to numbers
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1,
                     'Not Processed': 0, 'Not Available': 0}

    # Encode all available methods
    for method in available_methods:
        encoded_col = method.replace('_sentiment', '_encoded')
        df[encoded_col] = df[method].map(sentiment_map).fillna(0)

    # Calculate weighted ensemble based on available methods
    print("\nCalculating ensemble scores...")

    # Define weights (prioritize deep learning if available)
    weights = {
        'vader_encoded': 0.15,
        'textblob_encoded': 0.15,
        'roberta_encoded': 0.35,
        'bert_encoded': 0.35
    }

    # Adjust weights based on available methods
    available_encoded = [m.replace('_sentiment', '_encoded') for m in available_methods]
    total_weight = sum(weights.get(m, 0) for m in available_encoded)

    # Normalize weights
    normalized_weights = {m: weights.get(m, 0) / total_weight for m in available_encoded}

    print(f"Ensemble weights: {normalized_weights}")

    # Calculate ensemble score
    df['ensemble_score'] = 0
    for method, weight in normalized_weights.items():
        if method in df.columns:
            df['ensemble_score'] += df[method] * weight

    # Classify based on ensemble score
    df['ensemble_sentiment'] = df['ensemble_score'].apply(
        lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
    )

    print("\nEnsemble Sentiment Distribution:")
    print(df['ensemble_sentiment'].value_counts())

    # Validation if ground truth available
    if 'sentiment_label' in df.columns:
        accuracy = (df['ensemble_sentiment'] == df['sentiment_label']).mean()
        print(f"\nEnsemble Accuracy vs Ground Truth: {accuracy * 100:.2f}%")

        print("\nComparison of All Available Methods:")
        comparison_data = {}

        for method in available_methods:
            method_accuracy = (df[method] == df['sentiment_label']).mean()
            comparison_data[method.replace('_sentiment', '').upper()] = [method_accuracy]

        comparison_data['ENSEMBLE'] = [accuracy]

        comparison = pd.DataFrame(comparison_data)
        print("\nAccuracy Comparison:")
        print(comparison.T.round(4))

    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sentiment_comparison(csv_file='one_nz_with_deep_learning_sentiment.csv'):
    """
    Compare all sentiment analysis methods by loading from CSV file
    """
    import os

    # Load data from CSV
    if not os.path.exists("C:\\Users\\270708326\\PycharmProjects\\PythonProject\\Sentiment\\one_nz_with_deep_learning_sentiment.csv"):
        print(f"\nERROR: File '{csv_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please ensure the file exists before running visualizations.")
        return

    print(f"\nLoading data from '{csv_file}'...")
    df = pd.read_csv("one_nz_with_deep_learning_sentiment.csv")
    print(f"Loaded {len(df):,} reviews")

    # Check available sentiment columns
    available_methods = []
    method_display = []

    if 'vader_sentiment' in df.columns:
        available_methods.append('vader_sentiment')
        method_display.append('VADER')
    if 'textblob_sentiment' in df.columns:
        available_methods.append('textblob_sentiment')
        method_display.append('TextBlob')
    if 'roberta_sentiment' in df.columns:
        available_methods.append('roberta_sentiment')
        method_display.append('RoBERTa')
    if 'bert_sentiment' in df.columns:
        available_methods.append('bert_sentiment')
        method_display.append('BERT')
    if 'ensemble_sentiment' in df.columns:
        available_methods.append('ensemble_sentiment')
        method_display.append('Ensemble')

    print(f"Available methods: {method_display}")

    if len(available_methods) == 0:
        print("\nNo sentiment analysis columns found in CSV!")
        return

    # Create comparison plot
    n_methods = len(available_methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    # axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, (method, display_name) in enumerate(zip(available_methods, method_display)):
        ax = axes[idx // 2, idx % 2]

        if 'sentiment_label' in df.columns:
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix

            # Filter out non-predictions
            df_valid = df[
                (df[method].notna()) &
                (~df[method].isin(['Not Processed', 'Not Available']))
                ].copy()

            if len(df_valid) > 0:
                cm = confusion_matrix(
                    df_valid['sentiment_label'],
                    df_valid[method],
                    labels=['Negative', 'Neutral', 'Positive']
                )

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Negative', 'Neutral', 'Positive'],
                            yticklabels=['Negative', 'Neutral', 'Positive'],
                            ax=ax, cbar=False)

                accuracy = (df_valid[method] == df_valid['sentiment_label']).mean()
                ax.set_title(f'{display_name}\nAccuracy: {accuracy * 100:.2f}%',
                             fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{display_name}\nNo valid predictions',
                        ha='center', va='center', transform=ax.transAxes)
        else:
            # Just show distribution
            df[method].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'{display_name} Distribution', fontsize=12, fontweight='bold')

        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    # Hide unused subplots
    for idx in range(len(available_methods), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('sentiment_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved 'sentiment_comparison.png'")
    plt.show()


def plot_confidence_distribution(csv_file='one_nz_with_deep_learning_sentiment.csv'):
    """
    Plot confidence score distributions for deep learning models from CSV
    """
    import os

    # Load data from CSV
    if not os.path.exists("C:\\Users\\270708326\\PycharmProjects\\PythonProject\\Sentiment\\one_nz_with_deep_learning_sentiment.csv"):
        print(f"\nERROR: File '{csv_file}' not found!")
        return

    print(f"\nLoading data from '{csv_file}'...")
    df = pd.read_csv("one_nz_with_deep_learning_sentiment.csv")

    # Check which confidence columns are available
    has_roberta = 'roberta_confidence' in df.columns and 'roberta_sentiment' in df.columns
    has_bert = 'bert_confidence' in df.columns and 'bert_sentiment' in df.columns

    if not has_roberta and not has_bert:
        print("\nNo confidence data found in CSV")
        return

    # Determine subplot layout
    n_plots = int(has_roberta) + int(has_bert)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_roberta:
        df_roberta = df[
            df['roberta_sentiment'].notna() &
            (~df['roberta_sentiment'].isin(['Not Processed', 'Not Available']))
            ].copy()

        if len(df_roberta) > 0:
            df_roberta.boxplot(
                column='roberta_confidence',
                by='roberta_sentiment',
                ax=axes[plot_idx]
            )
            axes[plot_idx].set_title('RoBERTa Confidence by Sentiment',
                                     fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Sentiment')
            axes[plot_idx].set_ylabel('Confidence Score')
            axes[plot_idx].get_figure().suptitle('')
            plt.sca(axes[plot_idx])
            plt.xticks(rotation=0)
            plot_idx += 1

    if has_bert:
        df_bert = df[
            df['bert_sentiment'].notna() &
            (~df['bert_sentiment'].isin(['Not Processed', 'Not Available']))
            ].copy()

        if len(df_bert) > 0:
            df_bert.boxplot(
                column='bert_confidence',
                by='bert_sentiment',
                ax=axes[plot_idx]
            )
            axes[plot_idx].set_title('BERT Confidence by Sentiment',
                                     fontsize=12, fontweight='bold')
            axes[plot_idx].set_xlabel('Sentiment')
            axes[plot_idx].set_ylabel('Confidence Score')
            axes[plot_idx].get_figure().suptitle('')
            plt.sca(axes[plot_idx])
            plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved 'confidence_distribution.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_complete_deep_learning_analysis(df, fine_tune=False):
    """
    Run complete deep learning sentiment analysis pipeline

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'content_clean' and 'sentiment_label' columns
    fine_tune : bool
        Whether to fine-tune BERT (takes longer but more accurate)
    """
    print("\n" + "=" * 80)
    print("COMPLETE DEEP LEARNING SENTIMENT ANALYSIS PIPELINE")
    print("=" * 80)

    # # Method 1: RoBERTa
    # df_roberta = roberta_sentiment_analysis(df, max_samples=None)
    #
    # # Merge results back
    # df = df.merge(
    #     df_roberta[['reviewId', 'roberta_sentiment', 'roberta_confidence']],
    #     on='reviewId',
    #     how='left'
    # )

    # Method 2: Fine-tuned BERT (optional)
    if fine_tune:
        print("\n" + "=" * 80)
        print("FINE-TUNING BERT MODEL")
        print("=" * 80)
        print("This will take 15-30 minutes depending on dataset size and GPU availability")
        print("Set fine_tune=False to skip this step and use only RoBERTa")

        train_texts, val_texts, train_labels, val_labels, label_map = prepare_bert_data(df)
        model, tokenizer, results = fine_tune_bert(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=3,
            batch_size=16
        )

        df_bert = predict_with_finetuned_bert(df, model, tokenizer)
        df = df.merge(
            df_bert[['reviewId', 'bert_sentiment', 'bert_confidence']],
            on='reviewId',
            how='left'
        )

        output_path = 'C:\\Users\\270708326\\PycharmProjects\\PythonProject\\Sentiment\\one_nz_with_deep_learning_sentiment.csv'
        # Check if the file exists and set the header flag accordingly
        header_flag = not os.path.exists(output_path)

        # Append to the CSV, writing the header only if the file is new
        df.to_csv(output_path, mode='a', index=False, header=header_flag)

    # Method 3: Ensemble
    df = ensemble_sentiment_analysis(df)

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_sentiment_comparison(df)
    plot_confidence_distribution(df)

    # Save results
    output_file = 'one_nz_with_deep_learning_sentiment.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to '{output_file}'")

    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your processed data
    print("Loading data...")
    df = pd.read_csv('one_nz_processed_reviews.csv')

    print(f"Loaded {len(df)} reviews")

    # Run complete analysis
    # Set fine_tune=True if you want to fine-tune BERT (recommended for best accuracy)
    # Set fine_tune=False for faster execution with only RoBERTa
    fine_tune = False

    df_final = run_complete_deep_learning_analysis(df, fine_tune=False)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("1. one_nz_with_deep_learning_sentiment.csv")
    print("2. sentiment_comparison.png")
    print("3. confidence_distribution.png")
    if fine_tune:
        print("4. ./fine_tuned_bert/ (model directory)")