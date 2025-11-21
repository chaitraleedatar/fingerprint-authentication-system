"""
Visualization scripts for fingerprint authentication results.
Run this after pipeline execution to generate visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2


def load_metrics(metrics_path: Path):
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(val_metrics: dict, test_metrics: dict, output_path: Path):
    """Plot accuracy comparison between validation and test sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Closed-set accuracy
    categories = ['Validation', 'Test']
    closed_set = [val_metrics['closed_set_accuracy'], test_metrics['closed_set_accuracy']]
    open_set = [val_metrics['open_set_accuracy'], test_metrics['open_set_accuracy']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, closed_set, width, label='Closed-set', alpha=0.8)
    ax1.bar(x + width/2, open_set, width, label='Open-set', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # FAR and Unknown Detection
    far = [val_metrics['false_accept_rate'], test_metrics['false_accept_rate']]
    unknown_det = [val_metrics['unknown_detection_rate'], test_metrics['unknown_detection_rate']]
    
    ax2.bar(x - width/2, far, width, label='FAR', alpha=0.8, color='red')
    ax2.bar(x + width/2, unknown_det, width, label='Unknown Detection', alpha=0.8, color='green')
    ax2.set_ylabel('Rate')
    ax2.set_title('Security Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'accuracy_comparison.png'}")


def plot_score_distribution(predictions_path: Path, output_path: Path):
    """Plot distribution of match scores."""
    df = pd.read_csv(predictions_path)
    
    # Separate genuine and impostor scores
    genuine_scores = df[df['label'] == df['prediction']]['score']
    impostor_scores = df[df['label'] != df['prediction']]['score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine Matches', color='green')
    ax.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor Matches', color='red')
    ax.axvline(x=0.35, color='black', linestyle='--', label='Threshold (0.35)')
    ax.set_xlabel('Match Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Match Scores')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'score_distribution.png'}")


def visualize_minutiae(image_path: Path, minutiae_list, output_path: Path):
    """Visualize minutiae points on fingerprint image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for m in minutiae_list:
        x, y = int(m.x), int(m.y)
        color = (0, 255, 0) if m.type == 'ending' else (255, 0, 0)
        cv2.circle(img_color, (x, y), 3, color, -1)
        # Draw orientation
        angle = m.orientation
        end_x = int(x + 10 * np.cos(angle))
        end_y = int(y + 10 * np.sin(angle))
        cv2.line(img_color, (x, y), (end_x, end_y), color, 1)
    
    cv2.imwrite(str(output_path), img_color)
    print(f"Saved: {output_path}")


def plot_preprocessing_pipeline(image_path: Path, output_path: Path):
    """Visualize preprocessing steps."""
    from src.preprocessing import load_image, preprocess_image
    from src.config import PipelineConfig
    from pathlib import Path
    
    config = PipelineConfig.from_args(
        'Project-Data/Project-Data/train',
        'Project-Data/Project-Data/validate',
        'Project-Data/Project-Data/test'
    )
    
    original = load_image(image_path)
    thinned = preprocess_image(original, config)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Fingerprint')
    axes[0].axis('off')
    
    axes[1].imshow(thinned, cmap='gray')
    axes[1].set_title('After Preprocessing (Thinned)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    """Generate all visualizations."""
    artifacts_dir = Path('artifacts/minutiae_baseline')
    output_dir = artifacts_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Load metrics
    val_metrics = load_metrics(artifacts_dir / 'val_metrics.json')
    test_metrics = load_metrics(artifacts_dir / 'test_metrics.json')
    
    print("Generating visualizations...")
    
    # Plot accuracy comparison
    plot_accuracy_comparison(val_metrics, test_metrics, output_dir)
    
    # Plot score distributions
    plot_score_distribution(artifacts_dir / 'test_predictions.csv', output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nNote: For minutiae visualization, use:")
    print("  visualize_minutiae(image_path, minutiae_list, output_path)")
    print("\nFor preprocessing visualization, use:")
    print("  plot_preprocessing_pipeline(image_path, output_path)")


if __name__ == '__main__':
    main()

