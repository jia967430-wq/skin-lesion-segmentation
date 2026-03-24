"""
Utilities Package

This package contains utility functions for:
- Visualization (segmentation results, training curves, etc.)
- Metrics calculation
- Data processing helpers
- Model complexity analysis
- Uncertainty quantification
"""

from .visualization import (
    denormalize_image,
    visualize_prediction,
    create_overlay_visualization,
    plot_training_curves,
    plot_metric_comparison,
    create_confusion_matrix_visual,
    visualize_batch_predictions,
)

from .complexity import (
    analyze_model,
    count_parameters,
    get_model_size,
    measure_latency,
    calculate_flops,
    print_analysis,
)

from .uncertainty import (
    UncertaintyEstimator,
    test_time_augmentation,
    EnsembleUncertainty,
    compute_uncertainty_metrics,
)

__all__ = [
    'denormalize_image',
    'visualize_prediction',
    'create_overlay_visualization',
    'plot_training_curves',
    'plot_metric_comparison',
    'create_confusion_matrix_visual',
    'visualize_batch_predictions',
    'analyze_model',
    'count_parameters',
    'get_model_size',
    'measure_latency',
    'calculate_flops',
    'print_analysis',
    'UncertaintyEstimator',
    'test_time_augmentation',
    'EnsembleUncertainty',
    'compute_uncertainty_metrics',
]
