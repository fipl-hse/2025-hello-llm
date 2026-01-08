"""
Helper functions.
"""


def compare_pretrained_and_finetuned_results(
    pretrained_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    model_name: str,
    dataset_name: str,
) -> None:
    """
    Compares the metrics before and after fine-tuning and prints the results.

    Args:
        pretrained_metrics (dict[str, float]): results of pretrained model
        finetuned_metrics (dict[str, float]): results of finetuned model
        model_name (str): name of the model
        dataset_name (str): name of the dataset

    Raises:
        ValueError: if pretrained and finetuned results have different metrics
    """

    if pretrained_metrics.keys() != finetuned_metrics.keys():
        raise ValueError("Pretrained and finetuned metrics must have the same metrics.")
    for metric_name, pretrained_metric_result in pretrained_metrics.items():
        finetuned_metric_result = finetuned_metrics[metric_name]
        verb = "improved"
        if finetuned_metric_result < pretrained_metric_result:
            verb = "degraded"
        if finetuned_metric_result == pretrained_metric_result:
            verb = "remained the same"

        print(
            f"For model {model_name} on dataset {dataset_name} "
            f"the metric {metric_name} is {verb} after fine-tuning: "
            f"from {pretrained_metric_result} to {finetuned_metric_result}"
        )
