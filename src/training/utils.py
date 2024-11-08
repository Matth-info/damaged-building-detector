

from torch.utils.tensorboard import SummaryWriter

def log_metrics(writer: SummaryWriter, metrics: dict, epoch_number: int, phase: str = 'Validation'):
    """
    Logs each metric in the dictionary to TensorBoard.

    Parameters:
    - writer: The SummaryWriter instance.
    - metrics: Dictionary of metric name and value pairs.
    - epoch_number: The current epoch number.
    - phase: 'Validation' or 'Training', used to distinguish metrics in TensorBoard.
    """
    for metric_name, value in metrics.items():
        writer.add_scalar(f'{phase}/{metric_name}', value, epoch_number)