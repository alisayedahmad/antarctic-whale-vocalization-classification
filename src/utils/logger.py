import logging
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime


def setup_logger(
    name: str,
    config: Dict,
    log_to_file: bool = True
) -> logging.Logger:
    """Setup logger with configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    formatter = logging.Formatter(config['logging']['format'])
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_to_file and config['logging']['save_to_file']:
        logs_dir = Path(config['output']['logs_dir'])
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'metrics_{timestamp}.csv'
        
        self.metrics = []
    
    def log(self, epoch: int, metrics: Dict):
        """Log metrics for an epoch."""
        entry = {'epoch': epoch, **metrics}
        self.metrics.append(entry)
        
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_file, index=False)
    
    def get_metrics(self) -> list:
        """Get all logged metrics."""
        return self.metrics
