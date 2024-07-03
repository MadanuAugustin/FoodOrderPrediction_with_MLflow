

from dataclasses import dataclass
from pathlib import Path


#################################### DATA_INGESTION_CONFIG ################################


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    local_data_file : Path


#################################### DATA-VALIDATION-CONFIG ################################


@dataclass(frozen= True)
class DataValidationConfig:
    root_dir : Path
    local_data_file : Path
    STATUS_FILE : Path
    all_schema : Path


#################################### DATA-TRANSFORMATION-CONFIG ################################


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    local_data_file : Path
    train_path : Path
    test_path : Path



#################################### MODEL-TRAINER-CONFIG ################################


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir : Path
    train_data_path : Path
    test_data_path : Path
    model_name : str
    target_column : str