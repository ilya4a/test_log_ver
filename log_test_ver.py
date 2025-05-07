import os
import mlflow.data
from mlflow.tracking import MlflowClient
import torch.nn as nn
import inspect
from functools import wraps
from config import *


def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MNIST_Classification")


def log_data_info(train_loader, test_loader):
    """Логирование информации о данных с тегами"""
    data_path = os.path.abspath('./data')

    # data_path = os.path.abspath('./data').replace("\\", "/") !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def format_indices(loader, label):
        lines = [f"{label} indices:"]
        for batch_idx, (data, target) in enumerate(loader):
            start = batch_idx * loader.batch_size
            end = start + len(data)
            indices = list(range(start, end))
            line = f"Batch {batch_idx:03d}: " + ", ".join(map(str, indices))
            lines.append(line)
        return "\n".join(lines)

    indices_file_path = "data_indices.txt"
    with open(indices_file_path, "w") as f:
        f.write(format_indices(train_loader, "Train") + "\n\n")
        f.write(format_indices(test_loader, "Test") + "\n")

    info_file = "dataset_info.txt"

    # Создаем файл с информацией
    with open(info_file, "w") as f:
        f.write(f"Train dataset size: {len(train_loader.dataset)}\n")
        f.write(f"Test dataset size: {len(test_loader.dataset)}\n")
        f.write(f"Data path: {data_path}")

    # Логируем артефакты
    mlflow.log_artifact(info_file, "data_info")
    mlflow.log_artifact(indices_file_path, "data_info")
    # mlflow.log_artifact(data_path, "datasets") # логирует весь датасет

    # Логируем пути как теги
    mlflow.set_tags({
        "dataset.path": data_path,
        "dataset.info_file": f"data_info/{info_file}",
        "dataset.indices_file": f"data_info/{indices_file_path}"
    })


def log_model_to_mlflow(model):
    """Логирование модели в MLflow Registry"""
    with torch.no_grad():
        model.eval()
        input_example = torch.randn(1, 1, 28, 28).to(DEVICE)
        output_example = model(input_example)

        input_np = input_example.cpu().numpy() if DEVICE.type == 'cuda' else input_example.numpy()
        output_np = output_example.cpu().numpy() if DEVICE.type == 'cuda' else output_example.numpy()

    signature = mlflow.models.infer_signature(input_np, output_np)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="MNISTClassifier",
        code_paths=["model.py"],
        signature=signature,
        input_example=input_np,
        metadata={
            "task": "classification",
            "dataset": "MNIST",
            "model_type": "MLP"
        }
    )


def add_model_registry_info(model):
    """Добавление информации в Model Registry"""
    client = MlflowClient()
    model_version = client.get_latest_versions("MNISTClassifier", stages=["None"])[0].version

    # Извлечение архитектуры
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append(f"{module.in_features} → {module.out_features}")

    num_layers = len(layers)

    # Получаем имя файла
    model_file_path = inspect.getfile(model.__class__)
    model_file_name = os.path.basename(model_file_path)

    # Формирование описания
    description = f"""
    Многослойный перцептрон для классификации MNIST
    Архитектура:
    - Входной слой: {layers[0]}
    - Скрытый слой: {layers[1]}
    - Выходной слой: {layers[2]}
    Функции активации: ReLU для скрытых слоев, log_softmax для выходного слоя
    Код модели: {model_file_name}
    Параметры обучения:
    - Batch size: {BATCH_SIZE}
    - Learning rate: {LEARNING_RATE}
    - Epochs: {EPOCHS}
    Количество слоев: {num_layers}
    """

    # Обновление описания в Model Registry
    client.update_model_version(
        name="MNISTClassifier",
        version=model_version,
        description=description.strip()
    )

    # Добавление тегов
    tags = {
        "framework": "PyTorch",
        "dataset": "MNIST",
        "task_type": "classification",
        "code_reference": "model.py",
        "input_shape": "1x28x28",
        "output_classes": "10",
        "author": "Your Name",
        "num_layers": str(num_layers)  # Добавляем количество слоев в теги
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name="MNISTClassifier",
            version=model_version,
            key=key,
            value=value
        )

    # Перевод в продакшн
    client.transition_model_version_stage(
        name="MNISTClassifier",
        version=model_version,
        stage="Production"
    )


def create_model_card(model):
    """Создание документации модели с автоматическим извлечением информации"""
    with open("model_card.md", "w") as f:
        f.write("# Model Card\n\n")
        f.write("## Model Architecture\n\n")

        # Извлекаем информацию о слоях
        layers = []
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                layers.append({
                    'name': name,
                    'in_features': layer.in_features,
                    'out_features': layer.out_features
                })

        # Записываем информацию о слоях
        f.write("### Layers\n")
        f.write("| Layer Name | Input attributes | Output attributes |\n")
        f.write("|------------|------------------|-------------------|\n")
        for layer in layers:
            f.write(
                f"| {layer['name'].center(10)} | {str(layer['in_features']).center(16)} | {str(layer['out_features']).center(17)} |\n")

        # Общая информация о модели
        f.write("\n### Specifications\n")
        f.write(f"- Number of layers: {len(layers)}\n")
        f.write(f"- Total parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

        # Параметры обучения
        f.write("\n### Training parameters\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Learning rate: {LEARNING_RATE}\n")
        f.write(f"- Epochs: {EPOCHS}\n")

        f.write(f"- The learning device: {DEVICE}\n")

        # Функции активации
        f.write("\n### Activation functions\n")
        f.write("- ReLU (for hidden layers)\n")
        f.write("- LogSoftmax (for the output layer)\n")

    # Логируем сгенерированную карточку модели
    mlflow.log_artifact("model_card.md", "documentation")
    mlflow.log_artifact("model.py", "documentation")


def log_train_model(train_model):
    @wraps(train_model)
    def wrapper(*args, **kwargs):
        model = train_model(*args, **kwargs)

        mlflow.log_params({
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'device': str(DEVICE)
        })

        log_model_to_mlflow(model)
        # Дополнительная информация в реестре
        add_model_registry_info(model)
        # Документация
        create_model_card(model)

        return model

    return wrapper


def log_loaders(get_loaders):
    @wraps(get_loaders)
    def wrapper(*args, **kwargs):
        train_loader, test_loader = get_loaders(*args, **kwargs)

        log_data_info(train_loader, test_loader)

        return train_loader, test_loader

    return wrapper


def main():
    print("file log_test_ver.py")


if __name__ == "__main__":
    main()
