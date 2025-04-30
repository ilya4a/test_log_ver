import torch
import mlflow
import os
import mlflow.data
from model import Net, get_loaders, train, evaluate_model
from mlflow.tracking import MlflowClient
from test_model import run_and_log_tests

# Конфигурация
BATCH_SIZE = 256
LEARNING_RATE = 0.1
EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MNIST_Classification")


def log_data_info(train_loader, test_loader):
    """Логирование информации о данных"""
    with open("dataset_info.txt", "w") as f:
        f.write(f"Train dataset size: {len(train_loader.dataset)}\n")
        f.write(f"Test dataset size: {len(test_loader.dataset)}\n")
        f.write(f"Data saved in: {os.path.abspath('./data')}")

    mlflow.log_artifact("dataset_info.txt", "data_info")
    mlflow.log_artifact(os.path.abspath('./data'), "datasets")


def train_model(train_loader):
    """Инициализация и обучение модели"""
    model = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)

    return model


def evaluate_and_log_metrics(model, train_loader, test_loader):
    """Оценка модели и логирование метрик"""
    train_loss, train_acc = evaluate_model(model, DEVICE, train_loader)
    test_loss, test_acc = evaluate_model(model, DEVICE, test_loader)

    mlflow.log_metrics({
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc
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


def add_model_registry_info():
    """Добавление информации в Model Registry"""
    client = MlflowClient()
    model_version = client.get_latest_versions("MNISTClassifier", stages=["None"])[0].version

    # Обновление описания
    client.update_model_version(
        name="MNISTClassifier",
        version=model_version,
        description="""
        Многослойный перцептрон для классификации MNIST
        Архитектура:
        - Входной слой: 784 → 200
        - Скрытый слой: 200 → 200
        - Выходной слой: 200 → 10
        Функции активации: ReLU
        Код модели: model.py
        Параметры обучения:
        - Batch size: 256
        - Learning rate: 0.1
        - Epochs: 1
        """
    )

    # Добавление тегов
    tags = {
        "framework": "PyTorch",
        "dataset": "MNIST",
        "task_type": "classification",
        "code_reference": "model.py",
        "input_shape": "1x28x28",
        "output_classes": "10",
        "author": "Your Name"
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name="MNISTClassifier",
            version=model_version,
            key=key,
            value=value
        )


    #Устанавливаем продакшн
    client.transition_model_version_stage(
        name="MNISTClassifier",
        version=model_version,
        stage="Production"
    )


def create_model_card():
    """Создание документации модели"""
    with open("model_card.md", "w") as f:
        f.write("# Model Card\n\n")
        f.write("## Архитектура\n```python\n")
        f.write(open("model.py").read())
        f.write("\n```\n\n## Параметры\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Learning rate: {LEARNING_RATE}\n")
        f.write(f"- Epochs: {EPOCHS}\n")

    mlflow.log_artifact("model_card.md", "documentation")


def log_data_info(train_loader, test_loader):
    """Логирование информации о данных с тегами"""
    data_path = os.path.abspath('./data')
    info_file = "dataset_info.txt"

    # Создаем файл с информацией
    with open(info_file, "w") as f:
        f.write(f"Train dataset size: {len(train_loader.dataset)}\n")
        f.write(f"Test dataset size: {len(test_loader.dataset)}\n")
        f.write(f"Data path: {data_path}")

    # Логируем артефакты
    mlflow.log_artifact(info_file, "data_info")
    mlflow.log_artifact(data_path, "datasets")

    # Логируем пути как теги
    mlflow.set_tags({
        "dataset.path": data_path,
        "dataset.info_file": f"data_info/{info_file}",
    })


def main():
    setup_mlflow()

    with mlflow.start_run() as run:
        print("Текущий tracking URI:", mlflow.get_tracking_uri())
        print("URI артефактов:", mlflow.get_artifact_uri())

        # Запускаем тесты
        test_success = run_and_log_tests()
        if not test_success:
            mlflow.set_tag("training.status", "aborted")
            raise RuntimeError("Тесты не прошли, обучение отменено")

        # Логирование параметров
        mlflow.log_params({
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'device': str(DEVICE)
        })

        # Подготовка данных
        train_loader, test_loader = get_loaders(BATCH_SIZE)
        log_data_info(train_loader, test_loader)

        # Обучение модели
        model = train_model(train_loader)

        # Оценка и логирование метрик
        evaluate_and_log_metrics(model, train_loader, test_loader)

        # Логирование модели
        log_model_to_mlflow(model)

        # Дополнительная информация в реестре
        add_model_registry_info()

        # Документация
        create_model_card()


if __name__ == "__main__":
    main()
