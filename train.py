from train_utils import *
from log_test_ver import setup_mlflow
import mlflow.data
from tests.test_model import run_and_log_tests
from config import *


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


def main():
    setup_mlflow()
    with mlflow.start_run() as run:
        run_and_log_tests()

        print("Текущий tracking URI:", mlflow.get_tracking_uri())
        print("URI артефактов:", mlflow.get_artifact_uri())

        train_loader, test_loader = get_loaders(BATCH_SIZE)

        model = train_model(train_loader)

        evaluate_and_log_metrics(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
