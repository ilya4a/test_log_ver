import pytest
from torch.utils.data import TensorDataset
import math
import torch.optim as optim
import mlflow
from datetime import datetime
from pathlib import Path

from train_utils import *


# Фикстуры
@pytest.fixture(scope="session")
def mlflow_run():
    """Фикстура для создания MLflow run"""
    with mlflow.start_run(nested=True) as run:
        yield run


@pytest.fixture
def sample_batch():
    train_loader, _ = get_loaders.__wrapped__(batch_size=32)
    batch = next(iter(train_loader))
    mlflow.log_param("batch_size", batch[0].shape[0])
    return batch


@pytest.fixture
def simple_model():
    model = Net()
    mlflow.log_params({
        "model.input_size": 784,
        "model.hidden_size": 200,
        "model.output_size": 10
    })
    return model


# Тесты
def test_model_initialization(simple_model, mlflow_run):
    """Тестирует инициализацию модели"""
    start_time = datetime.now()
    try:
        assert isinstance(simple_model, torch.nn.Module)
        assert len(list(simple_model.parameters())) == 6
        mlflow.log_metric("test.model_initialization", 1)
    except AssertionError as e:
        mlflow.log_metric("test.model_initialization", 0)
        mlflow.log_text(str(e), "errors/test.model_initialization.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.model_initialization.duration", duration)


def test_model_forward_pass(simple_model, sample_batch, mlflow_run):
    """Тестирует прямой проход модели"""
    start_time = datetime.now()
    try:
        data, _ = sample_batch
        output = simple_model(data)
        assert output.shape == (32, 10)
        mlflow.log_metric("test.model_forward_pass", 1)
    except AssertionError as e:
        mlflow.log_metric("test.model_forward_pass", 0)
        mlflow.log_text(str(e), "errors/test.model_forward_pass.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.model_forward_pass.duration", duration)


def test_with_dataloader(simple_model, mlflow_run):
    start_time = datetime.now()
    try:
        train_loader, _ = get_loaders.__wrapped__(batch_size=32)
        loss, accuracy = evaluate_model(simple_model, torch.device('cpu'), train_loader)
        assert 0 <= loss < 100
        assert 0 <= accuracy <= 100
        mlflow.log_metric("test.with_dataloader", 1)
    except AssertionError as e:
        mlflow.log_metric("test.with_dataloader", 0)
        mlflow.log_text(str(e), "errors/test.with_dataloader.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.with_dataloader.duration", duration)


def test_train_step(simple_model, sample_batch, mlflow_run):
    start_time = datetime.now()
    try:
        optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
        data, target = sample_batch

        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=data.size(0))

        initial_loss = evaluate_model(simple_model, torch.device('cpu'), loader)[0]

        simple_model.train()
        optimizer.zero_grad()
        output = simple_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        final_loss = evaluate_model(simple_model, torch.device('cpu'), loader)[0]
        assert final_loss != initial_loss
        mlflow.log_metric("test.train_step", 1)
    except AssertionError as e:
        mlflow.log_metric("test.train_step", 0)
        mlflow.log_text(str(e), "errors/test.train_step.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.train_step.duration", duration)


def test_evaluation(simple_model, sample_batch, mlflow_run):
    start_time = datetime.now()
    try:
        data, target = sample_batch
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=data.size(0))

        loss, accuracy = evaluate_model(simple_model, torch.device('cpu'), loader)

        assert data.shape == (32, 1, 28, 28)
        assert target.shape == (32,)
        assert 0 <= loss < 100
        assert 0 <= accuracy <= 100
        mlflow.log_metric("test.evaluation", 1)
    except AssertionError as e:
        mlflow.log_metric("test.evaluation", 0)
        mlflow.log_text(str(e), "errors/test.evaluation.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.evaluation.duration", duration)


def test_empty_batch(simple_model, mlflow_run):
    start_time = datetime.now()
    try:
        empty_data = torch.empty(0, 1, 28, 28)
        empty_target = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(empty_data, empty_target)
        loader = DataLoader(dataset, batch_size=32)

        loss, accuracy = evaluate_model(simple_model, torch.device('cpu'), loader)
        assert isinstance(loss, float)
        assert accuracy == 0.0
        assert math.isnan(loss)
        mlflow.log_metric("test.empty_batch", 1)
    except AssertionError as e:
        mlflow.log_metric("test.empty_batch", 0)
        mlflow.log_text(str(e), "errors/test.empty_batch.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.empty_batch.duration", duration)


def test_with_batch_list(simple_model, sample_batch, mlflow_run):
    start_time = datetime.now()
    try:
        loss, accuracy = evaluate_model(simple_model, torch.device('cpu'), [sample_batch])
        assert 0 <= loss < 100
        assert 0 <= accuracy <= 100
        mlflow.log_metric("test.with_batch_list", 1)
    except AssertionError as e:
        mlflow.log_metric("test.with_batch_list", 0)
        mlflow.log_text(str(e), "errors/test.with_batch_list.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.with_batch_list.duration", duration)


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_different_batch_sizes(batch_size, mlflow_run):
    start_time = datetime.now()
    try:
        train_loader, _ = get_loaders.__wrapped__(batch_size=batch_size)
        batch = next(iter(train_loader))
        assert batch[0].shape[0] == batch_size
        mlflow.log_metric(f"test.batch_size_{batch_size}", 1)
    except AssertionError as e:
        mlflow.log_metric(f"test.batch_size_{batch_size}", 0)
        mlflow.log_text(str(e), f"errors/test.batch_size_{batch_size}.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric(f"test.batch_size_{batch_size}.duration", duration)


def test_model_on_cuda_if_available(simple_model, mlflow_run):
    start_time = datetime.now()
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simple_model.to(device)
        assert next(simple_model.parameters()).is_cuda == torch.cuda.is_available()
        mlflow.log_metric("test.cuda_available", 1)
        mlflow.log_param("cuda_used", torch.cuda.is_available())
    except AssertionError as e:
        mlflow.log_metric("test.cuda_available", 0)
        mlflow.log_text(str(e), "errors/test.cuda_available.txt")
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("test.cuda_available.duration", duration)


def run_and_log_tests():
    """Запускает тесты и логирует результаты в MLflow"""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MNIST_Classification")

    try:
        current_dir = Path.cwd()

        report_path = current_dir / "test_reports" / "test_results.xml"
        coverage_path = current_dir / "coverage.xml"

        report_path.parent.mkdir(exist_ok=True, parents=True)

        exit_code = pytest.main([
            "-v",
            "tests",
            f"--junit-xml={report_path}",
            "--cov=model",
            f"--cov-report=xml:{coverage_path}"
        ])

        print("EXIT_CODE", exit_code)

        success = exit_code == 0

        # Логируем артефакты тестов
        mlflow.log_artifacts(report_path.parent, artifact_path="test_reports")

        mlflow.log_artifacts(report_path.parent, artifact_path="test_reports")

        print(f"Report path: {report_path} - Exists: {report_path.exists()}")
        print(f"Coverage path: {coverage_path} - Exists: {coverage_path.exists()}")

        mlflow.log_param("tests_passed", success)

    except Exception as e:
        mlflow.log_text(f"Critical error: {str(e)}", "errors/critical.txt")
        return False

    if not success:
        mlflow.set_tag("training.status", "aborted")
        raise RuntimeError("Тесты не прошли, обучение отменено")
    return success


if __name__ == "__main__":
    success = run_and_log_tests()
    if not success:
        raise RuntimeError("Тесты завершились с ошибками!")
    print("Все тесты успешно пройдены!")
