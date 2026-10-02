"""터미널에서 전체 실험을 한 번에 실행합니다."""

from config import CONFIG
from src.experiment import run_experiment
from src.reporting import print_terminal_report


def main() -> None:
    result = run_experiment(CONFIG, progress=True)
    print_terminal_report(result, show_plots=True)


if __name__ == "__main__":
    main()
