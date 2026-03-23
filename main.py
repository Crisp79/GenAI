import yaml
import pandas as pd
from src.experiments.run_experiments import run_experiments

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    df = pd.read_csv(config["paths"]["labels"])
    df["image_path"] = df["image"].apply(
        lambda x: f"{config['paths']['processed']}/{x}"
    )

    results = run_experiments(config, df)
    print(results)

def run_pipeline(config):
    train_df = pd.read_csv(config["paths"]["train_csv"])
    test_df = pd.read_csv(config["paths"]["test_csv"])

    # build image paths
    train_df["image_path"] = train_df["id"].apply(
        lambda x: f"{config['paths']['processed']}/face{int(x)}.png"
    )

    test_df["image_path"] = test_df["id"].apply(
        lambda x: f"{config['paths']['processed']}/face{int(x)}.png"
    )

    results = run_experiments(config, train_df, test_df)
    return results

if __name__ == "__main__":
    main()