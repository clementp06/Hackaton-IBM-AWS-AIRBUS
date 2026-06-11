import argparse
import csv
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_CORROSIONS_PATH = DATA_DIR / "corrosions_training.csv"
DEFAULT_ENVIRONMENT_PATH = DATA_DIR / "environment_training.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "prediction_training.csv"


def parse_date(value):
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def parse_year_month(value):
    year, month = value.split("-")
    return date(int(year), int(month), 1)


def read_corrosion_dates(corrosions_path):
    corrosion_dates = {}

    with corrosions_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            corrosion_dates[row["aircraft_id"]] = parse_date(row["observation_date"])

    return corrosion_dates


def build_prediction_training(corrosions_path, environment_path):
    corrosion_dates = read_corrosion_dates(corrosions_path)
    rows = []
    missing_aircraft_ids = set()

    with environment_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for environment_row in reader:
            aircraft_id = environment_row["aircraft_id"]
            corrosion_date = corrosion_dates.get(aircraft_id)

            if corrosion_date is None:
                missing_aircraft_ids.add(aircraft_id)
                continue

            prediction_year_month = environment_row["year_month"]
            prediction_date = parse_year_month(prediction_year_month)
            corrosion_risk = 0 if corrosion_date > prediction_date else 1

            rows.append(
                {
                    "id": f"{aircraft_id}_{prediction_year_month}",
                    "corrosion_risk": corrosion_risk,
                }
            )

    return rows, missing_aircraft_ids


def write_prediction_training(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "corrosion_risk"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, missing_aircraft_ids, output_path):
    positives = sum(row["corrosion_risk"] for row in rows)
    negatives = len(rows) - positives

    print(f"Fichier cree : {output_path}")
    print(f"Lignes : {len(rows)}")
    print(f"corrosion_risk=0 : {negatives}")
    print(f"corrosion_risk=1 : {positives}")
    print(f"Avions environnement absents de corrosions_training : {len(missing_aircraft_ids)}")

    if missing_aircraft_ids:
        preview = ", ".join(sorted(missing_aircraft_ids)[:10])
        print(f"Apercu des avions absents : {preview}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cree prediction_training.csv avec une ligne par occurrence de "
            "aircraft_id/year_month dans environment_training.csv."
        )
    )
    parser.add_argument("--corrosions", type=Path, default=DEFAULT_CORROSIONS_PATH)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rows, missing_aircraft_ids = build_prediction_training(
        args.corrosions,
        args.environment,
    )
    write_prediction_training(rows, args.output)
    print_summary(rows, missing_aircraft_ids, args.output)


if __name__ == "__main__":
    main()
