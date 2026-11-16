
from flask import Flask, render_template, request
from model_inference import predict_energy
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Historical dataset location
DATA_PATH = Path(__file__).resolve().parent / "data" / "converted.csv"


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":
        try:
            node_count = int(request.form["node_count"])
            wall_time = float(request.form["wall_time"])
            queue = request.form["queue"]
            job_mode = request.form["job_mode"]
            gpu_enabled = request.form["gpu_enabled"] == "true"
            project_prefix = request.form["project_prefix"]
            submit_datetime = request.form["submit_datetime"]

            result = predict_energy(
                node_count=node_count,
                wall_time=wall_time,
                queue=queue,
                job_mode=job_mode,
                gpu_enabled=gpu_enabled,
                project_prefix=project_prefix,
                submit_datetime=submit_datetime
            )

        except (ValueError, KeyError) as e:
            error = f"Invalid input: {e}"

        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error
    )


@app.route("/dashboard")
def dashboard():

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)

        # Convert numeric columns safely
        runtime = pd.to_numeric(df["run_time"], errors="coerce")
        node_count = pd.to_numeric(df["node_count"], errors="coerce")
        node_hours = pd.to_numeric(
            df["node_hour_usage"], errors="coerce"
        )

        # Main dashboard metrics
        total_jobs = len(df)

        average_runtime_hours = round(
            runtime.mean() / 3600, 2
        )

        total_node_allocations = int(node_count.sum())

        total_node_hours = round(node_hours.sum(), 2)

        # Completion status
        completion_data = (
            df["completion_state"]
            .fillna("Unknown")
            .value_counts()
        )

        # Queue distribution
        queue_data = (
            df["queue"]
            .fillna("Unknown")
            .value_counts()
        )

        # Job mode distribution
        job_mode_data = (
            df["job_mode"]
            .fillna("Unknown")
            .value_counts()
        )

        return render_template(
            "dashboard.html",
            total_jobs=total_jobs,
            average_runtime_hours=average_runtime_hours,
            total_node_allocations=total_node_allocations,
            total_node_hours=total_node_hours,
            completion_labels=completion_data.index.tolist(),
            completion_values=completion_data.values.tolist(),
            queue_labels=queue_data.index.tolist(),
            queue_values=queue_data.values.tolist(),
            mode_labels=job_mode_data.index.tolist(),
            mode_values=job_mode_data.values.tolist()
        )

    except FileNotFoundError:
        return "Dataset not found. Check that data/converted.csv exists.", 404

    except Exception as e:
        return f"Dashboard error: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
