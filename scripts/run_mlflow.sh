# Create MLflow directory
mkdir -p ~/mlflow-server
chmod 755 ~/mlflow-server

# Start with SQLite backend
# Use full path, not ~
uv run mlflow server \
    --backend-store-uri sqlite:////home/ec2-user/mlflow-server/mlflow.db \
    --default-artifact-root file:///home/ec2-user/mlflow-server/artifacts \
    --host 0.0.0.0 \
    --port 8000
