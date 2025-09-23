# Latent Neural Dynamics Modeling

To set up the environment, please follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/latent-neural-dynamics-modeling.git
   cd latent-neural-dynamics-modeling
   ```

2. Set up the Conda environment by running the provided script:
    ```bash
    bash environment/create_env.sh
    ```

Once the environment is set up, you can process the recordings to generate Parquet files of the processed participants table by running:

```bash
python -m preprocessing.package_recordings --config preprocessing/config.yaml
```

However, first you would need to update the config in preprocesing/config.yaml