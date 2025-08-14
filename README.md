# Distillery

An agentic workflow for distilling LLM outputs into small models on Modal compute, built for AI Agent & Infra Hackathon

## Usage

Install dependencies from requirements.txt:
```
pip install -r requirements.txt
```

Host the backend on Modal:
```
modal deploy distillery_backend.py
```
This will output a URL for the backend at which the model is hosted.

Run the CLI:
```
python distillery_cli.py --backend <url from the previous step>
```