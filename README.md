# Queryflow

Graph retrieval with evolving query states. Includes data preparation, training, inference, evaluation, and the released model checkpoint.

## Install

Use Python 3.10, an NVIDIA GPU, and a full CUDA toolkit with nvcc and a C++ compiler. Run commands from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-12.4  # adjust to your installation
export MAX_JOBS=2
export OMP_NUM_THREADS=1
```

## Prepare data

The training and evaluation subsets are included under `data/raw/`. The text encoder downloads automatically. Graph construction requires an OpenAI-compatible LLM service.

```bash
export OPENAI_API_KEY=YOUR_KEY
# export OPENAI_API_BASE=https://your-server/v1
export QUERYFLOW_LLM_MODEL=qwen3.5-9B
python scripts/prepare_data.py --scope all --stage all
```

For an offline text encoder, set `QUERYFLOW_TEXT_MODEL` to a local `all-mpnet-base-v2` directory. Optional CPU encoding: `export QUERYFLOW_EMBEDDING_DEVICE=cpu`.

## Train

```bash
python scripts/train.py
```

To initialize from the released weights:

```bash
python scripts/train.py train.init_checkpoint=checkpoints/model_best.pth
```

## Inference

```bash
python scripts/infer.py --dataset wiki --max-questions 10 --top-k 10 --device cuda
```

## Evaluate

```bash
python scripts/evaluate.py --device cuda --output runs/metrics.json
```

To use another checkpoint, add `--checkpoint runs/YOUR_RUN/model_best.pth` to inference or evaluation. New training runs are saved under `runs/`.

## Quick start

After installation and service configuration:

```bash
python scripts/prepare_data.py --scope all --stage all && python scripts/evaluate.py --device cuda
```

Generated graph/tensor caches and text-encoder weights are excluded. Fresh LLM extraction may change results. The released checkpoint is ready for inference after data preparation; its historical training initialization and optimizer state are not included.

## License

Apache-2.0; see `LICENSE`. Includes code adapted from Apache-licensed graph-retrieval implementations, ULTRA/NBFNet, and HippoRAG. Original source credits are retained. Dataset subsets and text-encoder weights remain subject to their providers' terms.
