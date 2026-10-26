import hashlib
import json
import logging
import os
import time
from itertools import islice

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.utils import data as torch_data
from tqdm import tqdm

from queryflow import utils
from queryflow.datasets import QADataset
from queryflow.doc_rankers import SoftplusIDFWeightedRanker
from queryflow.ultra import query_utils
from queryflow.utils import GraphDatasetLoader

# A logger for this file
logger = logging.getLogger(__name__)

separator = ">" * 30
line = "-" * 30


def _phase_dataset_config(cfg: DictConfig, phase: str) -> DictConfig:
    """Resolve separate train/validation roots while preserving old configs."""
    payload = OmegaConf.to_container(cfg.datasets, resolve=True)
    cfgs = dict(payload["cfgs"])
    phase_root = cfgs.get(f"{phase}_root")
    if phase_root:
        cfgs["root"] = phase_root
    cfgs.pop("train_root", None)
    cfgs.pop("valid_root", None)
    cfgs.pop("test_root", None)
    payload["cfgs"] = cfgs
    return OmegaConf.create(payload)


def create_qa_dataloader(
    dataset: dict[str, QADataset],
    batch_size: int,
    world_size: int,
    rank: int,
    is_train: bool = True,
    shuffle: bool = True,
    max_questions: int | None = None,
    evidence_labels_dir: str | None = None,
) -> dict:
    """
    Create a dataloader for the QA dataset.
    """
    data_name = dataset["data_name"]
    qa_data = dataset["data"]
    train_data, valid_data = qa_data._data
    data = train_data if is_train else valid_data
    if is_train and evidence_labels_dir:
        from queryflow.evidence_supervision import load_evidence_dataset

        data = load_evidence_dataset(data, qa_data, evidence_labels_dir)
    if max_questions is not None:
        data = torch_data.Subset(data, range(min(int(max_questions), len(data))))

    sampler = torch_data.DistributedSampler(
        data,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    data_loader = torch_data.DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
    )

    # Return data
    return {
        "data_name": data_name,
        "data_loader": data_loader,
        "graph": qa_data.kg,
        "ent2docs": qa_data.ent2docs,
    }


def _prepare_graph(graph, device: torch.device):
    graph = graph.to(device)
    if not getattr(graph.edge_index, "_queryflow_rspmm_sorted", False):
        node_in, node_out = graph.edge_index
        order = (node_in * graph.num_nodes + node_out).argsort()
        graph.edge_index = graph.edge_index[:, order]
        graph.edge_type = graph.edge_type[order]
        graph.edge_index._queryflow_rspmm_sorted = True
    return graph


def _load_initial_weights(model: nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    logger.info(f"Initialized model weights with an exact match from {path}")


def _loss_gradient_norm(loss: torch.Tensor, parameters) -> float:
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared_norm = sum(
        (gradient.detach().float().square().sum() for gradient in gradients if gradient is not None),
        loss.new_zeros((), dtype=torch.float32),
    )
    return float(squared_norm.sqrt())


def train_and_validate(
    cfg: DictConfig,
    output_dir: str,
    model: nn.Module,
    train_dataset_loader: GraphDatasetLoader,
    valid_dataset_loader: GraphDatasetLoader | dict[str, GraphDatasetLoader],
    device: torch.device,
    batch_per_epoch: int | None = None,
) -> dict | None:
    if cfg.train.num_epoch == 0:
        return

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    optimizer = instantiate(cfg.optimizer, model.parameters())
    max_grad_norm = float(cfg.train.get("max_grad_norm", 0.0))
    start_epoch = 0
    best_result = float("-inf")
    best_epoch = -1
    best_metrics = None
    initial_validation = None
    epochs_without_improvement = 0
    batch_id = 0
    selection_metric = str(cfg.train.get("selection_metric", "entity_mrr"))
    patience = int(cfg.train.get("early_stopping_patience", 0))
    if patience > 0 and not cfg.train.do_eval:
        raise ValueError("Early stopping requires train.do_eval=true")
    # Load optimizer state and epoch if exists
    if "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = float(cfg.optimizer.lr)
            else:
                logger.warning(
                    f"Optimizer state not found in {cfg.train.checkpoint}, using default optimizer."
                )
            if "epoch" in state:
                start_epoch = state["epoch"]
                logger.warning(f"Resuming training from epoch {start_epoch}.")
            best_result = state.get("best_result", best_result)
            best_epoch = state.get("best_epoch", best_epoch)
            best_metrics = state.get("best_metrics")
            initial_validation = state.get("initial_validation")
            epochs_without_improvement = state.get("epochs_without_improvement", 0)
            batch_id = state.get("global_step", 0)
            if "rng_state" in state:
                torch.set_rng_state(state["rng_state"])
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(state["cuda_rng_state"], device)
        else:
            logger.warning(
                f"Checkpoint {cfg.train.checkpoint} does not exist, using default optimizer."
            )
    logger.info(f"Optimizer learning rates: {[group['lr'] for group in optimizer.param_groups]}")

    initial_metrics_path = cfg.train.get("initial_metrics")
    if initial_metrics_path and not cfg.train.get("checkpoint"):
        if not cfg.train.do_eval:
            raise ValueError("Initial metrics require validation to be enabled")
        with open(initial_metrics_path) as f:
            baseline = json.load(f)
        with open(cfg.train.init_checkpoint, "rb") as f:
            checkpoint_digest = hashlib.sha256(f.read()).hexdigest()
        if checkpoint_digest != baseline["checkpoint_sha256"]:
            raise ValueError("Initial metrics refer to a different checkpoint")
        initial_validation = baseline["metrics"]
        best_metrics = initial_validation
        best_result = float(np.mean([
            metrics[selection_metric] for metrics in best_metrics.values()
        ]))
        if not np.isfinite(best_result):
            raise ValueError("Non-finite initial validation score")
        best_epoch = 0
        if rank == 0:
            initial_state = {
                "epoch": 0, "global_step": 0,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best_result": best_result, "best_epoch": 0,
                "best_metrics": best_metrics, "initial_validation": initial_validation,
                "epochs_without_improvement": 0,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            torch.save(initial_state, os.path.join(output_dir, "model_best.pth"))
            torch.save(initial_state, os.path.join(output_dir, "model_last.pth"))
        logger.info(f"Reused initial validation from {initial_metrics_path}: {selection_metric}={best_result:g}")

    # Initialize Losses
    loss_fn_list = []
    has_doc_loss = False
    for loss_cfg in cfg.task.losses:
        loss_fn = instantiate(loss_cfg.loss)
        if loss_cfg.cfg.is_doc_loss:
            has_doc_loss = True
        loss_fn_list.append(
            {
                "name": loss_cfg.name,
                "loss_fn": loss_fn,
                **loss_cfg.cfg,
            }
        )

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    history_path = os.path.join(output_dir, "training_history.json")
    history = []
    if start_epoch > 0 and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)["epochs"][:start_epoch]
    for i in range(start_epoch, cfg.train.num_epoch):
        epoch = i + 1
        epoch_started = time.monotonic()
        parallel_model.train()

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} begin")

        losses: dict[str, list] = {loss_dict["name"]: [] for loss_dict in loss_fn_list}
        losses["loss"] = []
        gradient_diagnostics = []
        if max_grad_norm > 0:
            losses["grad_norm"] = []
        shard_stats = {}
        train_dataset_loader.set_epoch(
            epoch
        )  # Make sure the datasets order is the same across all processes
        for train_dataset in train_dataset_loader:
            if device.type == "cuda":
                # Large shards can leave cached graph activations close to the GPU limit.
                torch.cuda.empty_cache()
            train_dataset = create_qa_dataloader(
                train_dataset,
                cfg.train.batch_size,
                world_size,
                rank,
                is_train=True,
                shuffle=True,
                evidence_labels_dir=cfg.train.get("evidence_labels_dir"),
            )
            train_loader = train_dataset["data_loader"]
            train_loader.sampler.set_epoch(epoch)
            data_name = train_dataset["data_name"]
            graph = _prepare_graph(train_dataset["graph"], device)
            ent2docs = train_dataset["ent2docs"].to(device)
            doc_loss_ranker = SoftplusIDFWeightedRanker(ent2docs) if has_doc_loss else None
            entities_weight = None
            if cfg.train.init_entities_weight:
                entities_weight = utils.get_entities_weight(ent2docs)
            shard_batches = len(train_loader)
            if batch_per_epoch is not None:
                shard_batches = min(int(batch_per_epoch), shard_batches)
            shard_stats[data_name] = {"num_questions": 0, "num_batches": 0}
            if cfg.train.get("record_training_sample_ids", False):
                shard_stats[data_name]["sample_ids"] = []
            for batch in tqdm(
                islice(train_loader, shard_batches),
                desc=f"Training Batches: {data_name}: {epoch}",
                total=shard_batches,
                disable=not utils.is_main_process(),
            ):
                batch = query_utils.cuda(batch, device=device)
                doc_pred = None
                doc_target = None
                pred = parallel_model(graph, batch, entities_weight=entities_weight)
                target = batch.get("evidence_entity_targets", batch["supporting_entities_masks"])

                if has_doc_loss:
                    # Training always uses all entities, through a differentiable document readout.
                    doc_pred = doc_loss_ranker(pred)
                    doc_target = batch["supporting_docs_masks"]  # supporting_docs_mask

                loss = 0
                tmp_losses = {}
                log_loss_gradients = (
                    cfg.train.get("log_loss_grad_norms", False)
                    and batch_id % cfg.train.log_interval == 0
                )
                loss_gradients = {}
                for loss_dict in loss_fn_list:
                    loss_fn = loss_dict["loss_fn"]
                    weight = loss_dict["weight"]
                    if loss_dict["is_doc_loss"]:
                        single_loss = loss_fn(doc_pred, doc_target)
                    else:
                        single_loss = loss_fn(pred, target)
                    tmp_losses[loss_dict["name"]] = single_loss.item()
                    if log_loss_gradients:
                        grad_norm = _loss_gradient_norm(single_loss, model.parameters())
                        loss_gradients[loss_dict["name"]] = {
                            "raw_norm": grad_norm, "weighted_norm": abs(weight) * grad_norm,
                        }
                    loss += weight * single_loss
                tmp_losses["loss"] = loss.item()  # type: ignore
                if log_loss_gradients:
                    gradient_diagnostics.append({"global_step": batch_id, "losses": loss_gradients})
                    logger.info(f"Loss gradient norms before clipping: {loss_gradients}")

                loss.backward()  # type: ignore
                if max_grad_norm > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                    tmp_losses["grad_norm"] = float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                for loss_log in tmp_losses:
                    losses[loss_log].append(tmp_losses[loss_log])

                if utils.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.info(separator)
                    for loss_log in tmp_losses:
                        logger.info(f"{loss_log}: {tmp_losses[loss_log]:g}")
                batch_id += 1
                shard_stats[data_name]["num_questions"] += len(target)
                shard_stats[data_name]["num_batches"] += 1
                if cfg.train.get("record_training_sample_ids", False):
                    shard_stats[data_name]["sample_ids"].extend(batch["sample_id"].view(-1).tolist())
                # Release the previous batch before the next graph forward.  On large
                # shards, retaining the autograd graph here can overlap its peak with
                # the next batch and cause an avoidable CUDA OOM.
                del batch, pred, target, doc_pred, doc_target, loss
            # PyG's device transfer also mutates graphs held by the dataset cache.
            graph.cpu()
            del graph, ent2docs, entities_weight, doc_loss_ranker
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info(f"Epoch {epoch} end")
            logger.info(line)
            for loss_log in losses:
                logger.info(
                    f"Avg: {loss_log}: {sum(losses[loss_log]) / len(losses[loss_log]):g}"
                )

        utils.synchronize()

        if cfg.train.do_eval:
            if isinstance(valid_dataset_loader, dict):
                validation_metrics = {}
                for scope, dataset_loader in valid_dataset_loader.items():
                    if rank == 0:
                        logger.info(separator)
                        logger.info(f"Evaluate on {scope}")
                    validation_metrics[scope] = test(
                        cfg, model, dataset_loader, device=device, return_metrics=True
                    )
                selection_scope = str(cfg.train.get("selection_dataset", "dev"))
                if selection_scope not in validation_metrics:
                    raise ValueError(
                        f"Selection dataset {selection_scope!r} is not evaluated"
                    )
                selection_metrics = validation_metrics[selection_scope]
            else:
                if rank == 0:
                    logger.info(separator)
                    logger.info("Evaluate on valid")
                validation_metrics = test(
                    cfg, model, valid_dataset_loader, device=device, return_metrics=True
                )
                selection_metrics = validation_metrics
            result = float(np.mean(
                [metrics[selection_metric] for metrics in selection_metrics.values()]
            ))
            if not np.isfinite(result):
                raise ValueError(f"Non-finite validation {selection_metric}: {result}")
        else:
            validation_metrics = None
            result = float("inf")
            best_result = float("-inf")
        improved = result > best_result
        if improved:
            best_result = result
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = patience > 0 and epochs_without_improvement >= patience
        if rank == 0:
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": batch_id,
                "best_result": best_result,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "initial_validation": initial_validation,
                "epochs_without_improvement": epochs_without_improvement,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            }
            if improved:
                logger.info("Save checkpoint to model_best.pth")
                torch.save(state, os.path.join(output_dir, "model_best.pth"))
            if patience > 0:
                torch.save(state, os.path.join(output_dir, "model_last.pth"))
            if not cfg.train.save_best_only:
                logger.info(f"Save checkpoint to model_epoch_{epoch}.pth")
                torch.save(state, os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
            history.append({
                "epoch": epoch,
                "global_step": batch_id,
                "training_shards": shard_stats,
                "training_questions_per_rank": sum(
                    row["num_questions"] for row in shard_stats.values()
                ),
                "losses": {name: float(np.mean(values)) for name, values in losses.items()},
                "loss_gradient_norms": gradient_diagnostics,
                "validation": validation_metrics,
                "selection_score": result if cfg.train.do_eval else None,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": time.monotonic() - epoch_started,
            })
            status = "early_stopped" if should_stop else "running"
            if epoch == cfg.train.num_epoch and not should_stop:
                status = "max_epochs_reached"
            with open(history_path, "w") as f:
                json.dump({
                    "status": status,
                    "selection_metric": selection_metric,
                    "early_stopping_patience": patience,
                    "best_epoch": best_epoch,
                    "best_score": best_result if cfg.train.do_eval else None,
                    "initial_validation": initial_validation,
                    "epochs": history,
                }, f, indent=2)
            logger.info(
                f"Best {selection_metric}: {best_result:g} at epoch {best_epoch}; "
                f"epochs without improvement: {epochs_without_improvement}/{patience}"
            )
        if should_stop:
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    if rank == 0:
        logger.info("Load checkpoint from model_best.pth")
    utils.synchronize()
    state = torch.load(
        os.path.join(output_dir, "model_best.pth"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state["model"])
    return state.get("best_metrics")


@torch.no_grad()
def test(
    cfg: DictConfig,
    model: nn.Module,
    test_dataset_loader: GraphDatasetLoader,
    device: torch.device,
    return_metrics: bool = False,
) -> float | dict:
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    # process sequentially of test datasets
    all_metrics = {}
    all_scores = []
    selection_metric = str(cfg.train.get("selection_metric", "entity_mrr"))
    if selection_metric not in {"entity_mrr", "doc_mrr", "doc_r10", "doc_fr10"}:
        raise ValueError(f"Unsupported train.selection_metric: {selection_metric}")
    for dataset in test_dataset_loader:
        dataset = create_qa_dataloader(
            dataset,
            cfg.train.batch_size,
            world_size,
            rank,
            is_train=False,
            shuffle=False,
        )
        test_loader = dataset["data_loader"]
        test_loader.sampler.set_epoch(0)
        data_name = dataset["data_name"]
        graph = _prepare_graph(dataset["graph"], device)
        ent2docs = dataset["ent2docs"].to(device)

        model.eval()
        ent_preds = []
        ent_targets = []
        doc_preds = []
        doc_targets = []

        # Create doc retriever
        doc_ranker = instantiate(
            cfg.doc_ranker,
            ent2doc=ent2docs,
        )

        entities_weight = None
        if cfg.train.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)

        for batch in tqdm(
            test_loader,
            desc=f"Testing {data_name}",
            disable=not utils.is_main_process(),
        ):
            batch = query_utils.cuda(batch, device=device)
            ent_pred = model(graph, batch, entities_weight=entities_weight)
            doc_pred = doc_ranker(ent_pred)  # Ent2docs mapping
            target_entities_mask = batch[
                "supporting_entities_masks"
            ]  # supporting_entities_mask
            target_docs_mask = batch["supporting_docs_masks"]  # supporting_docs_mask
            target_entities = target_entities_mask.bool()
            target_docs = target_docs_mask.bool()
            ent_ranking, target_ent_ranking = utils.batch_evaluate(
                ent_pred, target_entities
            )
            doc_ranking, target_doc_ranking = utils.batch_evaluate(
                doc_pred, target_docs
            )

            # answer set cardinality prediction
            ent_prob = F.sigmoid(ent_pred)
            num_pred = (ent_prob * (ent_prob > 0.5)).sum(dim=-1)
            num_target = target_entities_mask.sum(dim=-1)
            ent_preds.append((ent_ranking, num_pred))
            ent_targets.append((target_ent_ranking, num_target))

            # document set cardinality prediction
            doc_prob = F.sigmoid(doc_pred)
            num_pred = (doc_prob * (doc_prob > 0.5)).sum(dim=-1)
            num_target = target_docs_mask.sum(dim=-1)
            doc_preds.append((doc_ranking, num_pred))
            doc_targets.append((target_doc_ranking, num_target))

            # Do not retain the full entity/document score tensors until the next
            # batch.  The rankings stored above are the only values needed later.
            del batch, ent_pred, doc_pred, target_entities_mask, target_docs_mask
            del target_entities, target_docs, ent_ranking, target_ent_ranking
            del doc_ranking, target_doc_ranking, ent_prob, doc_prob, num_pred, num_target

        ent_pred = query_utils.cat(ent_preds)
        ent_target = query_utils.cat(ent_targets)
        doc_pred = query_utils.cat(doc_preds)
        doc_target = query_utils.cat(doc_targets)

        ent_pred, ent_target = utils.gather_results(
            ent_pred, ent_target, rank, world_size, device
        )
        doc_pred, doc_target = utils.gather_results(
            doc_pred, doc_target, rank, world_size, device
        )
        ent_metrics = utils.evaluate(ent_pred, ent_target, cfg.task.metric)
        metrics = {}
        if rank == 0:
            doc_metrics = utils.evaluate(doc_pred, doc_target, cfg.task.metric)
            for key, value in ent_metrics.items():
                metrics[f"ent_{key}"] = value
            for key, value in doc_metrics.items():
                metrics[f"doc_{key}"] = value
            metrics["entity_mrr"] = ent_metrics["mrr"]
            metrics["doc_mrr"] = doc_metrics["mrr"]
            metrics["doc_r10"], metrics["doc_fr10"] = utils.document_recall_metrics(
                doc_target[0], doc_target[1], k=10
            )
            metrics["doc_r30"], metrics["doc_fr30"] = utils.document_recall_metrics(
                doc_target[0], doc_target[1], k=30
            )
            metrics["num_questions"] = len(doc_target[1])
            metrics["mrr"] = (
                metrics["doc_mrr"] if selection_metric.startswith("doc_")
                else metrics["entity_mrr"]
            )
            logger.info(f"{'-' * 15} Test on {data_name} {'-' * 15}")
            query_utils.print_metrics(metrics, logger)
        if world_size > 1:
            metrics_list = [metrics]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]
        all_metrics[data_name] = metrics
        all_scores.append(metrics[selection_metric])
        graph.cpu()
        del graph, ent2docs, doc_ranker, entities_weight
        if device.type == "cuda":
            torch.cuda.empty_cache()
    utils.synchronize()
    return all_metrics if return_metrics else float(np.mean(all_scores))


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs")),
    config_name=os.environ.get("QUERYFLOW_CONFIG_NAME", "stage2_qa_finetune"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.train.timeout)
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        output_dir = HydraConfig.get().runtime.output_dir
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")
        output_dir_list = [output_dir]
    else:
        output_dir_list = [None]
    if utils.get_world_size() > 1:
        dist.broadcast_object_list(
            output_dir_list, src=0
        )  # Use the output dir from rank 0
    output_dir = output_dir_list[0]

    # Initialize the datasets in the each process, make sure they are processed
    if cfg.datasets.init_datasets:
        rel_emb_dim_list = utils.init_multi_dataset(
            cfg, utils.get_world_size(), utils.get_rank()
        )
        rel_emb_dim = set(rel_emb_dim_list)
        assert len(rel_emb_dim) == 1, (
            "All datasets should have the same relation embedding dimension"
        )
    else:
        assert cfg.datasets.feat_dim is not None, (
            "If datasets.init_datasets is False, cfg.datasets.feat_dim must be set"
        )
        rel_emb_dim = {cfg.datasets.feat_dim}
    if utils.get_rank() == 0:
        logger.info(
            f"Datasets {cfg.datasets.train_names} and {cfg.datasets.valid_names} initialized"
        )

    device = utils.get_device()
    model = instantiate(cfg.model, rel_emb_dim=rel_emb_dim.pop())

    if cfg.train.get("init_checkpoint") and not cfg.train.get("checkpoint"):
        _load_initial_weights(model, cfg.train.init_checkpoint)
    elif "checkpoint" in cfg.train and cfg.train.checkpoint is not None:
        if os.path.exists(cfg.train.checkpoint):
            state = torch.load(
                cfg.train.checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state["model"], strict=False)
        # Try to load the model from the remote dictionary
        else:
            model, _ = utils.load_model_from_pretrained(cfg.train.checkpoint)

    model = model.to(device)
    if utils.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(line)
        logger.info(f"Number of parameters: {num_params}")

    train_dataset_loader = GraphDatasetLoader(
        _phase_dataset_config(cfg, "train"),
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    valid_dataset_loader = GraphDatasetLoader(
        _phase_dataset_config(cfg, "valid"),
        cfg.datasets.valid_names,
        shuffle=False,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    test_dataset_loader = None
    if cfg.datasets.get("test_names"):
        test_dataset_loader = GraphDatasetLoader(
            _phase_dataset_config(cfg, "test"),
            cfg.datasets.test_names,
            shuffle=False,
            max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
            data_loading_workers=cfg.datasets.data_loading_workers,
        )
    evaluation_loaders = (
        {"dev": valid_dataset_loader, "test": test_dataset_loader}
        if test_dataset_loader is not None else valid_dataset_loader
    )

    best_metrics = train_and_validate(
        cfg,
        output_dir,
        model,
        train_dataset_loader,
        evaluation_loaders,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
    )
    if isinstance(best_metrics, dict) and "dev" in best_metrics:
        best_metrics = best_metrics["dev"]

    if cfg.train.do_eval and best_metrics is None:
        if utils.get_rank() == 0:
            logger.info(separator)
            logger.info("Evaluate on valid")
        best_metrics = test(
            cfg, model, valid_dataset_loader, device=device, return_metrics=True
        )

    # Save the model into the format for QA inference
    if (
        utils.is_main_process()
        and cfg.train.save_pretrained
        and cfg.train.num_epoch > 0
    ):
        pre_trained_dir = os.path.join(output_dir, "pretrained")
        utils.save_model_to_pretrained(model, cfg, pre_trained_dir)

    # Shutdown the dataset loaders
    train_dataset_loader.shutdown()
    valid_dataset_loader.shutdown()
    if test_dataset_loader is not None:
        test_dataset_loader.shutdown()

    utils.synchronize()
    utils.cleanup()


if __name__ == "__main__":
    main()
