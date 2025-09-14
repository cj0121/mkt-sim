__utils_autoload__ = True
__all__ = ["WandbRegistry"]

import os
import json
import wandb
import joblib
import pickle
import hashlib
import platform
import sys
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta


class WandbRegistry:
    def __init__(
        self,
        env: str = "dev",
        base_project: str = "model-registry",
        base_registry_dir: str = "registry"
    ):
        """
        Hash-based model registry wrapper around W&B artifacts.

        Args:
            env: Environment name ("dev" or "prod").
            base_project: Base W&B project name (env will be appended).
            base_registry_dir: Base local dir (env subdir will be created).
        """
        self.env = env
        self.project = f"{base_project}-{env}"
        self.registry_dir = os.path.join(base_registry_dir, env)
        os.makedirs(self.registry_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Utility: compute model hash
    # -----------------------------------------------------------
    @staticmethod
    def _hash_model(model: Any) -> str:
        """Stable short hash for a Python object.

        We hash the serialized bytes plus the fully-qualified class name to
        avoid accidental collisions across different model classes with
        identical bytes on some serializers.
        """
        # Use pickle to serialize to bytes (joblib has no dumps API)
        try:
            data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # As fallback, dump to a temp file and read bytes
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix='.pkl', delete=True) as f:
                joblib.dump(model, f.name)
                data = open(f.name,'rb').read()
        fqcn = f"{type(model).__module__}.{type(model).__name__}".encode()
        h = hashlib.sha256()
        h.update(data)
        h.update(b"::")
        h.update(fqcn)
        return h.hexdigest()[:16]

    @staticmethod
    def _validate_name(name: str) -> None:
        import re
        if not re.match(r"^[A-Za-z0-9_\-]+$", name or ""):
            raise ValueError("name must contain only letters, numbers, '_', or '-' ")

    # -----------------------------------------------------------
    # Resolve version directory for a model hash or hash_semver
    # -----------------------------------------------------------
    def _resolve_version_dir(self, name: str, model_hash: str) -> str:
        base = os.path.join(self.registry_dir, name)
        exact = os.path.join(base, model_hash)
        if os.path.isdir(exact):
            return exact
        prefix = f"{model_hash}_"
        if os.path.isdir(base):
            for d in os.listdir(base):
                if d.startswith(prefix) and os.path.isdir(os.path.join(base, d)):
                    return os.path.join(base, d)
        raise FileNotFoundError(f"No local version dir found for hash {model_hash} under {base}")

    # -----------------------------------------------------------
    # Save model
    # -----------------------------------------------------------
    def save(
        self,
        model: Any,
        name: str,
        semantic_version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cutoff_date: Optional[str] = None,
        run_date: Optional[str] = None,
        metrics: Optional[Dict] = None,
        description: str = "",
        extra_meta: Optional[Dict] = None,
        run: Optional[wandb.sdk.wandb_run.Run] = None,
        tags: Optional[Union[str, list]] = None,
    ):
        """
        Save and version a model with W&B + local registry (hash-based).

        Args:
            model: Any Python object.
            name: Model family name (e.g., "sprint_model").
            semantic_version: Optional human-assigned semantic version.
            start_date, end_date: Training data range (YYYY-MM-DD).
            cutoff_date: First valid prediction date (default = end_date+1).
            run_date: Date of this run (YYYY-MM-DD).
            metrics: Dict of evaluation metrics.
            description: Free-text description.
            extra_meta: Extra metadata fields to include.
            run: Optional existing W&B run. If None, a new run is created.
            tags: Optional tags for run grouping.
        """
        # --- Validate required metadata ---
        if not start_date:
            raise ValueError("start_date must be provided and non-empty")
        if not end_date:
            raise ValueError("end_date must be provided and non-empty")
        if not run_date:
            raise ValueError("run_date must be provided and non-empty")

        # Default cutoff_date = end_date + 1
        if not cutoff_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                cutoff_date = (end_dt + timedelta(days=1)).date().isoformat()
            except Exception:
                raise ValueError("end_date must be a valid YYYY-MM-DD string")

        # --- Validate & Hash the model ---
        self._validate_name(name)
        model_hash = self._hash_model(model)

        # --- Run handling ---
        new_run_created = False
        if run is None:
            wandb_mode = os.getenv("WANDB_MODE", "offline")
            run = wandb.init(project=self.project, mode=wandb_mode, tags=tags)
            new_run_created = True
        else:
            # Append tags to existing run if provided
            if tags:
                existing = set(run.tags or ())
                extra = {tags} if isinstance(tags, str) else set(tags)
                run.tags = tuple(existing.union(extra))

        # --- Temp file for model hand-off ---
        with TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, f"{name}_{model_hash}.pkl")
            joblib.dump(model, tmp_path)

            artifact = wandb.Artifact(name=name, type="model", description=description)
            artifact.add_file(tmp_path)

            # Required metadata
            artifact.metadata = {
                "hash": model_hash,
                "environment": self.env,
                "semantic_version": semantic_version,
                "framework": type(model).__module__,
                "class": type(model).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "cutoff_date": cutoff_date,
                "run_date": run_date,
            }
            # System context
            artifact.metadata.update({
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            })
            # Optional git commit
            try:
                commit = subprocess.check_output([
                    "git", "rev-parse", "--short", "HEAD"
                ], stderr=subprocess.DEVNULL).decode().strip()
                if commit:
                    artifact.metadata["git_commit"] = commit
            except Exception:
                pass
            if metrics:
                artifact.metadata["metrics"] = metrics
            if extra_meta:
                artifact.metadata.update(extra_meta)

            run.log_artifact(artifact)

        if new_run_created:
            run.finish()

        # --- Local Registry Copy ---
        # Optional: include semver in folder name for readability
        folder = model_hash if not semantic_version else f"{model_hash}_{semantic_version}"
        version_dir = os.path.join(self.registry_dir, name, folder)
        os.makedirs(version_dir, exist_ok=True)

        registry_model_path = os.path.join(version_dir, "model.pkl")
        joblib.dump(model, registry_model_path)

        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(artifact.metadata, f, indent=2)

        return {"hash": model_hash, "registry_model": registry_model_path, "registry_meta": metadata_path}

    # -----------------------------------------------------------
    # Load model from local registry
    # -----------------------------------------------------------
    def load_local(self, name: str, model_hash: str) -> Any:
        version_dir = self._resolve_version_dir(name, model_hash)
        path = os.path.join(version_dir, "model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No local model file at {path}")
        return joblib.load(path)

    # -----------------------------------------------------------
    # List hashes
    # -----------------------------------------------------------
    def list_hashes(self, name: str):
        path = os.path.join(self.registry_dir, name)
        if not os.path.exists(path):
            return []
        return sorted(os.listdir(path))

    def latest_hash(self, name: str) -> Optional[str]:
        """Return the hash (or hash_semver folder) with the newest timestamp in metadata."""
        base = os.path.join(self.registry_dir, name)
        if not os.path.isdir(base):
            return None
        choices = []
        for h in os.listdir(base):
            meta = os.path.join(base, h, "metadata.json")
            if os.path.isfile(meta):
                try:
                    with open(meta) as f:
                        ts = json.load(f).get("timestamp", "")
                    choices.append((ts, h))
                except Exception:
                    continue
        return max(choices)[1] if choices else None

    # -----------------------------------------------------------
    # Promote model
    # -----------------------------------------------------------
    def promote(self, target_registry: "WandbRegistry", name: str, model_hash: str):
        model = self.load_local(name, model_hash)
        version_dir = self._resolve_version_dir(name, model_hash)
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)

        metadata["promoted_from"] = {"env": self.env, "hash": model_hash}
        return target_registry.save(
            model,
            name=name,
            semantic_version=metadata.get("semantic_version"),
            start_date=metadata["start_date"],
            end_date=metadata["end_date"],
            cutoff_date=metadata["cutoff_date"],
            run_date=metadata["run_date"],
            metrics=metadata.get("metrics"),
            description=f"Promoted from {self.env} ({model_hash})",
            extra_meta=metadata,
        )
