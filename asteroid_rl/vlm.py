"""VLM perception backend (Qwen) with geometry-stub fallback.

Planning-document path: Basilisk camera frame → VLM → structured JSON matching
``build_perception_stub``. When transformers/torch or weights are unavailable,
falls back to the geometry stub so training/play still run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import numpy as np

from asteroid_rl.perception import build_perception_stub

DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

VLM_SYSTEM_PROMPT = """You are a spacecraft optical navigation assistant for asteroid landing.
Given a camera image, reply with ONLY a JSON object (no markdown) with keys:
  target_visible: boolean — whether the asteroid/landing region is in frame
  landing_site_box: [xmin, ymin, xmax, ymax] normalized to [0,1] image coords
  hazard_score: float in [0,1] — lower is safer (commit threshold ~0.10)
  progress_assessment: short string describing visibility and centering
If the target is not visible, set landing_site_box to [0,0,0,0] and hazard_score to 1.0.
"""


def _parse_vlm_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a perception JSON object from model text.

    Args:
        text: Raw model output.

    Returns:
        Parsed dict or ``None`` if parsing fails.
    """
    if not text:
        return None
    text = text.strip()
    # Strip fenced code blocks if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    box = data.get("landing_site_box", [0, 0, 0, 0])
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        box = [0.0, 0.0, 0.0, 0.0]
    return {
        "target_visible": bool(data.get("target_visible", False)),
        "landing_site_box": [float(x) for x in box],
        "hazard_score": float(np.clip(float(data.get("hazard_score", 1.0)), 0.0, 1.0)),
        "progress_assessment": str(
            data.get("progress_assessment", "vlm response")
        ),
    }


class PerceptionBackend:
    """Callable perception source: ``geometry`` or ``vlm`` (+ auto fallback)."""

    def __init__(
        self,
        backend: str = "geometry",
        *,
        model_name: str = DEFAULT_VLM_MODEL,
        device: str = "cpu",
    ):
        """Create a perception backend.

        Args:
            backend: ``geometry``, ``vlm``, or ``auto`` (try VLM, else geometry).
            model_name: Hugging Face model id for Qwen-VL.
            device: Torch device string for VLM inference.
        """
        self.requested = str(backend).lower()
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self.active = "geometry"
        if self.requested in ("vlm", "auto"):
            self._try_load_vlm()

    def _try_load_vlm(self) -> None:
        """Attempt to load the VLM; leave ``active=geometry`` on failure."""
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:
            print(f"VLM deps unavailable ({exc}); using geometry perception stub")
            self.active = "geometry"
            return
        model_cls = None
        for name in (
            "AutoModelForVision2Seq",
            "AutoModelForImageTextToText",
            "AutoModel",
        ):
            try:
                mod = __import__("transformers", fromlist=[name])
                model_cls = getattr(mod, name)
                break
            except Exception:
                continue
        if model_cls is None:
            print("No compatible transformers VLM class; using geometry stub")
            self.active = "geometry"
            return
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self._model = model_cls.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self._model.to(self.device)
            self._model.eval()
            self.active = "vlm"
            print(f"Loaded VLM perception backend: {self.model_name} on {self.device}")
        except Exception as exc:
            print(f"VLM load failed ({exc}); using geometry perception stub")
            self._model = None
            self._processor = None
            self.active = "geometry"

    def __call__(
        self,
        *,
        position_N,
        velocity_N,
        sigma_BN,
        target_N,
        altitude_m: float,
        rgb: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Build a perception dict from geometry and/or a camera frame.

        Args:
            position_N: Hub inertial position.
            velocity_N: Hub inertial velocity.
            sigma_BN: Hub attitude MRP.
            target_N: Landing site.
            altitude_m: Altitude above terrain.
            rgb: Optional ``uint8`` HxWx3 camera frame.

        Returns:
            Perception JSON-like dict plus ``perception_source``.
        """
        geom = build_perception_stub(
            position_N=position_N,
            velocity_N=velocity_N,
            sigma_BN=sigma_BN,
            target_N=target_N,
            altitude_m=altitude_m,
        )
        if self.active != "vlm" or rgb is None or self._model is None:
            out = dict(geom)
            out["perception_source"] = "geometry"
            return out

        vlm_out = self._infer_vlm(rgb)
        if vlm_out is None:
            out = dict(geom)
            out["perception_source"] = "geometry_fallback"
            return out
        # Keep geometry debug fields for logging / diagnostics.
        vlm_out["site_uv"] = geom.get("site_uv")
        vlm_out["site_depth_m"] = geom.get("site_depth_m")
        vlm_out["lateral_miss_m"] = geom.get("lateral_miss_m")
        vlm_out["perception_source"] = "vlm"
        return vlm_out

    def _infer_vlm(self, rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run one VLM forward pass on an RGB frame.

        Args:
            rgb: ``uint8`` image array HxWx3.

        Returns:
            Parsed perception dict or ``None``.
        """
        try:
            from PIL import Image
            import torch
        except Exception:
            return None
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VLM_SYSTEM_PROMPT},
                ],
            }
        ]
        try:
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=256)
            out_text = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            return _parse_vlm_json(out_text)
        except Exception as exc:
            print(f"VLM inference failed ({exc}); falling back to geometry")
            return None
