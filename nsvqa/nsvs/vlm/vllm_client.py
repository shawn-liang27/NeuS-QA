from openai import OpenAI
import numpy as np
import base64
import time
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor

from nsvqa.utils.sigmoid import calibrate_sigmoid
from nsvqa.nsvs.vlm.obj import DetectedObject

logging.basicConfig(level=logging.INFO)


class VLLMClient:
    """
    VLM client that talks to a local vLLM server via its OpenAI-compatible API.
    The vLLM server handles KV cache / prefix caching automatically (VLLM_USE_V1=1).
    """

    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        model="OpenGVLab/InternVL3_5-14B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def _encode_frame(self, frame: np.ndarray) -> str:
        ret, buffer = cv2.imencode(".png", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def _build_prompt(self, scene_description: str) -> tuple[str, str]:
        """Returns (clean_name, prompt_text) for a scene description."""
        if "subtitle" in scene_description:
            clean = scene_description.replace("subtitle_", "").replace("_", " ")
            parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Does the video have the subtitle 'this is very interesting' present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
            prompt = rf"Does the video have the subtitle '{clean}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
        else:
            name = scene_description
            if name.startswith("n_"):
                name = name[2:]
            clean = name.replace("_", " ")
            parsing_rule = "You must only return a Yes or No, and not both, to any question asked. You must not include any other symbols, information, text, justification in your answer or repeat Yes or No multiple times. For example, if the question is \"Is there a cat present in the sequence of images?\", the answer must only be 'Yes' or 'No'."
            prompt = rf"Is there a '{clean}' present in the sequence of images? " f"\n[PARSING RULE]: {parsing_rule}"
        return name, prompt

    def _single_detect(self, prompt: str, image_content: list[dict]) -> dict:
        """
        Make one chat completion call to the vLLM server.
        Returns {"detected": bool, "confidence": float, "wall_time": float}.
        """
        user_content = [
            {"type": "text", "text": "The following is the sequence of images"},
            *image_content,
            {"type": "text", "text": prompt},
        ]

        t0 = time.perf_counter()
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        wall_time = time.perf_counter() - t0

        content = chat_response.choices[0].message.content
        is_detected = "yes" in content.lower()

        top_logprobs_list = chat_response.choices[0].logprobs.content[0].top_logprobs
        token_prob_map = {}
        for top_logprob in top_logprobs_list:
            token_text = top_logprob.token.strip()
            token_prob_map[token_text] = np.exp(top_logprob.logprob)

        yes_prob = token_prob_map.get("Yes", 0.0)
        no_prob = token_prob_map.get("No", 0.0)

        if yes_prob + no_prob > 0:
            confidence = yes_prob / (yes_prob + no_prob)
        else:
            raise ValueError(f"No 'Yes'/'No' logprobs in response: {content}")

        return {"detected": is_detected, "confidence": confidence, "wall_time": wall_time}

    def batch_detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_descriptions: list[str],
        threshold: float,
    ) -> list[DetectedObject]:
        """
        Detect all propositions for a frame window in one batched call.
        Fires concurrent requests to vLLM — prefix caching reuses the
        visual KV across all propositions sharing the same frames.
        """
        batch_t0 = time.perf_counter()

        encoded_images = [self._encode_frame(frame) for frame in seq_of_frames]
        image_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{enc}"}}
            for enc in encoded_images
        ]

        tasks = [self._build_prompt(desc) for desc in scene_descriptions]

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [
                executor.submit(self._single_detect, prompt, image_content)
                for _, prompt in tasks
            ]
            results = [f.result() for f in futures]

        batch_wall = time.perf_counter() - batch_t0

        per_request_times = [r["wall_time"] for r in results]
        logging.info(
            f"[VLLMClient] batch_detect: {len(tasks)} propositions | "
            f"total wall={batch_wall:.3f}s | "
            f"per-request=[{', '.join(f'{t:.3f}s' for t in per_request_times)}] | "
            f"sum={sum(per_request_times):.3f}s (speedup={sum(per_request_times)/batch_wall:.1f}x)"
        )

        detected_objects = []
        for (name, _), result in zip(tasks, results):
            probability = calibrate_sigmoid(result["confidence"], false_threshold=threshold)
            detected_objects.append(DetectedObject(
                name=name,
                is_detected=result["detected"],
                confidence=round(result["confidence"], 3),
                probability=round(probability, 3),
            ))

        return detected_objects

    def detect(
        self,
        seq_of_frames: list[np.ndarray],
        scene_description: str,
        threshold: float,
    ) -> DetectedObject:
        """Single-proposition detect (kept for backwards compatibility)."""
        return self.batch_detect(seq_of_frames, [scene_description], threshold)[0]

    def clear_gpu_memory(self) -> None:
        """No-op: vLLM manages GPU memory server-side."""
        pass
