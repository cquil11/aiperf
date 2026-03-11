# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class WildChatDatasetLoader(BaseFileLoader):
    """Dataset loader for WildChat conversation format.

    Loads multi-turn conversations from the WildChat dataset format used by
    InferenceX benchmarks. Each entry contains real user-assistant conversation
    pairs with natural language content and realistic turn distributions.

    Expected JSON format (array of objects):
    ```json
    [
        {
            "conversation_hash": "abc123",
            "turn_count": 3,
            "user_token_count": 1500,
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well."},
                {"role": "user", "content": "Great!"},
                {"role": "assistant", "content": "Thanks!"}
            ]
        }
    ]
    ```

    Each conversation is converted to an AIPerf Conversation with one Turn per
    user message. The assistant responses are stored as raw_messages context so
    that subsequent turns include the full conversation history (enabling prefix
    cache reuse on the inference server).
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format."""
        if data is None:
            return False
        return (
            isinstance(data, dict)
            and "conversation" in data
            and "conversation_hash" in data
        )

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """WildChat conversations are best sampled sequentially."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load WildChat conversations from a JSON file.

        Returns:
            A list of raw conversation dictionaries.
        """
        with open(self.filename, "rb") as f:
            data = orjson.loads(f.read())

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array of conversations, got {type(data).__name__}"
            )

        return data

    def convert_to_conversations(
        self, data: list[dict[str, Any]]
    ) -> list[Conversation]:
        """Convert WildChat format to AIPerf Conversation objects.

        Each user message becomes a Turn. The Turn's raw_messages field contains
        the full conversation history up to and including that user message,
        enabling the chat endpoint to send the complete context (which allows
        prefix caching on the server).
        """
        conversations = []

        for entry in data:
            session_id = entry.get("conversation_hash", self.session_id_generator.next())
            messages = entry.get("conversation", [])

            if not messages:
                continue

            # Count user turns in this conversation
            user_turn_count = sum(1 for m in messages if m.get("role") == "user")
            if user_turn_count < 2:
                continue  # Skip single-turn conversations (incompatible with user-centric mode)

            conversation = Conversation(session_id=session_id)

            # Build turns: each user message = one turn, with full history as raw_messages
            history: list[dict[str, str]] = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                history.append({"role": role, "content": content})

                if role == "user":
                    # Create a turn with the full conversation history up to this point.
                    # raw_messages bypasses normal turn construction and sends the
                    # messages array directly to the chat completions API.
                    turn = Turn(
                        raw_messages=[dict(m) for m in history],
                        texts=[Text(name="text", contents=[content])],
                    )
                    conversation.turns.append(turn)

            if conversation.turns:
                conversations.append(conversation)

        return conversations
