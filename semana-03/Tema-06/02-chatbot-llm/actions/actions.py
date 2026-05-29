"""Acción personalizada que delega la respuesta a un LLM local servido por Ollama."""

import os
from typing import Any, Dict, List

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = (
    "Eres un asistente educativo en español para la materia de "
    "Procesamiento de Lenguaje Natural y Visión Computacional. "
    "Responde de forma breve, clara y en español."
)


class ActionLLMResponder(Action):
    def name(self) -> str:
        return "action_llm_responder"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        user_msg = tracker.latest_message.get("text", "") or ""
        prompt = f"{SYSTEM_PROMPT}\n\nUsuario: {user_msg}\nAsistente:"

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if not text:
                text = "No pude generar una respuesta."
        except requests.RequestException as exc:
            text = (
                "No pude contactar al LLM local. "
                f"Verifica que Ollama esté corriendo en {OLLAMA_URL}. "
                f"Detalle: {exc}"
            )

        dispatcher.utter_message(text=text)
        return []
