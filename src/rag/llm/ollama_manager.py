import subprocess
import time

import requests


class OllamaManager:
    """
    Manage the local Ollama service.

    Responsibilities:
        - Check whether Ollama is running
        - Start Ollama when necessary
        - Wait until the API becomes available
        - Check whether the requested model exists

    This class does not generate LLM responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3.5:4b-q4_K_M",
    ):

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

        self.process = None

    # ========================================================
    # CHECK OLLAMA
    # ========================================================

    def is_running(self) -> bool:

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=3,
            )

            return response.status_code == 200

        except requests.RequestException:

            return False

    # ========================================================
    # START OLLAMA
    # ========================================================

    def start(self):

        print(
            "[OLLAMA] Ollama is not running.",
            flush=True,
        )

        print(
            "[OLLAMA] Starting Ollama...",
            flush=True,
        )

        try:

            self.process = subprocess.Popen(
                [
                    "ollama",
                    "serve",
                ],

                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,

                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                    )
                    else 0
                ),
            )

        except FileNotFoundError:

            raise RuntimeError(
                "Ollama executable was not found. "
                "Make sure Ollama is installed and "
                "available in PATH."
            )

    # ========================================================
    # WAIT FOR API
    # ========================================================

    def wait_until_ready(
        self,
        timeout: int = 30,
    ):

        print(
            "[OLLAMA] Waiting for API...",
            flush=True,
        )

        start = time.time()

        while (
            time.time() - start
            < timeout
        ):

            if self.is_running():

                print(
                    "[OLLAMA] API is ready.",
                    flush=True,
                )

                return

            time.sleep(0.5)

        raise RuntimeError(
            "Ollama started but the API did not "
            "become available within the timeout."
        )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    def model_available(self) -> bool:

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )

            response.raise_for_status()

            data = response.json()

            models = data.get(
                "models",
                [],
            )

            for model in models:

                name = model.get(
                    "name",
                    "",
                )

                if name == self.model_name:

                    return True

            return False

        except requests.RequestException:

            return False

    # ========================================================
    # ENSURE READY
    # ========================================================

    def ensure_ready(self):

        print(
            "[OLLAMA] Checking Ollama...",
            flush=True,
        )

        # ----------------------------------------------------
        # Already running
        # ----------------------------------------------------

        if self.is_running():

            print(
                "[OLLAMA] Ollama is already running.",
                flush=True,
            )

        # ----------------------------------------------------
        # Not running
        # ----------------------------------------------------

        else:

            self.start()

            self.wait_until_ready()

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        print(
            f"[OLLAMA] Checking model: "
            f"{self.model_name}",
            flush=True,
        )

        if not self.model_available():

            raise RuntimeError(
                f"Ollama model '{self.model_name}' "
                "is not installed.\n\n"
                f"Run:\n"
                f"ollama pull {self.model_name}"
            )

        print(
            f"[OLLAMA] Model available: "
            f"{self.model_name}",
            flush=True,
        )

        print(
            "[OLLAMA] Ready.",
            flush=True,
        )

    # ========================================================
    # STOP ONLY IF WE STARTED IT
    # ========================================================

    def shutdown(self):

        # We intentionally do not stop Ollama here.
        #
        # Ollama is a local service and keeping it alive
        # allows the next RAG request/application to reuse it.

        self.process = None