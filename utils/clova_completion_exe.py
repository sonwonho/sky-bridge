import json

import requests

from env.config import HCX_CONFIG, HCX_DASH_CONFIG


class CompletionExecutor:
    def __init__(self, is_dash):
        if is_dash:
            self._host = HCX_DASH_CONFIG["HOST"]
            self._api_key = HCX_DASH_CONFIG["API_KEY"]
            self._api_key_primary_val = HCX_DASH_CONFIG["API_KEY_PRIMARY_VAL"]
            self._request_id = HCX_DASH_CONFIG["ID"]
            self._model_url = "/testapp/v1/chat-completions/HCX-DASH-001"
        else:
            self._host = HCX_CONFIG["HOST"]
            self._api_key = HCX_CONFIG["API_KEY"]
            self._api_key_primary_val = HCX_CONFIG["API_KEY_PRIMARY_VAL"]
            self._request_id = HCX_CONFIG["ID"]
            self._model_url = "/serviceapp/v1/chat-completions/HCX-003"

    def execute(self, completion_request, response_type="stream"):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
        }
        if response_type == "stream":
            headers["Accept"] = "text/event-stream"
            is_stream = True
        else:
            is_stream = False

        with requests.post(
            self._host + self._model_url,
            headers=headers,
            json=completion_request,
            stream=is_stream,
        ) as r:
            if response_type == "stream":
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data:"):
                            event_data = json.loads(decoded_line[len("data:") :])
                            if event_data.get("index", {}) == 0:
                                message_content = event_data.get("message", {}).get(
                                    "content", ""
                                )
                                yield message_content

            elif response_type == "single":
                final_answer = r.json()["result"]["message"]["content"]
                yield final_answer
