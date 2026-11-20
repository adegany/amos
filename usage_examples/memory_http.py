"""The HTTP calls used by the two introductory examples. Python standard library."""

import json
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class Memory:
    base_url: str
    actor: str
    scope: dict

    def _post(self, operation, body):
        request = Request(
            f"{self.base_url}/v1/{operation}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def remember(self, *, source_ref, source_type, evidence, atom_type, payload):
        # The host supplies a globally unique source event ID, stable across retries.
        evidence_id = f"evidence:{source_ref}"
        access = {"visibility": [self.actor]}
        return self._post("memory-transactions:commit", {
            "actor": self.actor,
            "scope": self.scope,
            "idempotency_key": source_ref,
            "evidence": [{
                "evidence_id": evidence_id,
                "source_type": source_type,
                "source_ref": source_ref,
                "payload": evidence,
                "access_policy": access,
            }],
            "atoms": [{
                "type": atom_type,
                "payload": payload,
                "evidence_refs": [evidence_id],
                "access_policy": access,
            }],
        })

    def recall(self, atom_type, **fields):
        packet = self._post("packets:retrieve", {
            "requester": self.actor,
            "scope": self.scope,
            "type_filter": [atom_type],
            "payload_filter": {key: [value] for key, value in fields.items()},
            "max_items": 8,
            "token_or_byte_budget": {"tokens": 3000},
        })
        return packet["items"]

    def evidence(self, evidence_id):
        return self._post("evidence:get", {
            "requester": self.actor,
            "scope": self.scope,
            "evidence_id": evidence_id,
        })["record"]
