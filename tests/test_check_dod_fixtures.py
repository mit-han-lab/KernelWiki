import importlib.util
import copy
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_dod_fixtures", ROOT / "scripts" / "check_dod_fixtures.py"
)
DOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOD)


class DefinitionOfDoneFixtureTests(unittest.TestCase):
    def test_current_active_and_retired_contracts_are_valid(self):
        data = yaml.safe_load((ROOT / "data" / "phase3-dod-fixtures.yaml").read_text())
        self.assertEqual(3, data["contract_version"])
        self.assertFalse(DOD.check_fixture_contract(data))
        self.assertFalse([
            error
            for entry in data["fixtures"]
            for error in DOD.check_entry(entry)
        ])
        self.assertFalse([
            error
            for entry in data["retired_fixtures"]
            for error in DOD.check_retired_entry(entry)
        ])

    def test_retirement_cannot_silently_drop_reason_or_asset_history(self):
        errors = DOD.check_retired_entry({"question": "missing evidence"})
        self.assertTrue(any("reason" in error for error in errors))
        self.assertTrue(any("former_required_assets" in error for error in errors))

    def test_active_fixture_cannot_be_silently_deleted(self):
        data = yaml.safe_load((ROOT / "data" / "phase3-dod-fixtures.yaml").read_text())
        mutated = copy.deepcopy(data)
        removed = mutated["fixtures"].pop()
        errors = DOD.check_fixture_contract(mutated)
        self.assertTrue(any(removed["id"] in error for error in errors))

    def test_fixture_and_roster_row_cannot_be_jointly_deleted(self):
        data = yaml.safe_load((ROOT / "data" / "phase3-dod-fixtures.yaml").read_text())
        mutated = copy.deepcopy(data)
        removed = mutated["fixtures"].pop()
        del mutated["fixture_roster"][removed["id"]]
        errors = DOD.check_fixture_contract(mutated)
        self.assertTrue(any("pinned contract-v3 identities" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
