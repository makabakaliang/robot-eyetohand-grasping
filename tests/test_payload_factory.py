"""Tests for QueryPayloadFactory JSON command builders."""
import json
import pytest
from json_payloads import QueryPayloadFactory


def test_joint_angle_query_structure():
    p = QueryPayloadFactory.create_joint_angle_payload(pack_id="7")
    assert p["reqType"] == "query"
    assert p["packID"] == "7"
    assert p["queryAddr"] == ["axis-0", "axis-1", "axis-2", "axis-3", "axis-4", "axis-5"]
    # must be JSON-serializable
    json.dumps(p)


def test_world_coord_query_has_six_axes():
    p = QueryPayloadFactory.create_world_coord_payload()
    assert len(p["queryAddr"]) == 6
    assert p["queryAddr"][0].startswith("world-")


def test_stop_is_command_not_query():
    p = QueryPayloadFactory.create_robot_stop()
    assert p["reqType"] == "command"
    assert p["cmdData"][0] == "actionStop"


def test_sport_payload_injects_m0_to_m7():
    values = ["0.000"] * 8
    values[4] = "-90.000"
    p = QueryPayloadFactory.create_sport_payload(pack_id="1234", m_values=values, speed="100.0")
    assert p["reqType"] == "AddRCC"
    instr = p["instructions"][0]
    for i in range(8):
        assert instr[f"m{i}"] == values[i]
    assert instr["speed"] == "100.0"


def test_sport_payload_defaults_to_zero_joints():
    p = QueryPayloadFactory.create_sport_payload()
    instr = p["instructions"][0]
    assert instr["m0"] == "0.000"
    assert instr["m7"] == "0.000"


def test_add_points_payload_ds_data_shape():
    p = QueryPayloadFactory.create_add_points_payload(cam_id="0")
    assert p["reqType"] == "AddPoints"
    assert len(p["dsData"]) == 1
    assert p["dsData"][0]["camID"] == "0"
    assert "ModelID" in p["dsData"][0]["data"][0]


def test_photo_payload_default_success():
    p = QueryPayloadFactory.create_photo_payload()
    assert p["ret"] == 1
    assert p["camID"] == 0
