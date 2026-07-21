from keep_npu.utilities.npu_info import parse_npu_smi_output

SAMPLE = """
| NPU   Name          Health | Power(W) Temp(C) | AICore(%) Memory-Usage(MB) |
| 0     Ascend 910B   OK     | 88.0     41      | 12        4096 / 65536    |
| 1     Ascend 910B   OK     | 91.0     42      | 101       70000 / 65536   |
"""


def test_parse_npu_smi_output_normalizes_records():
    assert parse_npu_smi_output(SAMPLE) == [
        {
            "physical_id": 0,
            "name": "Ascend 910B",
            "memory_total": 65536 * 1024**2,
            "memory_used": 4096 * 1024**2,
            "utilization": 12,
        },
        {
            "physical_id": 1,
            "name": "Ascend 910B",
            "memory_total": 65536 * 1024**2,
            "memory_used": None,
            "utilization": None,
        },
    ]


def test_parse_npu_smi_output_ignores_unrelated_lines():
    assert parse_npu_smi_output("npu-smi 24.1.0\nno devices") == []

