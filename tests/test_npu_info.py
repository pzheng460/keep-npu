from keep_npu.utilities.npu_info import parse_npu_smi_output

SAMPLE = """
| NPU   Name          Health | Power(W) Temp(C) | AICore(%) Memory-Usage(MB) |
| 0     Ascend 910B   OK     | 88.0     41      | 12        4096 / 65536    |
| 1     Ascend 910B   OK     | 91.0     42      | 101       70000 / 65536   |
"""

ASCEND_25_SAMPLE = """
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
| 0     910B2               | OK            | 97.1        50                0    / 0             |
| 0                         | 0000:C1:00.0  | 12          0    / 0          7752 / 65536         |
+===========================+===============+====================================================+
| 1     910B1               | Warning       | 96.8        50                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3440 / 65536         |
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


def test_parse_npu_smi_output_accepts_ascend_25_two_line_records():
    assert parse_npu_smi_output(ASCEND_25_SAMPLE) == [
        {
            "physical_id": 0,
            "name": "910B2",
            "memory_total": 65536 * 1024**2,
            "memory_used": 7752 * 1024**2,
            "utilization": 12,
        },
        {
            "physical_id": 1,
            "name": "910B1",
            "memory_total": 65536 * 1024**2,
            "memory_used": 3440 * 1024**2,
            "utilization": 0,
        },
    ]
