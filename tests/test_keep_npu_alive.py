import argparse
import unittest

import keep_npu_alive


class KeepNpuAliveTests(unittest.TestCase):
    def test_parse_args_uses_low_load_defaults(self):
        args = keep_npu_alive.parse_args([])

        self.assertEqual(args.device, "npu:0")
        self.assertEqual(args.interval, 5.0)
        self.assertEqual(args.size, 256)
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.log_every, 12)

    def test_estimated_tensor_bytes_uses_three_square_tensors(self):
        self.assertEqual(
            keep_npu_alive.estimate_tensor_bytes(size=256, dtype_name="float16"),
            256 * 256 * 2 * 3,
        )

    def test_normalize_device_accepts_integer_id(self):
        self.assertEqual(keep_npu_alive.normalize_device("1"), "npu:1")

    def test_normalize_device_keeps_explicit_npu_device(self):
        self.assertEqual(keep_npu_alive.normalize_device("npu:2"), "npu:2")

    def test_positive_float_rejects_non_finite_values(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    keep_npu_alive.positive_float(value)

    def test_legacy_config_translates_to_global_controller_parameters(self):
        config = keep_npu_alive.KeepAliveConfig(
            device="npu:2",
            interval=5.0,
            size=256,
            dtype_name="float16",
            log_every=12,
            warmup=3,
            once=False,
        )

        self.assertEqual(
            keep_npu_alive.controller_kwargs(config),
            {
                "npu_ids": [2],
                "interval": 5.0,
                "vram_to_keep": 256 * 256 * 2 * 3,
                "busy_threshold": -1,
            },
        )


if __name__ == "__main__":
    unittest.main()
