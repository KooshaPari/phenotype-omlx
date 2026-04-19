#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Entry point for hwLedger JSON-RPC RPC server.

Invoked as: python -m omlx.__main_hwledger__
or via uv: uv run --project . python -m omlx.__main_hwledger__
"""

from omlx.hwledger_rpc import main

if __name__ == "__main__":
    main()
