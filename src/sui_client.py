from pysui import SuiConfig, SyncClient


class SuiClient:
    """Wrapper around pysui SyncClient with simple helpers."""

    def __init__(self, rpc_url: str = "https://fullnode.mainnet.sui.io:443"):
        self.rpc_url = rpc_url
        try:
            cfg = SuiConfig.user_config(rpc_url=rpc_url)
            self.client = SyncClient(cfg)
        except Exception:
            self.client = None

    def current_gas_price(self) -> float | None:
        """Return current gas price in MIST or ``None`` if unavailable."""
        if not self.client:
            return None
        try:
            return self.client.current_gas_price()
        except Exception:
            return None

    def chain_id(self) -> str | None:
        """Return the chain identifier if available."""
        if not self.client:
            return None
        try:
            return self.client.get_chain_id()
        except Exception:
            return None
