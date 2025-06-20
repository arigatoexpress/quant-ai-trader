# -*- coding: utf-8 -*-
"""Simplified ElizaOS configuration loader."""
import json
import logging
import os
import re
import hvac
import requests
import yaml


class ConfigLoaderError(Exception):
    def __init__(self, message=""):
        logging.getLogger(__name__).error(message)
        self.message = message


class ConfigLoader:
    """Loads YAML config files and optionally Vault secrets."""

    def __init__(self, use_vault=True, vault_addr="", vault_token="", verify=True):
        self.use_vault = use_vault
        self.environ_pattern = re.compile(r"^<%= ENV\[\'(.*)\'\] %\>(.*)$")
        self.vault_pattern = re.compile(r"^<%= VAULT\[\'(.*)\'\] %\>(.*)$")
        self.vault_addr = vault_addr or os.environ.get("VAULT_ADDR", "")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN", "")
        self.client = self._get_vault_client(verify)

    @staticmethod
    def load_application_info(path):
        with open(os.path.join(path, "info.json"), "r") as f:
            return json.load(f)

    def load_config(self, path, environments):
        if not path.endswith("/"):
            path += "/"
        if not isinstance(environments, list):
            environments = [environments]
        yaml.add_implicit_resolver("!environ", self.environ_pattern)
        yaml.add_constructor("!environ", self._get_from_environment)
        yaml.add_implicit_resolver("!vault", self.vault_pattern)
        yaml.add_constructor("!vault", self._get_from_vault)
        config = {}
        try:
            for env in environments:
                with open(path + env + ".yaml", "r") as f:
                    env_config = yaml.load(f.read(), Loader=yaml.Loader) or {}
                config.update(env_config)
            return config
        except Exception as error:
            raise ConfigLoaderError(f"Failed loading config: {error}")

    def _get_vault_client(self, verify):
        if not self.use_vault:
            return None
        try:
            return hvac.Client(
                url=self.vault_addr, token=self.vault_token, verify=verify
            )
        except Exception as error:
            raise ConfigLoaderError(f"Could not create vault client: {error}")

    def _get_from_environment(self, loader, node):
        value = loader.construct_scalar(node)
        env_var, rest = self.environ_pattern.match(value).groups()
        return os.environ.get(env_var, "") + rest

    def _get_from_vault(self, loader, node):
        value = loader.construct_scalar(node)
        vault_path, rest = self.vault_pattern.match(value).groups()
        if self.use_vault and self.client:
            try:
                return self.client.read(vault_path)["data"]["value"] + rest
            except Exception:
                raise ConfigLoaderError(f"Vault read failed: {vault_path}")
        return rest
