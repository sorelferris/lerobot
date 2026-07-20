#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Interactive TUI Dashboard for managing multiple LeRobot policy servers.
Supports auto-detection of checkpoints, port assignment, device allocation,
real-time log viewing, and persistence.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import torch
import yaml
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, RichLog, Select, Static

# Default config path
CONFIG_PATH = Path.home() / ".cache" / "lerobot" / "policy_dashboard_config.yaml"


# ==============================================================================
# Helper Functions
# ==============================================================================


def find_free_port(start_port: int = 9000) -> int | None:
    """Find a free port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return None


def get_cuda_devices() -> list[str]:
    """Get list of available PyTorch devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices


def detect_policy_type_from_config(config_path: Path) -> str:
    """Read a policy config.json and try to parse the policy_type/type."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Standard keys for policy types in LeRobot configs
            for key in ["type", "policy_type", "_choice_name"]:
                if key in data and isinstance(data[key], str):
                    return data[key]
    except Exception:
        pass
    return "act"  # Default fallback


def scan_checkpoints() -> list[dict]:
    """Scan local workspace and HF cache for policy checkpoints."""
    checkpoints = []
    seen_paths = set()

    # 1. Search in local workspace outputs directory
    workspace_outputs = Path("outputs/train")
    if workspace_outputs.exists():
        for path in workspace_outputs.rglob("config.json"):
            ckpt_path = path.parent
            # Ignore duplicate configurations
            if str(ckpt_path) in seen_paths:
                continue

            # Check if this is inside a "pretrained_model" directory
            if ckpt_path.name == "pretrained_model":
                display_name = f"{ckpt_path.parent.parent.name}/{ckpt_path.parent.name} (Local)"
                value_path = str(ckpt_path)
            else:
                display_name = f"{ckpt_path.name} (Local)"
                value_path = str(ckpt_path)

            policy_type = detect_policy_type_from_config(path)
            checkpoints.append(
                {"name": display_name, "path": value_path, "policy_type": policy_type, "source": "local"}
            )
            seen_paths.add(value_path)

    # 2. Search in HF Cache directory
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache_dir.exists():
        for model_folder in hf_cache_dir.iterdir():
            if model_folder.is_dir() and model_folder.name.startswith("models--"):
                # Parse repo ID from models--author--name format
                parts = model_folder.name.split("--")
                if len(parts) >= 3:
                    org = parts[1]
                    name = parts[2]
                    repo_id = f"{org}/{name}"
                else:
                    repo_id = model_folder.name[8:]

                snapshots_dir = model_folder / "snapshots"
                if snapshots_dir.exists():
                    for snap_folder in snapshots_dir.iterdir():
                        config_file = snap_folder / "config.json"
                        # Or check within pretrained_model folder inside snapshot
                        config_file_sub = snap_folder / "pretrained_model" / "config.json"

                        final_config = None
                        final_path = None
                        if config_file.exists():
                            final_config = config_file
                            final_path = snap_folder
                        elif config_file_sub.exists():
                            final_config = config_file_sub
                            final_path = snap_folder / "pretrained_model"

                        if final_config and final_path:
                            path_str = str(final_path)
                            if path_str not in seen_paths:
                                policy_type = detect_policy_type_from_config(final_config)
                                checkpoints.append(
                                    {
                                        "name": f"{repo_id} (HF Cache)",
                                        "path": path_str,
                                        "policy_type": policy_type,
                                        "source": "hf_cache",
                                        "repo_id": repo_id,
                                    }
                                )
                                seen_paths.add(path_str)
                                break  # Just grab first snapshot to keep list clean

    # 3. Add default popular Presets if not already found
    presets = [
        {
            "name": "lerobot/diffusion_pusht (Preset)",
            "path": "lerobot/diffusion_pusht",
            "policy_type": "diffusion",
            "source": "preset",
        },
        {
            "name": "lerobot/act_aloha_sim_transfer_cube_human (Preset)",
            "path": "lerobot/act_aloha_sim_transfer_cube_human",
            "policy_type": "act",
            "source": "preset",
        },
    ]
    for p in presets:
        if p["path"] not in seen_paths:
            checkpoints.append(p)
            seen_paths.add(p["path"])

    return checkpoints


# ==============================================================================
# Custom Widgets
# ==============================================================================


class ServerItem(ListItem):
    """List item representing a Policy Server."""

    def __init__(self, server_id: str, name: str, status: str, port: int):
        super().__init__()
        self.server_id = server_id
        self.server_name = name
        self.status = status
        self.port = port
        self.label = Static("")
        self._update_label_text()

    def compose(self) -> ComposeResult:
        yield self.label

    def update_server_info(self, name: str, status: str, port: int):
        self.server_name = name
        self.status = status
        self.port = port
        self._update_label_text()

    def _update_label_text(self):
        status_markers = {
            "Stopped": "○ Stopped ",
            "Starting": "▲ Starting",
            "Running": "● Running ",
            "Failed": "■ Failed  ",
        }
        marker = status_markers.get(self.status, "○ Stopped ")

        status_colors = {"Stopped": "gray", "Starting": "yellow", "Running": "green", "Failed": "red"}
        color = status_colors.get(self.status, "white")

        text = f"[{color}]{marker}[/] [bold]{self.server_name}[/] [dim](Port: {self.port})[/]"
        self.label.update(text)


# ==============================================================================
# Main Dashboard Application
# ==============================================================================


class PolicyDashboardApp(App):
    """Textual TUI for managing multiple policy servers."""

    TITLE = "🤗 LeRobot Policy Server Dashboard"
    SUB_TITLE = "Multi-Server TUI Management & Auto-Detection Tool"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "start_selected", "Start Selected", show=True),
        Binding("x", "stop_selected", "Stop Selected", show=True),
        Binding("a", "add_new_server", "Add Server", show=True),
        Binding("d", "delete_selected", "Delete Server", show=True),
    ]

    # Style sheet matching the requirements
    CSS = """
    Screen {
        background: $background;
    }

    #app-container {
        layout: horizontal;
        height: 100%;
    }

    #sidebar {
        width: 38;
        height: 100%;
        border-right: solid $primary;
        padding: 1;
        background: $surface;
    }

    #main-panel {
        width: 1fr;
        height: 100%;
        padding: 1;
    }

    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }

    #server-list {
        height: 1fr;
        border: solid $primary-muted;
        background: $background;
        margin-bottom: 1;
    }

    .form-container {
        border: solid $primary-muted;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
        height: auto;
    }

    .form-row {
        layout: horizontal;
        height: auto;
        align: left middle;
        margin-bottom: 1;
    }

    .form-label {
        width: 15;
        text-align: right;
        margin-right: 2;
        text-style: bold;
        color: $accent;
    }

    .form-input {
        width: 1fr;
    }

    .button-row {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    .action-btn {
        margin-left: 1;
    }

    #log-container {
        height: 1fr;
        border: solid $accent;
        background: $background;
    }

    #log-view {
        height: 100%;
        padding: 0 1;
    }

    .section-title {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self):
        super().__init__()
        # State
        self.servers: dict[str, dict] = {}  # server_id -> config_dict
        self.processes: dict[str, subprocess.Popen] = {}  # server_id -> Popen object
        self.log_buffers: dict[str, list[str]] = {}  # server_id -> lines list
        self.discovered_checkpoints: list[dict] = []
        self.selected_server_id: str | None = None
        self.cuda_devices: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="app-container"):
            # LEFT SIDEBAR
            with Vertical(id="sidebar"):
                yield Label("ACTIONS", classes="section-title")
                yield Button(
                    "Auto Detect & Configure",
                    id="btn-auto-detect",
                    variant="primary",
                    classes="sidebar-button",
                )
                yield Button(
                    "Add New Server", id="btn-add-server", variant="success", classes="sidebar-button"
                )
                yield Button(
                    "Save Dashboard Config", id="btn-save-all", variant="default", classes="sidebar-button"
                )

                yield Label("SERVERS LIST", classes="section-title")
                yield ListView(id="server-list")

                yield Static("[dim]Choose a server to configure or view its logs.[/]")

            # RIGHT MAIN PANEL
            with Vertical(id="main-panel"):
                yield Label("SERVER CONFIGURATION", classes="section-title")

                # Form Container
                with Vertical(classes="form-container", id="server-form"):
                    with Horizontal(classes="form-row"):
                        yield Label("Server Name:", classes="form-label")
                        yield Input(placeholder="e.g., PushT-Diffusion", id="inp-name", classes="form-input")

                    with Horizontal(classes="form-row", id="checkpoint-select-row"):
                        yield Label("Checkpoint:", classes="form-label")
                        yield Select([], id="sel-checkpoint", classes="form-input")

                    with Horizontal(classes="form-row", id="custom-checkpoint-row"):
                        yield Label("Custom Path/ID:", classes="form-label")
                        yield Input(
                            placeholder="e.g., lerobot/diffusion_pusht or path/to/pretrained_model",
                            id="inp-custom-checkpoint",
                            classes="form-input",
                        )

                    with Horizontal(classes="form-row"):
                        yield Label("Policy Type:", classes="form-label")
                        yield Select(
                            [(t, t) for t in ["act", "diffusion", "pi0", "smolvla", "groot", "tdmpc"]],
                            id="sel-policy-type",
                            classes="form-input",
                        )

                    with Horizontal(classes="form-row"):
                        yield Label("Device:", classes="form-label")
                        yield Select([], id="sel-device", classes="form-input")

                    with Horizontal(classes="form-row"):
                        yield Label("Server Port:", classes="form-label")
                        yield Input(placeholder="e.g., 8000", id="inp-port", classes="form-input")

                    # Form Control Buttons
                    with Horizontal(classes="button-row"):
                        yield Button(
                            "Save Settings", id="btn-save-server", variant="primary", classes="action-btn"
                        )
                        yield Button("Start Server", id="btn-start", variant="success", classes="action-btn")
                        yield Button("Stop Server", id="btn-stop", variant="error", classes="action-btn")
                        yield Button("Delete Server", id="btn-delete", variant="error", classes="action-btn")

                yield Label("LIVE CONSOLE LOGS", classes="section-title")
                with Container(id="log-container"):
                    yield RichLog(id="log-view", highlight=True, max_lines=1000)

        yield Footer()

    # ==============================================================================
    # Lifecycle and Initialization
    # ==============================================================================

    def on_mount(self) -> None:
        """Called when app is mounted. Load hardware devices and config."""
        # 1. Get available devices
        self.cuda_devices = get_cuda_devices()
        device_select = self.query_one("#sel-device", Select)
        device_select.set_options([(d, d) for d in self.cuda_devices])

        # 2. Scan checkpoints
        self.discovered_checkpoints = scan_checkpoints()
        self._update_checkpoint_select_options()

        # 3. Load persisted config
        self._load_dashboard_config()

        # 4. Hide Custom Path/ID row by default
        self.query_one("#custom-checkpoint-row").visible = False

        # 5. Populate servers list
        self._refresh_servers_list()

    def on_unmount(self) -> None:
        """Called when app exits. Kill all running server processes."""
        for server_id, process in list(self.processes.items()):
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        self.processes.clear()

    # ==============================================================================
    # Config Persistence
    # ==============================================================================

    def _load_dashboard_config(self):
        """Load servers from the YAML configuration file."""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

                loaded_servers = config.get("servers", [])
                self.servers.clear()

                for s in loaded_servers:
                    server_id = s.get("id", str(uuid.uuid4()))
                    self.servers[server_id] = {
                        "id": server_id,
                        "name": s.get("name", "Unnamed Server"),
                        "checkpoint": s.get("checkpoint", ""),
                        "policy_type": s.get("policy_type", "act"),
                        "device": s.get("device", "cpu"),
                        "port": int(s.get("port", 8000)),
                        "status": "Stopped",
                    }
                    self.log_buffers[server_id] = [f"--- Config loaded for {s.get('name')} ---\n"]
            except Exception as e:
                self.log_to_console(f"Error loading config: {e}\n")

    def _save_dashboard_config(self):
        """Save servers configuration to the YAML configuration file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_data = {
                "servers": [
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "checkpoint": s["checkpoint"],
                        "policy_type": s["policy_type"],
                        "device": s["device"],
                        "port": s["port"],
                    }
                    for s in self.servers.values()
                ]
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
            self.log_to_console("[System] Saved dashboard configuration successfully to YAML file.\n")
        except Exception as e:
            self.log_to_console(f"[System] [red]Failed to save config: {e}[/]\n")

    # ==============================================================================
    # Checkpoint Dropdown Logic
    # ==============================================================================

    def _update_checkpoint_select_options(self):
        """Populate the checkpoint Select options."""
        select_options = []
        for ckpt in self.discovered_checkpoints:
            select_options.append((ckpt["name"], ckpt["path"]))

        # Add special custom option
        select_options.append(("Custom Path or Repo ID...", "custom"))

        checkpoint_select = self.query_one("#sel-checkpoint", Select)
        checkpoint_select.set_options(select_options)

    # ==============================================================================
    # Servers Management & Form population
    # ==============================================================================

    def _refresh_servers_list(self):
        """Rebuild the server list on the sidebar."""
        list_view = self.query_one("#server-list", ListView)
        list_view.clear()

        for server_id, s in self.servers.items():
            item = ServerItem(server_id, s["name"], s["status"], s["port"])
            list_view.append(item)

        # Select first item if available and none selected
        if self.servers and self.selected_server_id is None:
            first_id = list(self.servers.keys())[0]
            self.selected_server_id = first_id
            list_view.index = 0
            self._populate_form_with_server(first_id)

    def _populate_form_with_server(self, server_id: str):
        """Fill form widgets with values from the selected server."""
        if server_id not in self.servers:
            return

        s = self.servers[server_id]
        self.selected_server_id = server_id

        # Name
        self.query_one("#inp-name", Input).value = s["name"]

        # Checkpoint Select & Custom Input
        checkpoint_val = s["checkpoint"]
        checkpoint_select = self.query_one("#sel-checkpoint", Select)
        custom_input = self.query_one("#inp-custom-checkpoint", Input)
        custom_row = self.query_one("#custom-checkpoint-row")

        # Check if the checkpoint matches a known path in options
        is_known = any(ckpt["path"] == checkpoint_val for ckpt in self.discovered_checkpoints)
        if is_known:
            checkpoint_select.value = checkpoint_val
            custom_row.visible = False
            custom_input.value = ""
        else:
            checkpoint_select.value = "custom"
            custom_row.visible = True
            custom_input.value = checkpoint_val

        # Policy Type
        self.query_one("#sel-policy-type", Select).value = s["policy_type"]

        # Device
        device_select = self.query_one("#sel-device", Select)
        if s["device"] in self.cuda_devices:
            device_select.value = s["device"]
        else:
            device_select.value = "cpu"

        # Port
        self.query_one("#inp-port", Input).value = str(s["port"])

        # Update Live Console view
        self._refresh_log_view_for_selected()

    def _refresh_log_view_for_selected(self):
        """Clear and redraw logs for the currently selected server."""
        log_view = self.query_one("#log-view", RichLog)
        log_view.clear()

        if self.selected_server_id and self.selected_server_id in self.log_buffers:
            buffer = self.log_buffers[self.selected_server_id]
            for line in buffer:
                log_view.write(line)

    def log_to_console(self, text: str, server_id: str | None = None):
        """Append line to server log buffer and optionally update active log view."""
        target_id = server_id or self.selected_server_id
        if not target_id:
            return

        if target_id not in self.log_buffers:
            self.log_buffers[target_id] = []

        # Add clean lines
        self.log_buffers[target_id].append(text)

        # If this is the currently viewed server, write to screen
        if target_id == self.selected_server_id:
            self.query_one("#log-view", RichLog).write(text)

    # ==============================================================================
    # Event Handlers
    # ==============================================================================

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Triggered when a server is selected in the sidebar list."""
        if event.item and isinstance(event.item, ServerItem):
            self._populate_form_with_server(event.item.server_id)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle Select dropdown changes."""
        if event.select.id == "sel-checkpoint":
            custom_row = self.query_one("#custom-checkpoint-row")
            if event.value == "custom":
                custom_row.visible = True
            else:
                custom_row.visible = False

                # Automatically set policy type based on selected checkpoint config if known
                selected_path = event.value
                for ckpt in self.discovered_checkpoints:
                    if ckpt["path"] == selected_path:
                        self.query_one("#sel-policy-type", Select).value = ckpt["policy_type"]
                        break

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses."""
        btn_id = event.button.id

        if btn_id == "btn-add-server":
            self.action_add_new_server()
        elif btn_id == "btn-auto-detect":
            self._handle_auto_detect()
        elif btn_id == "btn-save-all":
            self._save_dashboard_config()
        elif btn_id == "btn-save-server":
            self._handle_save_server()
        elif btn_id == "btn-start":
            self.action_start_selected()
        elif btn_id == "btn-stop":
            self.action_stop_selected()
        elif btn_id == "btn-delete":
            self.action_delete_selected()

    # ==============================================================================
    # Dashboard Features Logic
    # ==============================================================================

    def _handle_save_server(self):
        """Save settings from the form back to the selected server config."""
        if not self.selected_server_id:
            return

        # Extract form values
        name = self.query_one("#inp-name", Input).value.strip() or "Unnamed Server"

        checkpoint_select = self.query_one("#sel-checkpoint", Select).value
        if checkpoint_select == "custom":
            checkpoint = self.query_one("#inp-custom-checkpoint", Input).value.strip()
        else:
            checkpoint = checkpoint_select or ""

        policy_type = self.query_one("#sel-policy-type", Select).value or "act"
        device = self.query_one("#sel-device", Select).value or "cpu"

        try:
            port = int(self.query_one("#inp-port", Input).value.strip() or "8000")
        except ValueError:
            port = 8000

        # Update server config
        s = self.servers[self.selected_server_id]
        s["name"] = name
        s["checkpoint"] = checkpoint
        s["policy_type"] = policy_type
        s["device"] = device
        s["port"] = port

        # Save config
        self._save_dashboard_config()

        # Refresh UI
        self._refresh_servers_list()
        self._populate_form_with_server(self.selected_server_id)
        self.log_to_console(f"[System] Server '{name}' settings saved.\n")

    def action_add_new_server(self):
        """Create a new default server configuration."""
        server_id = str(uuid.uuid4())
        free_port = find_free_port(8000 + len(self.servers)) or 8000
        device = self.cuda_devices[1 % len(self.cuda_devices)] if len(self.cuda_devices) > 1 else "cpu"

        self.servers[server_id] = {
            "id": server_id,
            "name": f"New Policy Server {len(self.servers) + 1}",
            "checkpoint": "lerobot/diffusion_pusht",
            "policy_type": "diffusion",
            "device": device,
            "port": free_port,
            "status": "Stopped",
        }
        self.log_buffers[server_id] = ["--- Server Created ---\n"]
        self.selected_server_id = server_id

        self._refresh_servers_list()
        self._populate_form_with_server(server_id)
        self._save_dashboard_config()
        self.log_to_console("[System] Added new server config.\n")

    def action_delete_selected(self):
        """Delete selected server from list (if stopped)."""
        if not self.selected_server_id:
            return

        s = self.servers[self.selected_server_id]
        if s["status"] in ["Running", "Starting"]:
            self.log_to_console(
                "[System] [red]Cannot delete a running or starting server. Stop it first![/]\n"
            )
            return

        server_name = s["name"]
        server_id = self.selected_server_id

        # Remove from state
        del self.servers[server_id]
        if server_id in self.log_buffers:
            del self.log_buffers[server_id]

        self.selected_server_id = None
        self._refresh_servers_list()
        self._save_dashboard_config()

        # Log to the new selected server log or fallback
        if self.selected_server_id:
            self.log_to_console(f"[System] Deleted server '{server_name}'.\n")

    def _handle_auto_detect(self):
        """Automatically find local checkpoints, allocate devices/ports, and register them."""
        self.log_to_console("[System] Scanning for local checkpoints...\n")
        self.discovered_checkpoints = scan_checkpoints()
        self._update_checkpoint_select_options()

        # Deduplicate paths already configured
        configured_paths = {s["checkpoint"] for s in self.servers.values()}

        added_count = 0
        allocated_port = 8000

        for ckpt in self.discovered_checkpoints:
            # We ignore presets unless they are locally cached
            if ckpt["source"] == "preset":
                continue

            ckpt_path = ckpt["path"]
            if ckpt_path in configured_paths:
                continue

            # Create a server configuration
            server_id = str(uuid.uuid4())
            name_prefix = ckpt["name"].split(" (")[0]

            # Find free port starting from allocated_port
            free_port = find_free_port(allocated_port) or 8000
            allocated_port = free_port + 1

            # Allocate CUDA device round-robin
            cuda_gpus = [d for d in self.cuda_devices if d.startswith("cuda:")]
            if cuda_gpus:
                gpu_index = len(self.servers) % len(cuda_gpus)
                device = cuda_gpus[gpu_index]
            else:
                device = "cpu"

            self.servers[server_id] = {
                "id": server_id,
                "name": f"Auto-{name_prefix}",
                "checkpoint": ckpt_path,
                "policy_type": ckpt["policy_type"],
                "device": device,
                "port": free_port,
                "status": "Stopped",
            }
            self.log_buffers[server_id] = ["--- Server Created via Auto-Detection ---\n"]
            configured_paths.add(ckpt_path)
            added_count += 1

        if added_count > 0:
            self._refresh_servers_list()
            self._save_dashboard_config()
            self.log_to_console(f"[System] [green]Success: Auto-configured {added_count} new servers.[/]\n")
        else:
            self.log_to_console("[System] No new checkpoints detected that weren't already configured.\n")

    # ==============================================================================
    # Subprocess execution, threads & logs streaming
    # ==============================================================================

    def action_start_selected(self):
        """Start the selected policy server."""
        if not self.selected_server_id:
            return

        s = self.servers[self.selected_server_id]
        if s["status"] in ["Running", "Starting"]:
            self.log_to_console("[System] Server is already active.\n")
            return

        server_id = self.selected_server_id
        s["status"] = "Starting"
        self._refresh_servers_list()

        self.log_to_console(f"[System] Starting server {s['name']} on port {s['port']} ({s['device']})...\n")

        # Build command-line arguments to run policy_server.py
        server_script = Path(__file__).parent / "policy_server.py"

        cmd = [
            sys.executable,
            str(server_script),
            "--host=0.0.0.0",
            f"--port={s['port']}",
            f"--policy_type={s['policy_type']}",
            f"--pretrained_name_or_path={s['checkpoint']}",
            f"--policy_device={s['device']}",
        ]

        # We can also pass CUDA_VISIBLE_DEVICES to the environment of the process
        env = os.environ.copy()
        if s["device"].startswith("cuda:"):
            gpu_idx = s["device"].split(":")[1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
            # Note: We still pass `--policy_device=cuda` because CUDA_VISIBLE_DEVICES hides other GPUs,
            # so the policy_server will see the allocated GPU as cuda:0
            cmd[-1] = "--policy_device=cuda"

        # Spawn subprocess
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                preexec_fn=None if sys.platform == "win32" else os.setpgrp,
            )
            self.processes[server_id] = process

            # Start background thread to read process output
            thread = threading.Thread(
                target=self._read_process_output, args=(process, server_id), daemon=True
            )
            thread.start()

        except Exception as e:
            s["status"] = "Failed"
            self._refresh_servers_list()
            self.log_to_console(f"[System] [red]Failed to launch policy_server.py: {e}[/]\n")

    def action_stop_selected(self):
        """Stop the selected policy server."""
        if not self.selected_server_id:
            return

        server_id = self.selected_server_id
        s = self.servers[server_id]

        if s["status"] not in ["Running", "Starting", "Failed"]:
            self.log_to_console("[System] Server is not running.\n")
            return

        self.log_to_console(f"[System] Stopping server {s['name']}...\n")

        if server_id in self.processes:
            process = self.processes[server_id]
            try:
                process.terminate()
                # Wait briefly
                process.wait(timeout=1.5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            finally:
                if server_id in self.processes:
                    del self.processes[server_id]

        s["status"] = "Stopped"
        self._refresh_servers_list()
        self.log_to_console("[System] Server stopped.\n")

    def _read_process_output(self, process: subprocess.Popen, server_id: str):
        """Read standard output from policy_server.py process line-by-line (runs in thread)."""
        is_running = False

        while True:
            line = process.stdout.readline()
            if not line:
                break

            # If we see startup confirmations, update server status to Running
            if (
                "Policy server listen on:" in line
                or "warm up the policy." in line
                or "Connected from" in line
            ):
                if not is_running:
                    is_running = True
                    self.call_from_thread(self._update_server_status, server_id, "Running")

            # Safe update to log buffer inside Textual TUI
            self.call_from_thread(self.log_to_console, line, server_id)

        # Process has terminated
        exit_code = process.wait()

        status = "Stopped"
        if exit_code != 0 and exit_code != -15 and exit_code != -9:  # -15 is SIGTERM, -9 is SIGKILL
            status = "Failed"
            self.call_from_thread(
                self.log_to_console,
                f"[System] [red]Server process exited with non-zero code: {exit_code}[/]\n",
                server_id,
            )
        else:
            self.call_from_thread(
                self.log_to_console, f"[System] Server process exited with code: {exit_code}\n", server_id
            )

        self.call_from_thread(self._update_server_status, server_id, status)

        # Clean up process reference
        if server_id in self.processes:
            del self.processes[server_id]

    def _update_server_status(self, server_id: str, status: str):
        """Update server status in state and refresh TUI list (thread-safe UI update)."""
        if server_id in self.servers:
            self.servers[server_id]["status"] = status
            self._refresh_servers_list()


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main():
    app = PolicyDashboardApp()
    app.run()


if __name__ == "__main__":
    main()
