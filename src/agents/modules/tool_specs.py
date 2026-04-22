def build_tool_specs(valid_msg_types: set[str]) -> tuple[list[dict], list[dict]]:
    child_tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run one shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file and return content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace exact text in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "Load full body of a named skill.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compact",
                "description": "Compact history with optional focus.",
                "parameters": {
                    "type": "object",
                    "properties": {"focus": {"type": "string"}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_create",
                "description": "Create a new task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["subject"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_update",
                "description": "Update a task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
                        "removeBlockedBy": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["task_id", "status", "addBlockedBy", "removeBlockedBy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_list",
                "description": "List all tasks.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_get",
                "description": "Get a task by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "background_run",
                "description": "Run a background command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_background",
                "description": "Check background task status.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_teammate",
                "description": "Spawn persistent teammate thread.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["name", "role", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_teammates",
                "description": "List teammates.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send teammate message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "content": {"type": "string"},
                        "msg_type": {"type": "string", "enum": sorted(valid_msg_types)},
                    },
                    "required": ["to", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_inbox",
                "description": "Read and drain inbox.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "broadcast",
                "description": "Broadcast message to teammates.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "shutdown_request",
                "description": "Lead requests graceful shutdown.",
                "parameters": {
                    "type": "object",
                    "properties": {"teammate": {"type": "string"}},
                    "required": ["teammate"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "shutdown_response",
                "description": "Shutdown response or status check.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "approve": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["request_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan_approval",
                "description": "Submit or review plan approval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"},
                        "request_id": {"type": "string"},
                        "approve": {"type": "boolean"},
                        "feedback": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "idle",
                "description": "Enter idle polling phase.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "claim_task",
                "description": "Claim a task id.",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_bind_worktree",
                "description": "Bind task to worktree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "worktree": {"type": "string"},
                        "owner": {"type": "string"},
                    },
                    "required": ["task_id", "worktree"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_create",
                "description": "Create worktree lane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "task_id": {"type": "integer"},
                        "base_ref": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_list",
                "description": "List worktrees.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_status",
                "description": "Show worktree git status.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_run",
                "description": "Run command in worktree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "required": ["name", "command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_keep",
                "description": "Mark worktree kept.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_remove",
                "description": "Remove worktree lane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "force": {"type": "boolean"},
                        "complete_task": {"type": "boolean"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "worktree_events",
                "description": "Read lifecycle events.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                    "required": [],
                },
            },
        },
    ]

    parent_tools = child_tools + [
        {
            "type": "function",
            "function": {
                "name": "run_subtask",
                "description": (
                    "Run a sub-investigation in fresh context and return summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]
    return child_tools, parent_tools

