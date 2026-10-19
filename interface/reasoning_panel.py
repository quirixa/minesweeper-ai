"""Separate window showing the AI's current reasoning, a reasoning log and the move history."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

BG, FG, ACCENT, MUTED = "#15181e", "#e6e9ef", "#4fc3f7", "#8b93a1"


class ReasoningPanel(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("AI reasoning")
        self.geometry("560x720")
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)  # closing only hides it
        tk.Label(self, text="Current reasoning", bg=BG, fg=MUTED, font=("TkDefaultFont", 9)).pack(anchor="w", padx=12, pady=(10, 0))
        self.current = tk.Label(self, text="Idle. Calibrate, then press Start.", bg=BG, fg=ACCENT, justify="left",
                                wraplength=520, font=("TkDefaultFont", 12, "bold"), anchor="w")
        self.current.pack(fill="x", padx=12, pady=(2, 8))
        tk.Label(self, text="Reasoning log", bg=BG, fg=MUTED).pack(anchor="w", padx=12)
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True, padx=12, pady=4)
        self.log = tk.Text(frame, bg="#0f1115", fg=FG, wrap="word", height=14, relief="flat", state="disabled",
                           insertbackground=FG)
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("guess", foreground="#ffb74d")
        self.log.tag_configure("warn", foreground="#ef5350")
        self.log.tag_configure("time", foreground=MUTED)
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(top, text="Move history (this session)", bg=BG, fg=MUTED).pack(side="left")
        ttk.Button(top, text="Clear history", command=self.clear_history).pack(side="right")
        cols = ("time", "action", "cell", "type", "risk", "outcome")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=9)
        for c, w in zip(cols, (70, 60, 80, 90, 60, 90)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def set_current(self, text: str) -> None:
        self.current.configure(text=text)

    def add_reasoning(self, text: str) -> None:
        tag = "warn" if any(k in text.lower() for k in ("too low", "contradict", "pausing", "lost", "emergency")) \
            else ("guess" if "GUESS" in text else None)
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S "), "time")
        self.log.insert("end", text + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")
        self.set_current(text)

    def add_history(self, data: dict) -> None:
        r, c = data["cell"]
        p = data.get("probability")
        self.tree.insert("", "end", values=(
            time.strftime("%H:%M:%S"), data["action"], f"r{r+1} c{c+1}",
            "certain" if data["certain"] else "GUESS", "-" if p is None else f"{p:.0%}", data["outcome"]))
        self.tree.yview_moveto(1)

    def clear_history(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
