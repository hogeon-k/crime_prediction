import tkinter as tk

BG = "#F8F9FA"
PANEL_BG = "#FFFFFF"
ACCENT = "#4F46E5"
ACCENT_H = "#4338CA"
SUCCESS = "#059669"
DANGER = "#DC2626"
TEXT_PRI = "#111827"
TEXT_SEC = "#6B7280"
BORDER = "#E5E7EB"

FONT_TITLE = ("맑은 고딕", 14, "bold")
FONT_LABEL = ("맑은 고딕", 10)
FONT_BOLD = ("맑은 고딕", 10, "bold")
FONT_SMALL = ("맑은 고딕", 9)
FONT_MONO = ("Consolas", 9)


def card(parent: tk.Widget, **kw) -> tk.Frame:
    frame = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1, **kw)
    return frame


def label(parent, text, font=None, fg=TEXT_PRI, bg=PANEL_BG, **kw):
    return tk.Label(parent, text=text, font=font or FONT_LABEL, fg=fg, bg=bg, **kw)


def button(parent, text, command, bg=ACCENT, fg="white", font=None, **kw):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=ACCENT_H,
        activeforeground="white",
        relief="flat",
        cursor="hand2",
        font=font or FONT_BOLD,
        padx=12,
        pady=6,
        **kw,
    )
    btn.bind("<Enter>", lambda _e: btn.config(bg=ACCENT_H))
    btn.bind("<Leave>", lambda _e: btn.config(bg=bg))
    return btn
