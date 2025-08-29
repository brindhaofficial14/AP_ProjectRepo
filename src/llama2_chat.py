# llama2_chat.py
# A lightweight CustomTkinter desktop UI for chatting with a local GGUF (Llama 2).
# - Loads model in-process (llama-cpp-python)
# - Streams tokens with a background thread
# - New chat, Stop, Save transcript, Choose model file
# - Basic controls: temperature, top_p, context, threads, GPU layers

import os, threading, queue, json, time
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox

# pip install llama-cpp-python==0.2.90
from llama_cpp import Llama

DEFAULT_MODEL_PATH = Path("models/llama-2-7b-chat.Q4_K_M.gguf")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_CTX = 4096
DEFAULT_THREADS = 8
DEFAULT_GPU_LAYERS = 0

class LlamaChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_default_color_theme("blue")
        self.title("Llama 2 Desktop Chat")
        self.geometry("1000x650")
        self.minsize(860, 560)

        # Runtime state
        self.llm = None
        self.model_path = None
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.stream_q = queue.Queue()
        self.stream_thread = None
        self.stop_event = threading.Event()
        self.streaming = False
        self.gen_start_time = 0.0

        # Layout
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Chat area
        self.chat_box = ctk.CTkTextbox(self, wrap="word")
        self.chat_box.grid(row=0, column=0, sticky="nsew", padx=(16,8), pady=(16,8))
        self._log(f"System: {DEFAULT_SYSTEM_PROMPT}\n")

        # Sidebar controls
        side = ctk.CTkFrame(self)
        side.grid(row=0, column=1, sticky="ns", padx=(8,16), pady=(16,8))
        side.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(side, text="Model file (.gguf)").grid(sticky="ew", padx=12, pady=(12,4))
        path_row = ctk.CTkFrame(side)
        path_row.grid(sticky="ew", padx=12, pady=(0,8))
        path_row.grid_columnconfigure(0, weight=1)
        self.model_path_var = ctk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.model_entry = ctk.CTkEntry(path_row, textvariable=self.model_path_var)
        self.model_entry.grid(row=0, column=0, sticky="ew", padx=(0,6))
        ctk.CTkButton(path_row, text="Browse", width=70, command=self.choose_model).grid(row=0, column=1)

        self.load_btn = ctk.CTkButton(side, text="Load model", command=self.load_model)
        self.load_btn.grid(sticky="ew", padx=12, pady=(0,10))

        # System prompt
        ctk.CTkLabel(side, text="System prompt").grid(sticky="ew", padx=12, pady=(6,4))
        self.system_var = ctk.StringVar(value=DEFAULT_SYSTEM_PROMPT)
        self.system_entry = ctk.CTkEntry(side, textvariable=self.system_var)
        self.system_entry.grid(sticky="ew", padx=12, pady=(0,8))

        # Sliders / entries
        row = ctk.CTkFrame(side); row.grid(sticky="ew", padx=12, pady=(6,4)); row.grid_columnconfigure((0,1), weight=1)
        ctk.CTkLabel(row, text="Temperature").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row, text="Top-p").grid(row=0, column=1, sticky="w")
        self.temp = ctk.DoubleVar(value=0.6)
        self.topp = ctk.DoubleVar(value=0.95)
        ctk.CTkSlider(side, from_=0.0, to=1.0, number_of_steps=20, variable=self.temp).grid(sticky="ew", padx=12)
        ctk.CTkSlider(side, from_=0.1, to=1.0, number_of_steps=18, variable=self.topp).grid(sticky="ew", padx=12, pady=(0,8))

        # Context / threads / gpu layers / max tokens
        grid = ctk.CTkFrame(side); grid.grid(sticky="ew", padx=12, pady=6)
        for i in range(4): grid.grid_columnconfigure(i, weight=1)
        ctk.CTkLabel(grid, text="Context").grid(row=0, column=0)
        ctk.CTkLabel(grid, text="Threads").grid(row=0, column=1)
        ctk.CTkLabel(grid, text="GPU layers").grid(row=0, column=2)
        ctk.CTkLabel(grid, text="Max new").grid(row=0, column=3)
        self.ctx_var = ctk.IntVar(value=DEFAULT_CTX)
        self.thr_var = ctk.IntVar(value=DEFAULT_THREADS)
        self.gpu_var = ctk.IntVar(value=DEFAULT_GPU_LAYERS)
        self.max_new_var = ctk.IntVar(value=512)
        ctk.CTkEntry(grid, textvariable=self.ctx_var).grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ctk.CTkEntry(grid, textvariable=self.thr_var).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ctk.CTkEntry(grid, textvariable=self.gpu_var).grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        ctk.CTkEntry(grid, textvariable=self.max_new_var).grid(row=1, column=3, padx=4, pady=4, sticky="ew")

        # Session buttons
        self.new_btn = ctk.CTkButton(side, text="New chat", command=self.new_chat)
        self.new_btn.grid(sticky="ew", padx=12, pady=(8,4))
        self.save_btn = ctk.CTkButton(side, text="Save transcript", command=self.save_transcript)
        self.save_btn.grid(sticky="ew", padx=12, pady=4)
        self.stop_btn = ctk.CTkButton(side, text="Stop generation", command=self.stop_generation, state="disabled")
        self.stop_btn.grid(sticky="ew", padx=12, pady=4)

        self.status = ctk.CTkLabel(side, text="Model: not loaded", text_color=("gray20","gray70"))
        self.status.grid(sticky="ew", padx=12, pady=(8,12))

        # Bottom input row
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0,16))
        bottom.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(bottom, height=96)
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(8,8), pady=8)
        self.input_box.bind("<Shift-Return>", lambda e: self.input_box.insert("insert", "\n"))
        self.input_box.bind("<Return>", self._enter_send)

        self.send_btn = ctk.CTkButton(bottom, text="Send (Enter)", width=140, command=self.send)
        self.send_btn.grid(row=0, column=1, sticky="e", padx=(0,8), pady=8)

        # Kick off stream pump
        self.after(30, self._drain_stream)

        # Auto-load default model if present
        if DEFAULT_MODEL_PATH.exists():
            self.model_path = str(DEFAULT_MODEL_PATH.resolve())
            self.model_path_var.set(self.model_path)
            self.load_model()

    # ---------------- UI helpers ----------------
    def _log(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _log_line(self, who, text):
        self._log(f"{who}: {text}\n")

    def _enter_send(self, event):
        self.send()
        return "break"

    # ---------------- Actions ----------------
    def choose_model(self):
        fp = filedialog.askopenfilename(
            title="Choose GGUF model",
            initialdir=str(Path("models").resolve()),
            filetypes=[("GGUF models","*.gguf"), ("All files","*.*")]
        )
        if fp:
            self.model_path_var.set(fp)

    def load_model(self):
        if self.streaming:
            messagebox.showinfo("Busy", "Please stop/finish generation before loading a model.")
            return
        path = Path(self.model_path_var.get()).expanduser()
        if not path.exists():
            messagebox.showerror("Not found", f"Model file not found:\n{path}")
            return
        self.status.configure(text="Loading model… this can take a moment")
        self.update_idletasks()
        try:
            # Create Llama with Llama-2 chat format so messages[] just works
            self.llm = Llama(
                model_path=str(path),
                n_ctx=int(self.ctx_var.get()),
                n_gpu_layers=int(self.gpu_var.get()),
                n_threads=int(self.thr_var.get()),
                chat_format="llama-2",
                verbose=False,
            )
            self.model_path = str(path)
            self.status.configure(text=f"Loaded: {path.name}")
        except Exception as e:
            self.llm = None
            messagebox.showerror("Load error", f"Failed to load model:\n{e}")
            self.status.configure(text="Model: not loaded")

    def new_chat(self):
        if self.streaming:
            messagebox.showinfo("Busy", "Please stop/finish generation first.")
            return
        sysmsg = self.system_var.get().strip() or DEFAULT_SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": sysmsg}]
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0","end")
        self.chat_box.configure(state="disabled")
        self._log_line("System", sysmsg)
        self.status.configure(text="New chat started")

    def save_transcript(self):
        fp = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=[("Text","*.txt")]
        )
        if not fp:
            return
        text = self.chat_box.get("1.0","end")
        Path(fp).write_text(text, encoding="utf-8")
        self.status.configure(text=f"Saved: {Path(fp).name}")

    def stop_generation(self):
        if self.streaming:
            self.stop_event.set()
            self.status.configure(text="Stopping… (may take a moment)")
            self.stop_btn.configure(state="disabled")

    def send(self):
        if self.llm is None:
            messagebox.showinfo("No model", "Load a GGUF model first.")
            return
        if self.streaming:
            return
        user_text = self.input_box.get("1.0", "end").strip()
        if not user_text:
            return

        # sync system message if user changed it since start
        if not self.messages or self.messages[0]["role"] != "system" or self.messages[0]["content"] != self.system_var.get().strip():
            self.messages[0] = {"role":"system","content": self.system_var.get().strip() or DEFAULT_SYSTEM_PROMPT}

        self.input_box.delete("1.0","end")
        self.messages.append({"role":"user","content":user_text})
        self._log_line("You", user_text)
        self._log("Assistant: ")
        self.streaming = True
        self.stop_event.clear()
        self.gen_start_time = time.time()
        self.status.configure(text="Generating…")
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        # background thread
        snapshot = list(self.messages)  # copy for this turn
        self.stream_thread = threading.Thread(target=self._generate, args=(snapshot,))
        self.stream_thread.daemon = True
        self.stream_thread.start()

    # ---------------- Generation (background) ----------------
    def _generate(self, messages_snapshot):
        temp = float(self.temp.get())
        topp = float(self.topp.get())
        max_new = max(1, int(self.max_new_var.get()))
        collected = []

        try:
            for ev in self.llm.create_chat_completion(
                messages=messages_snapshot,
                temperature=temp,
                top_p=topp,
                max_tokens=max_new,
                stream=True,
            ):
                if self.stop_event.is_set():
                    break
                delta = ev.get("choices",[{}])[0].get("delta",{}).get("content","")
                if delta:
                    collected.append(delta)
                    self.stream_q.put(delta)
        except Exception as e:
            self.stream_q.put(f"\n[Generation error: {e}]")

        # finalize
        text = "".join(collected)
        self.stream_q.put(f"\n__END__{text}")

    # ---------------- UI pump for stream ----------------
    def _drain_stream(self):
        try:
            while True:
                chunk = self.stream_q.get_nowait()
                if chunk.startswith("\n__END__"):
                    content = chunk.replace("\n__END__","",1)
                    # Close the assistant line
                    self._log("\n")
                    # Record in history
                    self.messages.append({"role":"assistant","content":content})
                    self.streaming = False
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    dur = time.time() - self.gen_start_time if self.gen_start_time else 0.0
                    self.status.configure(text=f"Done in {dur:.1f}s")
                else:
                    # print incrementally on same line
                    self._log(chunk)
        except queue.Empty:
            pass
        # keep pumping
        self.after(25, self._drain_stream)

if __name__ == "__main__":
    app = LlamaChatApp()
    app.mainloop()
