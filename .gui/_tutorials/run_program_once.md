# 📝 The `--run_program_once` parameter

<div id="toc"></div>

<!-- Install programs and prepare environment before you install OmniOpt2 -->
<!-- Category: Preparations, Basics and Setup -->

The parameter `run_program_once` 🐚 allows you to run a shell command or script **once before** your actual program starts.

Use it to:
- ⚙️ install dependencies  
- 🌐 download datasets  
- 🧹 prepare folders  
- 🔧 configure environments

## 💡 How to use variables

You can use variable placeholders like `%(lr)` or `%(epochs)` that will be replaced with actual values during the run.

```bash
bash /absolute/path/to/install.sh --lr=%(lr) --epochs=%(epochs)
```

## 📁 Example: `install.sh`

Here’s an example `install.sh` you might call with `run_program_once`:

```bash
#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading dataset..."
wget https://example.com/dataset.zip
unzip dataset.zip -d ./data

echo "Done ✅"
```

## 💡 Tips

<div class="caveat warning">
- This runs **once per experiment**, not per trial.
- Great for setup work you don’t want repeated.
- You can call any Bash-compatible command or script.
</div>

```bash
# Example with absolute path and custom args
bash /my/install.sh --batch-size=%(batch_size)
```
