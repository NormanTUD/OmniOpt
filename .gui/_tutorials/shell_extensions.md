# ⌨️ Shell-Extensions `bash`, `zsh`

<!-- Extensions for tab-completion for ZSH and Bash-Shells -->

<!-- Category: Advanced Usage -->

<div id="toc"></div>

## What is Tab Completion?

Tab completion is a feature in shells like **Bash** and **Zsh** that allows you to quickly complete commands, filenames, and parameters by pressing the **Tab** key. This saves time, reduces typing errors, and helps explore available options more efficiently.

## Enabling Tab Completion for OmniOpt

To enable tab completion for **OmniOpt**, follow these steps:

### 1. Install the Completion Script

Run the following command in your terminal:

```bash
bash .shells/install
```

### 2. Restart Your Shell

After running the installation command, restart your shell by either:

- Closing and reopening your terminal
- Running the following command:

```bash
exec "$SHELL"
```

## Using Tab Completion

Once installed, you can tab-complete all parameters of **OmniOpt** in both **Bash** and **Zsh**. Simply type part of a command and press **Tab** to see available options.

Example:

```bash
omniopt --[Tab]
```

This will display all available options for **OmniOpt**.

## Troubleshooting
<div class="caveat tip">
If tab completion is not working, try the following:
- Ensure you have restarted your shell.
- Check that the `.shells/install` script ran successfully.

Enjoy faster and more efficient command-line usage with tab completion!
</div>
