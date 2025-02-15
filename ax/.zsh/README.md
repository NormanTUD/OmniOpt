# What is this?

This allows you to do easy autocompletion for lots of the OmniOpt2 programs in ZSH.

## Installation

```
cp omniopt_compdefs $HOME/.omniopt_compdefs
grep -qxF "source .omniopt_compdefs" "$HOME/.zshrc" || echo "source .omniopt_compdefs" >> "$HOME/.zshrc"
```
